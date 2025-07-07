import os
import asyncio
import json
import re
import time
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import ChatHistory, FunctionResultContent, StreamingChatMessageContent
from semantic_kernel.functions import KernelArguments
from .base_framework import BaseFramework
from backend.db_manager import DBManager

# Import our unified logging system
from backend.utils.simple_logger import (
    setup_simple_logger, 
    ExecutionTimer, 
    log_execution_time,
    log_tokens,
    log_user_input,
    log_tool_usage
)

from dotenv import load_dotenv
load_dotenv()


class UserInfo(BaseModel):
    """Information about a user."""
    name: Optional[str] = Field(default=None, description="The name of the user")
    email: Optional[str] = Field(default=None, description="The email address of the user")
    age: Optional[int] = Field(default=None, description="The age of the user")


class ProductInfo(BaseModel):
    """Information about a product purchase."""
    name: Optional[str] = Field(default=None, description="The name of the product")
    number: Optional[int] = Field(default=1, description="The number of products to purchase")


class SemanticKernelPlugin:
    """Semantic Kernel plugin for SQLite database operations"""
    
    def __init__(self, db_manager: DBManager, kernel: Kernel, chat_completion, logger):
        self.db_manager = db_manager
        self.kernel = kernel
        self.chat_completion = chat_completion
        self.logger = logger  # Add logger reference

    async def _extract_info(self, text: str, info_type: str, model_class: BaseModel) -> BaseModel:
        """Extract structured information using Semantic Kernel with LLM structured output"""
        start_time = time.time()
        self.logger.info(f"Starting structured extraction for {info_type}")
        
        try:
            # Create prompt for information extraction
            if info_type == "user":
                prompt = f"""
Extract user information from the following text. If information is not available, set the field to null.

Text: {text}

Extract the following information:
- name: The person's name
- email: The person's email address 
- age: The person's age (as a number)
"""
            elif info_type == "product":
                prompt = f"""
Extract product purchase information from the following text. If information is not available, set the field to null.

Text: {text}

Extract the following information:
- name: The product name (should be one of: Smartphone, Laptop, Headphones)
- number: The quantity to purchase (as a number, default to 1 if not specified)
"""
            else:
                return model_class()

            # Get request settings and set structured output format
            request_settings = self.chat_completion.get_prompt_execution_settings_class()(
                service_id="semantic_kernel",
                max_tokens=500,
                temperature=0.1,
                response_format=model_class  # This enables structured output
            )

            # Create extraction function with structured output
            extraction_function = self.kernel.add_function(
                prompt=prompt,
                function_name=f"extract_{info_type}_{id(self)}",
                plugin_name="extraction_temp",
                prompt_execution_settings=request_settings,
            )

            # Invoke the function
            result = await self.kernel.invoke(extraction_function)
            
            execution_time = time.time() - start_time
            self.logger.info(f"Extraction completed in {execution_time:.2f}s")
            
            if hasattr(result, 'value') and result.value:
                if isinstance(result.value, list) and len(result.value) > 0:
                    content = result.value[0].content
                else:
                    content = str(result.value)
                
                # Parse JSON and validate with model
                try:
                    parsed_data = json.loads(content)
                    return model_class.model_validate(parsed_data)
                except (json.JSONDecodeError, ValueError):
                    # If JSON parsing fails, try to extract manually as fallback
                    return self._fallback_extraction(text, info_type, model_class)
            else:
                return model_class()
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Structured extraction failed after {execution_time:.2f}s: {e}")
            # Fallback to manual extraction
            return self._fallback_extraction(text, info_type, model_class)

    def _fallback_extraction(self, text: str, info_type: str, model_class: BaseModel) -> BaseModel:
        """Fallback extraction method using simple parsing"""
        try:
            data = {}
            if info_type == "user":
                # Extract user info using simple parsing
                # Extract name - look for common patterns
                name_patterns = [
                    r'(?:add|member|user|named)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
                    r'([A-Za-z]+(?:\s+[A-Za-z]+)?)\s*,',
                    r'name\s*:?\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)'
                ]
                
                for pattern in name_patterns:
                    name_match = re.search(pattern, text, re.IGNORECASE)
                    if name_match:
                        data["name"] = name_match.group(1).strip()
                        break
                
                # Extract email
                email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
                if email_match:
                    data["email"] = email_match.group()
                
                # Extract age
                age_match = re.search(r'\b(\d{1,3})\s*(?:years?\s*old|age)\b', text, re.IGNORECASE)
                if age_match:
                    data["age"] = int(age_match.group(1))
            
            elif info_type == "product":
                # Extract product info
                # Extract product name
                product_names = ["smartphone", "laptop", "headphones"]
                for product in product_names:
                    if product in text.lower():
                        data["name"] = product.title()
                        break
                
                # Extract quantity
                qty_match = re.search(r'\b(\d+)\s*(?:smartphone|laptop|headphones)', text, re.IGNORECASE)
                if qty_match:
                    data["number"] = int(qty_match.group(1))
                elif "buy" in text.lower() or "purchase" in text.lower():
                    data["number"] = 1  # Default to 1 if purchase intent is detected
            
            return model_class.model_validate(data)
        except Exception:
            return model_class()

    @kernel_function(
        name="add_member",
        description="Add a new member to the database. Use this when the user wants to create, add, or register a new member/user/person. Extract name, email and age from user input."
    )
    async def add_member(self, user_input: str) -> str:
        """Add a new member by extracting info from natural language."""
        log_tool_usage(self.logger, "add_member", user_input)
        
        try:
            user_info = await self._extract_info(user_input, "user", UserInfo)
            
            if not user_info.name:
                result = "❌ Could not extract user name from input."
                self.logger.error("Tool failed: add_member - no name extracted")
                return result

            existing_member = self.db_manager.get_member_by_name(user_info.name)
            if existing_member:
                result = f"Member {user_info.name} already exists with ID: {existing_member[0]}"
                self.logger.info("Tool completed: add_member - member exists")
                return result

            self.db_manager.insert_member(
                user_info.name, 
                user_info.email or "unknown@example.com", 
                user_info.age or 0
            )
            
            new_member = self.db_manager.get_member_by_name(user_info.name)
            result = f"✅ Successfully added member: {user_info.name} (ID: {new_member[0]})"
            self.logger.info(f"Tool completed: add_member - {user_info.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Tool failed: add_member - {e}")
            return f"❌ Error adding member: {str(e)}"

    @kernel_function(
        name="make_purchase",
        description="Process a purchase transaction. Use this when the user wants to buy, purchase, or order products. Extract user name and product information from user input."
    )
    async def make_purchase(self, purchase_input: str) -> str:
        """Process a purchase by extracting user and product info from natural language."""
        log_tool_usage(self.logger, "make_purchase", purchase_input)
        
        try:
            user_info = await self._extract_info(purchase_input, "user", UserInfo)
            product_info = await self._extract_info(purchase_input, "product", ProductInfo)
            
            if not user_info.name:
                self.logger.error("Tool failed: make_purchase - no user name")
                return "❌ Could not extract user name from input."
            if not product_info.name:
                self.logger.error("Tool failed: make_purchase - no product name")
                return "❌ Could not extract product name from input."

            member = self.db_manager.get_member_by_name(user_info.name)
            if not member:
                self.logger.error(f"Tool failed: make_purchase - member {user_info.name} not found")
                return f"❌ Member '{user_info.name}' not found. Please add member first."

            product = self.db_manager.get_product_by_name(product_info.name)
            if not product:
                self.logger.error(f"Tool failed: make_purchase - product {product_info.name} not found")
                return f"❌ Product '{product_info.name}' not found."
            
            member_id = member[0]
            product_id = product[0]
            quantity = product_info.number or 1
            
            self.db_manager.insert_record(member_id, product_id, quantity)
            
            result = f"✅ Purchase successful! {user_info.name} bought {quantity} {product_info.name}(s)."
            self.logger.info(f"Tool completed: make_purchase - {user_info.name} bought {product_info.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Tool failed: make_purchase - {e}")
            return f"❌ Error processing purchase: {str(e)}"

    @kernel_function(
        name="get_purchase_history",
        description="Retrieve purchase history for a specific member. Use this when the user asks about purchase records, transaction history, or what someone bought."
    )
    async def get_purchase_history(self, user_input: str) -> str:
        """Get purchase history by extracting member name from natural language."""
        log_tool_usage(self.logger, "get_purchase_history", user_input)
        
        try:
            user_info = await self._extract_info(user_input, "user", UserInfo)
            
            if not user_info.name:
                self.logger.error("Tool failed: get_purchase_history - no user name")
                return "❌ Could not extract user name from input."
            
            member = self.db_manager.get_member_by_name(user_info.name)
            if not member:
                self.logger.error(f"Tool failed: get_purchase_history - member {user_info.name} not found")
                return f"❌ Member '{user_info.name}' not found."
            
            member_id = member[0]
            records = self.db_manager.get_member_records(member_id)
            
            if not records:
                result = f"No purchase records found for {user_info.name}."
            else:
                response = f"📋 Purchase records for {user_info.name}:\n"
                for record in records:
                    response += f"• Product: {record[1]}, Price: ${record[2]}, Quantity: {record[3]}, Total: ${record[4]}\n"
                result = response
            
            self.logger.info(f"Tool completed: get_purchase_history - {user_info.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Tool failed: get_purchase_history - {e}")
            return f"❌ Error retrieving records: {str(e)}"

    @kernel_function(
        name="view_all_members",
        description="Display all members in the database with their information (name, email, age). Use this when the user wants to see all members, list members, or get member information."
    )
    async def view_all_members(self) -> str:
        """Return all members from the SQLite database."""
        log_tool_usage(self.logger, "view_all_members")
        
        try:
            members = self.db_manager.list_all_members()
            result = f"👥 All Members:\n{members.to_string(index=False)}"
            self.logger.info("Tool completed: view_all_members")
            return result
        except Exception as e:
            self.logger.error(f"Tool failed: view_all_members - {e}")
            return f"❌ Error retrieving members: {str(e)}"

    @kernel_function(
        name="view_all_products",
        description="Display all products available in the database with their information (name, price). Use this when the user wants to see all products, list products, or get product information."
    )
    async def view_all_products(self) -> str:
        """Return all products from the SQLite database."""
        log_tool_usage(self.logger, "view_all_products")
        
        try:
            products = self.db_manager.list_all_products()
            result = f"📦 All Products:\n{products.to_string(index=False)}"
            self.logger.info("Tool completed: view_all_products")
            return result
        except Exception as e:
            self.logger.error(f"Tool failed: view_all_products - {e}")
            return f"❌ Error retrieving products: {str(e)}"


class SemanticKernelFramework(BaseFramework):
    """Semantic Kernel framework implementation with Azure OpenAI"""
    
    def __init__(self):
        # Setup unified logger
        self.logger = setup_simple_logger("SemanticKernel")
        self.kernel = None
        self.chat_completion = None
        self.plugin = None
        self.logger.info("Semantic Kernel framework initialized")

    def _initialize_kernel(self):
        """Initialize Semantic Kernel with Azure OpenAI"""
        if self.kernel is None:
            with ExecutionTimer(self.logger, "Kernel initialization"):
                self.kernel = Kernel()
                
                # Add Azure OpenAI Chat Completion service
                self.chat_completion = AzureChatCompletion(
                    service_id="semantic_kernel",
                    deployment_name=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                    endpoint=os.getenv("AZURE_ENDPOINT"),
                    api_key=os.getenv("AZURE_API_KEY"),
                    api_version=os.getenv("AZURE_API_VERSION", "2025-04-01-preview"),
                )
                
                self.kernel.add_service(self.chat_completion)
                
                # Initialize database manager and plugin with logger
                db_manager = DBManager("customer_database.db")
                self.plugin = SemanticKernelPlugin(db_manager, self.kernel, self.chat_completion, self.logger)
                
                # Add plugin to kernel
                self.kernel.add_plugin(self.plugin, plugin_name="SQLitePlugin")
        
        return self.kernel

    @log_execution_time("Agent Creation")
    def create_agent(self, tools: List, system_prompt: str):
        """Create Semantic Kernel agent with database plugin and function calling"""
        self.logger.info("Semantic Kernel starting agent creation")
        
        kernel = self._initialize_kernel()
        
        # Store system prompt for later use
        self.system_prompt = system_prompt
        
        # Create execution settings with function choice behavior
        execution_settings = self.chat_completion.get_prompt_execution_settings_class()(
            service_id="semantic_kernel",
            max_tokens=1000,
            temperature=0.7,
            function_choice_behavior=FunctionChoiceBehavior.Auto()  # Let model choose functions automatically
        )
        
        # Create the main chat function that can call tools
        self.chat_function = kernel.add_function(
            prompt=system_prompt + """

Available tools:
- add_member: Add new members to the database
- make_purchase: Process purchase transactions
- get_purchase_history: Get purchase history for specific members
- view_all_members: Show all members and their information
- view_all_products: Show all available products

Use these tools when appropriate based on the user's request. If the user asks about member information, ages, or wants to see member details, use view_all_members.

{{$chat_history}}
User: {{$user_input}}
Assistant: """,
            function_name="chat_with_tools",
            plugin_name="chat",
            prompt_execution_settings=execution_settings,
        )
        
        self.logger.info("Semantic Kernel agent created successfully")
        return kernel

    def stream_execute(self, agent, message: str):
        """Execute using Semantic Kernel's native streaming and function calling"""
        start_time = time.time()
        
        # Log user input
        log_user_input(self.logger, message)
        self.logger.info("Semantic Kernel starting stream execution")
        
        try:
            async def execute_with_streaming():
                try:
                    # Create chat history for context
                    chat_history = ChatHistory()
                    
                    # Track token usage
                    total_prompt_tokens = 0
                    total_completion_tokens = 0
                    
                    # Use streaming invoke to get function call visibility
                    stream = self.kernel.invoke_stream(
                        self.chat_function,
                        user_input=message,
                        chat_history=chat_history
                    )
                    
                    function_calls = []
                    function_results = []
                    chat_content = []
                    
                    # Process streaming results
                    async for chunk in stream:
                        if isinstance(chunk, list):
                            for item in chunk:
                                # Check if item has metadata with token usage
                                if hasattr(item, 'metadata') and item.metadata:
                                    if item.metadata['usage']:
                                        total_prompt_tokens += item.metadata['usage'].prompt_tokens
                                        total_completion_tokens += item.metadata['usage'].completion_tokens
                                
                                await self._process_stream_item(item, function_calls, function_results, chat_content)
                        else:
                            # Check chunk for metadata
                            if hasattr(chunk, 'metadata') and chunk.metadata and item.metadata['usage'] != None:
                                usage = chunk.metadata['usage']
                                total_prompt_tokens += usage.prompt_tokens
                                total_completion_tokens += usage.completion_tokens
                            
                            await self._process_stream_item(chunk, function_calls, function_results, chat_content)
                    
                    # Log token usage if available
                    if total_prompt_tokens > 0 or total_completion_tokens > 0:
                        log_tokens(
                            "SemanticKernel",
                            prompt_tokens=total_prompt_tokens,
                            completion_tokens=total_completion_tokens,
                            total_tokens=total_prompt_tokens + total_completion_tokens
                        )
                    
                    return function_calls, function_results, chat_content
                        
                except Exception as e:
                    self.logger.error(f"Error in streaming execution: {e}")
                    return [], [], [f"Error executing with streaming: {str(e)}"]
            
            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                function_calls, function_results, chat_content = loop.run_until_complete(execute_with_streaming())
                
                # Emit results in LangGraph-compatible format
                if function_calls:
                    # Step 1: Agent decides to use tool(s)
                    for func_call in function_calls:
                        yield {
                            "agent": {
                                "messages": [{
                                    "content": "",
                                    "tool_calls": [{"name": func_call}]
                                }]
                            }
                        }
                
                if function_results:
                    # Step 2: Tool execution results
                    for i, result in enumerate(function_results):
                        tool_name = function_calls[i] if i < len(function_calls) else "unknown"
                        yield {
                            "tools": {
                                "messages": [{
                                    "name": tool_name,
                                    "content": result
                                }]
                            }
                        }
                
                # Step 3: Final agent response
                final_content = "".join(chat_content) if chat_content else "I've processed your request."
                yield {
                    "agent": {
                        "messages": [{
                            "content": final_content,
                            "tool_calls": None
                        }]
                    }
                }
                
                execution_time = time.time() - start_time
                self.logger.info(f"Semantic Kernel completed execution in {execution_time:.2f}s")
                    
            finally:
                loop.close()
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Semantic Kernel execution failed after {execution_time:.2f}s: {e}")
            yield {
                "agent": {
                    "messages": [{
                        "content": f"I encountered an error: {str(e)}",
                        "tool_calls": None
                    }]
                }
            }
    
    async def _process_stream_item(self, item, function_calls, function_results, chat_content):
        """Process individual streaming items to extract function calls and results"""
        try:
            if isinstance(item, StreamingChatMessageContent):
                # Check for function calls in the streaming content
                if hasattr(item, 'items') and item.items:
                    for sub_item in item.items:
                        if isinstance(sub_item, FunctionResultContent):
                            # Found a function result
                            if hasattr(sub_item, 'function_name'):
                                function_calls.append(sub_item.function_name)
                            if hasattr(sub_item, 'result'):
                                function_results.append(str(sub_item.result))
                
                # Collect chat content
                if hasattr(item, 'content') and item.content:
                    # Filter out function call content from chat
                    content = str(item.content)
                    if not any(func_pattern in content.lower() for func_pattern in ['calling', 'function', 'tool']):
                        chat_content.append(content)
            
            elif hasattr(item, 'content'):
                # Regular content item
                chat_content.append(str(item.content))
                
        except Exception as e:
            self.logger.error(f"Error processing stream item: {e}")
            # Continue processing even if one item fails

    def get_framework_name(self) -> str:
        return "Semantic Kernel"

    def get_framework_info(self) -> Dict[str, Any]:
        return {
            "name": "Semantic Kernel",
            "description": "Microsoft Semantic Kernel with Azure OpenAI for enterprise AI applications",
            "model": "GPT-4o via Azure OpenAI",
            "features": [
                "🧠 Enterprise-grade AI Orchestration",
                "🔌 Plugin-based Architecture", 
                "💬 Chat Completion Support",
                "🔗 Azure OpenAI Integration",
                "⚡ Kernel Function Decorators",
                "📊 SQLite Database Integration",
                "🎯 Structured Information Extraction"
            ],
            "dependencies": ["semantic-kernel", "azure-openai", "pydantic"]
        }