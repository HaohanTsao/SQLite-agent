import os
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field
import json
import asyncio
import time
import logging
from autogen.agentchat.utils import gather_usage_summary
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.logging import LLMCallEvent
from autogen_core import EVENT_LOGGER_NAME
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from autogen_core.models import UserMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
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

class LLMUsageTracker(logging.Handler):
    """AutoGen 官方的 LLM Usage Tracker"""
    def __init__(self) -> None:
        super().__init__()
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    def reset(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if isinstance(record.msg, LLMCallEvent):
                event = record.msg
                self._prompt_tokens += event.prompt_tokens
                self._completion_tokens += event.completion_tokens
        except Exception:
            self.handleError(record)

class AutoGenFramework(BaseFramework):
    """AutoGen framework implementation with Structured Output"""

    def __init__(self):
        # Setup unified logger
        self.logger = setup_simple_logger("AutoGen")
        self.agent = None
        self.model_client = None
        self.extraction_client = None
        
        # Setup AutoGen's official LLM usage tracker
        self.llm_usage_tracker = LLMUsageTracker()
        autogen_logger = logging.getLogger(EVENT_LOGGER_NAME)
        autogen_logger.setLevel(logging.INFO)
        autogen_logger.handlers = [self.llm_usage_tracker]
        
        self.logger.info("AutoGen framework initialized")
    
    def _initialize_model_client(self):
        """Initialize Azure OpenAI model client"""
        if self.model_client is None:
            with ExecutionTimer(self.logger, "Azure OpenAI client initialization"):
                self.model_client = AzureOpenAIChatCompletionClient(
                    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                    api_key=os.getenv("AZURE_API_KEY"),
                    azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                    api_version=os.getenv("AZURE_API_VERSION", "2025-04-01-preview"),
                    model='gpt-4o-2024-11-20',
                    temperature=0,
                )

        if self.extraction_client is None:
            self.extraction_client = AzureOpenAIChatCompletionClient(
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_API_KEY"),
                azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                api_version=os.getenv("AZURE_API_VERSION", "2025-04-01-preview"),
                model='gpt-4o-2024-11-20',
                temperature=0,
            )
        
        return self.model_client

    async def _extract_structured_data(self, text: str, model_class: BaseModel) -> BaseModel:
        start_time = time.time()
        self.logger.info(f"Starting structured data extraction for {model_class.__name__}")
        
        try:
            messages = [UserMessage(content=f"Extract information from this text: {text}", source="user")]
            
            response = await self.extraction_client.create(
                messages=messages,
                extra_create_args={"response_format": model_class}
            )
            
            execution_time = time.time() - start_time
            
            if response.content:
                data = json.loads(response.content)
                result = model_class.model_validate(data)
                self.logger.info(f"Structured extraction completed in {execution_time:.2f}s")
                
                return result
            else:
                self.logger.warning(f"No content in extraction response after {execution_time:.2f}s")
                return model_class()
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Structured extraction failed after {execution_time:.2f}s: {e}")
            return model_class()    
  
    def _create_autogen_tools(self) -> List[FunctionTool]:
        """Create AutoGen tools with natural language understanding"""
        db_manager = DBManager("customer_database.db")
        
        def extract_and_add_member(user_input: str) -> str:
            """Add a new member by extracting info from natural language."""
            log_tool_usage(self.logger, "extract_and_add_member", user_input)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    user_info = loop.run_until_complete(
                        self._extract_structured_data(user_input, UserInfo)
                    )
                finally:
                    loop.close()
                
                if not user_info.name:
                    result = "Could not extract user name from input."
                    self.logger.error("Tool failed: extract_and_add_member - no name extracted")
                    return result

                existing_member = db_manager.get_member_by_name(user_info.name)
                if existing_member:
                    result = f"Member {user_info.name} already exists with ID: {existing_member[0]}"
                    self.logger.info(f"Tool completed: extract_and_add_member - member exists")
                    return result

                db_manager.insert_member(
                    user_info.name, 
                    user_info.email or "unknown@example.com", 
                    user_info.age or 0
                )
                
                new_member = db_manager.get_member_by_name(user_info.name)
                result = f"Successfully added member: {user_info.name} (ID: {new_member[0]})"
                self.logger.info(f"Tool completed: extract_and_add_member - {user_info.name}")
                return result
                
            except Exception as e:
                self.logger.error(f"Tool failed: extract_and_add_member - {e}")
                return f"Error adding member: {str(e)}"
        
        def extract_and_make_purchase(purchase_input: str) -> str:
            """Process a purchase by extracting user and product info from natural language."""
            log_tool_usage(self.logger, "extract_and_make_purchase", purchase_input)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    user_info = loop.run_until_complete(
                        self._extract_structured_data(purchase_input, UserInfo)
                    )
                    product_info = loop.run_until_complete(
                        self._extract_structured_data(purchase_input, ProductInfo)
                    )
                finally:
                    loop.close()
                
                if not user_info.name:
                    self.logger.error("Tool failed: extract_and_make_purchase - no user name")
                    return "Could not extract user name from input."
                if not product_info.name:
                    self.logger.error("Tool failed: extract_and_make_purchase - no product name")
                    return "Could not extract product name from input."

                member = db_manager.get_member_by_name(user_info.name)
                if not member:
                    self.logger.error(f"Tool failed: extract_and_make_purchase - member {user_info.name} not found")
                    return f"Member '{user_info.name}' not found. Please add member first."

                product = db_manager.get_product_by_name(product_info.name)
                if not product:
                    self.logger.error(f"Tool failed: extract_and_make_purchase - product {product_info.name} not found")
                    return f"Product '{product_info.name}' not found."
                
                member_id = member[0]
                product_id = product[0]
                quantity = product_info.number or 1
                
                db_manager.insert_record(member_id, product_id, quantity)
                
                result = f"Purchase successful! {user_info.name} bought {quantity} {product_info.name}(s)."
                self.logger.info(f"Tool completed: extract_and_make_purchase - {user_info.name} bought {product_info.name}")
                return result
                
            except Exception as e:
                self.logger.error(f"Tool failed: extract_and_make_purchase - {e}")
                return f"Error processing purchase: {str(e)}"
        
        def get_member_purchase_history(user_input: str) -> str:
            """Get purchase history by extracting member name from natural language."""
            log_tool_usage(self.logger, "get_member_purchase_history", user_input)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    user_info = loop.run_until_complete(
                        self._extract_structured_data(user_input, UserInfo)
                    )
                finally:
                    loop.close()
                
                if not user_info.name:
                    self.logger.error("Tool failed: get_member_purchase_history - no user name")
                    return "Could not extract user name from input."
                
                member = db_manager.get_member_by_name(user_info.name)
                if not member:
                    self.logger.error(f"Tool failed: get_member_purchase_history - member {user_info.name} not found")
                    return f"Member '{user_info.name}' not found."
                
                member_id = member[0]
                records = db_manager.get_member_records(member_id)
                
                if not records:
                    result = f"No purchase records found for {user_info.name}."
                else:
                    response = f"Purchase records for {user_info.name}:\n"
                    for record in records:
                        response += f"• Product: {record[1]}, Price: ${record[2]}, Quantity: {record[3]}, Total: ${record[4]}\n"
                    result = response
                
                self.logger.info(f"Tool completed: get_member_purchase_history - {user_info.name}")
                return result
                
            except Exception as e:
                self.logger.error(f"Tool failed: get_member_purchase_history - {e}")
                return f"Error retrieving records: {str(e)}"
        
        def view_all_members() -> str:
            """Return all members from the SQLite database."""
            log_tool_usage(self.logger, "view_all_members")
            
            try:
                members = db_manager.list_all_members()
                result = f"All Members:\n{members.to_string(index=False)}"
                self.logger.info("Tool completed: view_all_members")
                return result
            except Exception as e:
                self.logger.error(f"Tool failed: view_all_members - {e}")
                return f"Error retrieving members: {str(e)}"
        
        def view_all_products() -> str:
            """Return all products from the SQLite database."""
            log_tool_usage(self.logger, "view_all_products")
            
            try:
                products = db_manager.list_all_products()
                result = f"All Products:\n{products.to_string(index=False)}"
                self.logger.info("Tool completed: view_all_products")
                return result
            except Exception as e:
                self.logger.error(f"Tool failed: view_all_products - {e}")
                return f"Error retrieving products: {str(e)}"
        
        tools = [
            FunctionTool(
                extract_and_add_member,
                name="add_member_nl",
                description="Add a new member by understanding natural language input like 'Add John, 25 years old, email john@example.com'"
            ),
            FunctionTool(
                extract_and_make_purchase,
                name="make_purchase_nl",
                description="Process a purchase by understanding natural language like 'John wants to buy 2 smartphones'"
            ),
            FunctionTool(
                get_member_purchase_history,
                name="get_history_nl",
                description="Get purchase history by understanding natural language like 'What did John buy?'"
            ),
            FunctionTool(
                view_all_members,
                name="view_members", 
                description="View all members in the database"
            ),
            FunctionTool(
                view_all_products,
                name="view_products",
                description="View all products in the database"
            )
        ]
        
        self.logger.info(f"Created {len(tools)} AutoGen tools")
        return tools
    
    @log_execution_time("Agent Creation")
    def create_agent(self, tools: List, system_prompt: str):
        """Create AutoGen agent with structured output capabilities"""
        self.logger.info("AutoGen starting agent creation")
        
        model_client = self._initialize_model_client()
        autogen_tools = self._create_autogen_tools()
        
        self.agent = AssistantAgent(
            name="sqlite_assistant",
            model_client=model_client,
            system_message=system_prompt + "\n\nYou can understand natural language inputs and extract relevant information automatically.",
            tools=autogen_tools,
        )
        
        self.logger.info("AutoGen agent created successfully")
        return self.agent
    
    def stream_execute(self, agent, message: str):
        """Execute and mimic LangGraph's streaming format for consistent UI"""
        start_time = time.time()
        
        # Log user input
        log_user_input(self.logger, message)
        self.logger.info("AutoGen starting stream execution")
        
        try:
            import asyncio
            
            message_obj = TextMessage(content=message, source="user")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    agent.on_messages([message_obj], CancellationToken())
                )
                
                execution_time = time.time() - start_time
                
                # TODO: Gather proper usage summary from AutoGen's official tracker
                actual_usage = gather_usage_summary([agent])
                
                if hasattr(response, 'chat_message') and hasattr(response.chat_message, 'content'):
                    content = response.chat_message.content
                    
                    # Detect tool usage and map to LangGraph equivalents
                    tool_mapping = {
                        "All Members:": ("ViewAllMembers", "Here are all the members in the database."),
                        "All Products:": ("ViewAllProducts", "Here are all the products in the database."),
                        "Successfully added member": ("ExtractAndWriteUserInfo", "I've successfully added the new member to the database."),
                        "Purchase successful": ("Purchase", "The purchase has been processed successfully."),
                        "Purchase records for": ("PurchaseRecordFetcher", "I've retrieved the purchase history for you.")
                    }
                    
                    tool_used = None
                    tool_result = None
                    final_message = content
                    
                    # Check for tool patterns
                    for pattern, (tool_name, message_text) in tool_mapping.items():
                        if pattern in content:
                            tool_used = tool_name
                            tool_result = content
                            final_message = message_text
                            break
                    
                    # If tool was used, mimic LangGraph's three-step process exactly
                    if tool_used:
                        # Step 1: Agent decides to use tool (like LangGraph)
                        yield {
                            "agent": {
                                "messages": [{
                                    "content": "",
                                    "tool_calls": [{"name": tool_used}]
                                }]
                            }
                        }
                        
                        # Step 2: Tool execution result (like LangGraph)
                        yield {
                            "tools": {
                                "messages": [{
                                    "name": tool_used,
                                    "content": tool_result.replace("All Members:\n", "").replace("All Products:\n", "")
                                }]
                            }
                        }
                        
                        # Step 3: Final agent response (like LangGraph)
                        yield {
                            "agent": {
                                "messages": [{
                                    "content": final_message,
                                    "tool_calls": None
                                }]
                            }
                        }
                    else:
                        # No tool used, direct response
                        yield {
                            "agent": {
                                "messages": [{
                                    "content": content,
                                    "tool_calls": None
                                }]
                            }
                        }
                    
                    self.logger.info(f"AutoGen completed execution in {execution_time:.2f}s")
                    
                else:
                    self.logger.warning("AutoGen response has no content")
                    yield {
                        "agent": {
                            "messages": [{
                                "content": "I'm ready to help you with database operations.",
                                "tool_calls": None
                            }]
                        }
                    }
                    
            finally:
                loop.close()
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"AutoGen execution failed after {execution_time:.2f}s: {e}")
            yield {
                "agent": {
                    "messages": [{
                        "content": f"I encountered an error: {str(e)}",
                        "tool_calls": None
                    }]
                }
            }
    
    def get_framework_name(self) -> str:
        return "AutoGen (Structured Output)"
    
    def get_framework_info(self) -> Dict[str, Any]:
        return {
            "name": "AutoGen",
            "description": "Microsoft AutoGen with GPT-4o Structured Output for natural language understanding",
            "model": "GPT-4o with Structured Output",
            "features": [
                "Natural Language Understanding",
                "Structured Output Extraction", 
                "Conversational Agents",
                "Azure OpenAI Integration",
                "Function Calling",
                "SQLite Integration"
            ],
            "dependencies": ["autogen-agentchat", "autogen-ext[openai,azure]", "pydantic"]
        }