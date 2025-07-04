import os
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field
import json
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from autogen_core.models import UserMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from .base_framework import BaseFramework
from backend.db_manager import DBManager

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


class AutoGenFramework(BaseFramework):
    """AutoGen framework implementation with Structured Output"""
    
    def __init__(self):
        self.agent = None
        self.model_client = None
        self.extraction_client = None
    
    def _initialize_model_client(self):
        """Initialize Azure OpenAI model client"""
        if self.model_client is None:
            self.model_client = AzureOpenAIChatCompletionClient(
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_API_KEY"),
                azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                api_version=os.getenv("AZURE_API_VERSION", "2025-04-01-preview"),
                model=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                temperature=0,
            )

        if self.extraction_client is None:
            self.extraction_client = AzureOpenAIChatCompletionClient(
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_API_KEY"),
                azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                api_version=os.getenv("AZURE_API_VERSION", "2025-04-01-preview"),
                model=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                temperature=0,
            )
        
        return self.model_client
    
    async def _extract_structured_data(self, text: str, model_class: BaseModel) -> BaseModel:
        try:
            messages = [UserMessage(content=f"Extract information from this text: {text}", source="user")]
            
            response = await self.extraction_client.create(
                messages=messages,
                extra_create_args={"response_format": model_class}
            )
            
            if response.content:
                data = json.loads(response.content)
                return model_class.model_validate(data)
            else:
                return model_class()
                
        except Exception as e:
            print(f"Extraction error: {e}")
            return model_class()
    
    def _create_autogen_tools(self) -> List[FunctionTool]:
        """Create AutoGen tools with natural language understanding"""
        db_manager = DBManager("customer_database.db")
        
        def extract_and_add_member(user_input: str) -> str:
            """Add a new member by extracting info from natural language."""
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
                    return "❌ Could not extract user name from input."

                existing_member = db_manager.get_member_by_name(user_info.name)
                if existing_member:
                    return f"Member {user_info.name} already exists with ID: {existing_member[0]}"

                db_manager.insert_member(
                    user_info.name, 
                    user_info.email or "unknown@example.com", 
                    user_info.age or 0
                )
                
                new_member = db_manager.get_member_by_name(user_info.name)
                return f"✅ Successfully added member: {user_info.name} (ID: {new_member[0]})"
                
            except Exception as e:
                return f"❌ Error adding member: {str(e)}"
        
        def extract_and_make_purchase(purchase_input: str) -> str:
            """Process a purchase by extracting user and product info from natural language."""
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
                    return "❌ Could not extract user name from input."
                if not product_info.name:
                    return "❌ Could not extract product name from input."

                member = db_manager.get_member_by_name(user_info.name)
                if not member:
                    return f"❌ Member '{user_info.name}' not found. Please add member first."

                product = db_manager.get_product_by_name(product_info.name)
                if not product:
                    return f"❌ Product '{product_info.name}' not found."
                
                member_id = member[0]
                product_id = product[0]
                quantity = product_info.number or 1
                
                db_manager.insert_record(member_id, product_id, quantity)
                
                return f"✅ Purchase successful! {user_info.name} bought {quantity} {product_info.name}(s)."
                
            except Exception as e:
                return f"❌ Error processing purchase: {str(e)}"
        
        def get_member_purchase_history(user_input: str) -> str:
            """Get purchase history by extracting member name from natural language."""
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
                    return "❌ Could not extract user name from input."
                
                member = db_manager.get_member_by_name(user_info.name)
                if not member:
                    return f"❌ Member '{user_info.name}' not found."
                
                member_id = member[0]
                records = db_manager.get_member_records(member_id)
                
                if not records:
                    return f"No purchase records found for {user_info.name}."
                
                response = f"📋 Purchase records for {user_info.name}:\n"
                for record in records:
                    response += f"• Product: {record[1]}, Price: ${record[2]}, Quantity: {record[3]}, Total: ${record[4]}\n"
                
                return response
                
            except Exception as e:
                return f"❌ Error retrieving records: {str(e)}"
        
        def view_all_members() -> str:
            """Return all members from the SQLite database."""
            try:
                members = db_manager.list_all_members()
                return f"👥 All Members:\n{members.to_string(index=False)}"
            except Exception as e:
                return f"❌ Error retrieving members: {str(e)}"
        
        def view_all_products() -> str:
            """Return all products from the SQLite database."""
            try:
                products = db_manager.list_all_products()
                return f"📦 All Products:\n{products.to_string(index=False)}"
            except Exception as e:
                return f"❌ Error retrieving products: {str(e)}"
        
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
        
        return tools
    
    def create_agent(self, tools: List, system_prompt: str):
        """Create AutoGen agent with structured output capabilities"""
        model_client = self._initialize_model_client()

        autogen_tools = self._create_autogen_tools()
        
        self.agent = AssistantAgent(
            name="sqlite_assistant",
            model_client=model_client,
            system_message=system_prompt + "\n\nYou can understand natural language inputs and extract relevant information automatically.",
            tools=autogen_tools,
        )
        
        return self.agent
    
    def stream_execute(self, agent, message: str):
        """Execute and mimic LangGraph's streaming format for consistent UI"""
        try:
            import asyncio
            
            message_obj = TextMessage(content=message, source="user")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    agent.on_messages([message_obj], CancellationToken())
                )
                
                if hasattr(response, 'chat_message') and hasattr(response.chat_message, 'content'):
                    content = response.chat_message.content
                    
                    # Detect tool usage and map to LangGraph equivalents
                    tool_mapping = {
                        "👥 All Members:": ("ViewAllMembers", "Here are all the members in the database."),
                        "📦 All Products:": ("ViewAllProducts", "Here are all the products in the database."),
                        "✅ Successfully added member": ("ExtractAndWriteUserInfo", "I've successfully added the new member to the database."),
                        "✅ Purchase successful": ("Purchase", "The purchase has been processed successfully."),
                        "📋 Purchase records for": ("PurchaseRecordFetcher", "I've retrieved the purchase history for you.")
                    }
                    
                    tool_used = None
                    tool_result = None
                    final_message = content
                    
                    # Check for tool patterns
                    for pattern, (tool_name, message) in tool_mapping.items():
                        if pattern in content:
                            tool_used = tool_name
                            tool_result = content
                            final_message = message
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
                                    "content": tool_result.replace("👥 All Members:\n", "").replace("📦 All Products:\n", "")
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
                else:
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
                "🧠 Natural Language Understanding",
                "🔧 Structured Output Extraction", 
                "💬 Conversational Agents",
                "🔗 Azure OpenAI Integration",
                "⚡ Function Calling",
                "📊 SQLite Integration"
            ],
            "dependencies": ["autogen-agentchat", "autogen-ext[openai,azure]", "pydantic"]
        }