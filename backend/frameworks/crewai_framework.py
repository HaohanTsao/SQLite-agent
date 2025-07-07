import os
import time
from typing import List, Any, Dict, Optional, Iterator
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events import (
    ToolUsageStartedEvent,
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
)
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
    price: Optional[str] = Field(default=None, description="The price of the product")
    number: Optional[int] = Field(default=1, description="The number of products to purchase")


class CrewAIEventListener(BaseEventListener):
    """Event listener to capture CrewAI execution events for streaming"""
    
    def __init__(self, logger):
        super().__init__()
        self.events = []
        self.tool_calls = []
        self.tool_results = []
        self.agent_messages = []
        self.logger = logger
        
    def setup_listeners(self, crewai_event_bus):
        """Set up event listeners for various CrewAI events"""
        
        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event):
            self.logger.info(f"Tool usage started: {event.tool_name}")
            self.tool_calls.append({
                "name": event.tool_name,
                "input": getattr(event, 'tool_input', ''),
                "started": True
            })
            self.events.append(("tool_started", event))
    
    def clear_events(self):
        """Clear all captured events"""
        self.events.clear()
        self.tool_calls.clear()
        self.tool_results.clear()
        self.agent_messages.clear()

class ExtractAndWriteUserInfoTool(BaseTool):
    name: str = "ExtractAndWriteUserInfo"
    description: str = "Extract user information from text and write it to SQLite database"
    framework: Any = Field(default=None, exclude=True)
    
    def __init__(self, framework_instance, **kwargs):
        super().__init__(framework=framework_instance, **kwargs)
    
    def _run(self, text: str) -> str:
        """Extract user information and write to database"""
        log_tool_usage(self.framework.logger, self.name, text)
        
        try:
            user_info = self.framework._extract_user_info(text)
            db_manager = self.framework._initialize_db_manager()
            
            if not user_info.name:
                result = "❌ Could not extract user name from input."
                self.framework.logger.error("Tool failed: ExtractAndWriteUserInfo - no name extracted")
                return result
            
            member = db_manager.get_member_by_name(user_info.name)
            if member:
                result = f"Member {user_info.name} already exists with ID: {member[0]}"
                self.framework.logger.info("Tool completed: ExtractAndWriteUserInfo - member exists")
                return result
            else:
                db_manager.insert_member(user_info.name, user_info.email, user_info.age)
                new_member = db_manager.get_member_by_name(user_info.name)
                result = f"Extracted and wrote user info: {new_member}"
                self.framework.logger.info(f"Tool completed: ExtractAndWriteUserInfo - {user_info.name}")
                return result
        except Exception as e:
            self.framework.logger.error(f"Tool failed: ExtractAndWriteUserInfo - {e}")
            return f"❌ Error adding member: {str(e)}"

class PurchaseRecordFetcherTool(BaseTool):
    name: str = "PurchaseRecordFetcher"
    description: str = "Extract user information from text and fetch purchase records from SQLite database"
    framework: Any = Field(default=None, exclude=True)
    
    def __init__(self, framework_instance, **kwargs):
        super().__init__(framework=framework_instance, **kwargs)
    
    def _run(self, text: str) -> str:
        """Get purchase records for user"""
        log_tool_usage(self.framework.logger, self.name, text)
        
        try:
            user_info = self.framework._extract_user_info(text)
            db_manager = self.framework._initialize_db_manager()
            
            if not user_info.name:
                result = f"❌ Could not extract user name from input."
                self.framework.logger.error("Tool failed: PurchaseRecordFetcher - no name extracted")
                return result
            
            member = db_manager.get_member_by_name(user_info.name)
            if not member:
                result = f"No member found for name '{user_info.name}'"
                self.framework.logger.error(f"Tool failed: PurchaseRecordFetcher - member {user_info.name} not found")
                return result
            
            member_id = member[0]
            purchase_records = db_manager.get_member_records(member_id)
            
            if not purchase_records:
                result = f"No purchase records found for member {user_info.name} (ID: {member_id})"
            else:
                response = f"Purchase records for {user_info.name} (ID: {member_id}):\n"
                for record in purchase_records:
                    response += f"- Record ID: {record[0]}, Product: {record[1]}, Price: {record[2]}, Number: {record[3]}, Payment: {record[2]*record[3]}\n"
                result = response
            
            self.framework.logger.info(f"Tool completed: PurchaseRecordFetcher - {user_info.name}")
            return result
        except Exception as e:
            self.framework.logger.error(f"Tool failed: PurchaseRecordFetcher - {e}")
            return f"❌ Error retrieving records: {str(e)}"

class PurchaseTool(BaseTool):
    name: str = "Purchase"
    description: str = "Extract user and product information and complete purchase process"
    framework: Any = Field(default=None, exclude=True)
    
    def __init__(self, framework_instance, **kwargs):
        super().__init__(framework=framework_instance, **kwargs)
    
    def _run(self, text: str) -> str:
        """Process purchase transaction"""
        log_tool_usage(self.framework.logger, self.name, text)
        
        try:
            user_info = self.framework._extract_user_info(text)
            product_info = self.framework._extract_product_info(text)
            db_manager = self.framework._initialize_db_manager()
            
            if user_info.name is None:
                result = "User information is incomplete."
                self.framework.logger.error("Tool failed: Purchase - no user name")
                return result
            if product_info.name is None:
                result = "Product information is incomplete."
                self.framework.logger.error("Tool failed: Purchase - no product name")
                return result
            
            member = db_manager.get_member_by_name(user_info.name)
            
            if not member:
                db_manager.insert_member(user_info.name, user_info.email, user_info.age)
                member = db_manager.get_member_by_name(user_info.name)
                self.framework.logger.info(f"Added new member during purchase: {user_info.name}")
            
            member_id = member[0]
            
            product = db_manager.get_product_by_name(product_info.name)
            if not product:
                result = f"Sorry, the product '{product_info.name}' does not exist."
                self.framework.logger.error(f"Tool failed: Purchase - product {product_info.name} not found")
                return result
            
            product_id = product[0]
            db_manager.insert_record(member_id, product_id, product_info.number)
            
            result = f"Purchase successful! Member {user_info.name} bought {product_info.number} {product_info.name}(s)."
            self.framework.logger.info(f"Tool completed: Purchase - {user_info.name} bought {product_info.name}")
            return result
        except Exception as e:
            self.framework.logger.error(f"Tool failed: Purchase - {e}")
            return f"❌ Error processing purchase: {str(e)}"

class ViewAllProductsTool(BaseTool):
    name: str = "ViewAllProducts"
    description: str = "View all products in database"
    framework: Any = Field(default=None, exclude=True)
    
    def __init__(self, framework_instance, **kwargs):
        super().__init__(framework=framework_instance, **kwargs)
    
    def _run(self, text: str) -> str:
        """Return all products from database"""
        log_tool_usage(self.framework.logger, self.name)
        
        try:
            db_manager = self.framework._initialize_db_manager()
            products = db_manager.list_all_products()
            result = products.to_string(index=False)
            self.framework.logger.info("Tool completed: ViewAllProducts")
            return result
        except Exception as e:
            self.framework.logger.error(f"Tool failed: ViewAllProducts - {e}")
            return f"Error retrieving products: {e}"

class ViewAllMembersTool(BaseTool):
    name: str = "ViewAllMembers"
    description: str = "View all members in database"
    framework: Any = Field(default=None, exclude=True)
    
    def __init__(self, framework_instance, **kwargs):
        super().__init__(framework=framework_instance, **kwargs)
    
    def _run(self, text: str) -> str:
        """Return all members from database"""
        log_tool_usage(self.framework.logger, self.name)
        
        try:
            db_manager = self.framework._initialize_db_manager()
            members = db_manager.list_all_members()
            result = members.to_string(index=False)
            self.framework.logger.info("Tool completed: ViewAllMembers")
            return result
        except Exception as e:
            self.framework.logger.error(f"Tool failed: ViewAllMembers - {e}")
            return f"Error retrieving members: {e}"

class CrewAIFramework(BaseFramework):
    """CrewAI framework with native event listening and structured extraction"""
    
    def __init__(self):
        # Setup unified logger
        self.logger = setup_simple_logger("CrewAI")
        self.agent = None
        self.crew = None
        self.current_task = None
        self.db_manager = None
        self.llm = None
        self.extraction_agents = {}
        self.event_listener = None
        self.logger.info("CrewAI framework initialized")
    
    def _initialize_azure_llm(self):
        """Initialize Azure OpenAI LLM with streaming"""
        if self.llm is None:
            with ExecutionTimer(self.logger, "Azure LLM initialization"):
                self.llm = LLM(
                    model="azure/gpt-4o",
                    api_key=os.getenv("AZURE_API_KEY"),
                    base_url=os.getenv("AZURE_ENDPOINT"),
                    api_version=os.getenv("AZURE_API_VERSION", "2024-04-01-preview"),
                    temperature=0.1,
                    stream=True  # Enable streaming for better event tracking
                )
        return self.llm
    
    def _initialize_db_manager(self):
        """Initialize database manager"""
        if self.db_manager is None:
            self.db_manager = DBManager("customer_database.db")
            self.db_manager.create_tables()
        return self.db_manager
    
    def _create_extraction_agents(self):
        """Create specialized agents for structured extraction"""
        if not self.extraction_agents:
            azure_llm = self._initialize_azure_llm()
            
            # User extraction agent
            self.extraction_agents['user'] = Agent(
                role="User Information Extraction Specialist",
                goal="Extract user information (name, email, age) from text with high accuracy",
                backstory="You are an expert at identifying and extracting user information from natural language text.",
                llm=azure_llm,
                verbose=False,
                allow_delegation=False
            )
            
            # Product extraction agent  
            self.extraction_agents['product'] = Agent(
                role="Product Information Extraction Specialist", 
                goal="Extract product information (name, quantity) from purchase-related text",
                backstory="You specialize in understanding purchase intentions and extracting product details from customer requests.",
                llm=azure_llm,
                verbose=False,
                allow_delegation=False
            )
        
        return self.extraction_agents
    
    def _extract_user_info(self, text: str) -> UserInfo:
        """Extract user information using CrewAI structured output"""
        start_time = time.time()
        self.logger.info("Starting user info extraction")
        
        try:
            extraction_agents = self._create_extraction_agents()
            
            extraction_task = Task(
                description=f"Extract user information from: '{text}'. Find name, email, and age.",
                expected_output="Structured user information",
                agent=extraction_agents['user'],
                output_pydantic=UserInfo
            )
            
            extraction_crew = Crew(
                agents=[extraction_agents['user']],
                tasks=[extraction_task],
                process=Process.sequential,
                verbose=False
            )
            
            result = extraction_crew.kickoff()
            
            execution_time = time.time() - start_time
            self.logger.info(f"User extraction completed in {execution_time:.2f}s")
            
            if hasattr(result, 'pydantic') and result.pydantic:
                return result.pydantic
            else:
                return self._fallback_user_extraction(text)
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error in user extraction after {execution_time:.2f}s: {e}")
            return self._fallback_user_extraction(text)
    
    def _extract_product_info(self, text: str) -> ProductInfo:
        """Extract product information using CrewAI structured output"""
        start_time = time.time()
        self.logger.info("Starting product info extraction")
        
        try:
            extraction_agents = self._create_extraction_agents()
            
            extraction_task = Task(
                description=f"Extract product purchase info from: '{text}'. Find product name and quantity.",
                expected_output="Structured product information",
                agent=extraction_agents['product'],
                output_pydantic=ProductInfo
            )
            
            extraction_crew = Crew(
                agents=[extraction_agents['product']],
                tasks=[extraction_task],
                process=Process.sequential,
                verbose=False
            )
            
            result = extraction_crew.kickoff()
            
            execution_time = time.time() - start_time
            self.logger.info(f"Product extraction completed in {execution_time:.2f}s")
            
            if hasattr(result, 'pydantic') and result.pydantic:
                return result.pydantic
            else:
                return self._fallback_product_extraction(text)
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error in product extraction after {execution_time:.2f}s: {e}")
            return self._fallback_product_extraction(text)
    
    def _fallback_user_extraction(self, text: str) -> UserInfo:
        """Fallback user extraction using simple pattern matching"""
        import re
        data = {}
        
        # Extract name patterns
        name_patterns = [
            r'(?:add|member|user|named|name)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'([A-Za-z]+(?:\s+[A-Za-z]+)*)\s*,',
            r'([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:wants|bought|purchase)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["name"] = match.group(1).strip()
                break
        
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            data["email"] = email_match.group()
        
        # Extract age
        age_match = re.search(r'\b(\d{1,3})\s*(?:years?\s*old|age)\b', text, re.IGNORECASE)
        if age_match:
            data["age"] = int(age_match.group(1))
        
        return UserInfo.model_validate(data)
    
    def _fallback_product_extraction(self, text: str) -> ProductInfo:
        """Fallback product extraction using simple pattern matching"""
        import re
        data = {}
        
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
        elif any(word in text.lower() for word in ["buy", "purchase", "wants"]):
            data["number"] = 1
        
        return ProductInfo.model_validate(data)

    def _create_database_tools(self) -> List[BaseTool]:
        """Create CrewAI tools using specific tool classes"""
        
        tools = [
            ExtractAndWriteUserInfoTool(framework_instance=self),
            PurchaseRecordFetcherTool(framework_instance=self),
            PurchaseTool(framework_instance=self),
            ViewAllProductsTool(framework_instance=self),
            ViewAllMembersTool(framework_instance=self)
        ]
        
        return tools    
 
    @log_execution_time("Agent Creation")
    def create_agent(self, tools: List, system_prompt: str):
        """Create CrewAI agent with event listening and structured extraction"""
        self.logger.info("CrewAI starting agent creation")
        
        # Initialize components
        azure_llm = self._initialize_azure_llm()
        self._create_extraction_agents()
        
        # Create event listener with logger
        self.event_listener = CrewAIEventListener(self.logger)
        
        # Create database tools with structured extraction
        crewai_tools = self._create_database_tools()
        
        # Create main agent
        self.agent = Agent(
            role="Database Assistant with Event-Driven Extraction",
            goal="Manage SQLite database operations using intelligent information extraction and real-time event tracking",
            backstory=system_prompt + " You use advanced extraction techniques and provide real-time feedback on your operations.",
            tools=crewai_tools,
            llm=azure_llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Create the crew
        self.crew = Crew(
            agents=[self.agent],
            tasks=[],  # Tasks created dynamically
            process=Process.sequential,
            verbose=True
        )
        
        self.logger.info("CrewAI agent created successfully")
        return self.crew

    def stream_execute(self, agent, message: str) -> Iterator[Dict[str, Any]]:
        """Execute CrewAI with dual output: raw tool results + AI intelligent response"""
        start_time = time.time()
        
        # Log user input
        log_user_input(self.logger, message)
        self.logger.info("CrewAI starting stream execution")
        
        try:
            # Clear previous events
            if self.event_listener:
                self.event_listener.clear_events()
            
            # Create dynamic task
            self.current_task = Task(
                description=f"User request: {message}",
                expected_output="A helpful response using intelligent information extraction",
                agent=self.agent
            )
            
            # Update crew tasks
            self.crew.tasks = [self.current_task]
            
            # Capture tool outputs before execution
            captured_tool_outputs = {}
            original_run_methods = {}
            
            # Monkey patch tool _run methods to capture outputs
            for tool in self.agent.tools:
                tool_name = tool.name
                original_run = tool._run
                original_run_methods[tool_name] = original_run
                
                def create_wrapper(tool_name, original_method):
                    def wrapped_run(*args, **kwargs):
                        result = original_method(*args, **kwargs)
                        captured_tool_outputs[tool_name] = result
                        return result
                    return wrapped_run
                
                tool._run = create_wrapper(tool_name, original_run)
            
            try:
                # Execute the crew
                result = self.crew.kickoff()
                
                execution_time = time.time() - start_time
                
                # Get the final AI response
                ai_response = result.raw if hasattr(result, 'raw') else str(result)
                token_usage = result.token_usage

                log_tokens(
                    "CrewAI",
                    prompt_tokens=int(token_usage.prompt_tokens),
                    completion_tokens=int(token_usage.completion_tokens),
                    total_tokens=int(token_usage.total_tokens)
                )
                
                # Process captured events to create streaming output
                events_processed = []
                
                if self.event_listener and self.event_listener.tool_calls:
                    # Process tool calls captured by event listener
                    for tool_call in self.event_listener.tool_calls:
                        if tool_call.get("started"):
                            tool_name = tool_call["name"]
                            
                            # Step 1: Agent decides to use tool
                            yield {
                                "agent": {
                                    "messages": [{
                                        "content": "",
                                        "tool_calls": [{"name": tool_name}]
                                    }]
                                }
                            }
                            
                            events_processed.append(tool_name)
                    
                    # Step 2: Tool execution results (show raw tool outputs)
                    if events_processed:
                        for tool_name in events_processed:
                            # Use captured tool output for tabular display
                            tool_output = captured_tool_outputs.get(tool_name, "No output captured")
                            
                            yield {
                                "tools": {
                                    "messages": [{
                                        "name": tool_name,
                                        "content": tool_output
                                    }]
                                }
                            }
                        
                        # Step 3: AI intelligent response (CrewAI's analysis)
                        yield {
                            "agent": {
                                "messages": [{
                                    "content": ai_response,
                                    "tool_calls": None
                                }]
                            }
                        }
                    else:
                        # No events captured, fallback to direct response
                        yield {
                            "agent": {
                                "messages": [{
                                    "content": ai_response,
                                    "tool_calls": None
                                }]
                            }
                        }
                else:
                    # No events captured, direct response
                    yield {
                        "agent": {
                            "messages": [{
                                "content": ai_response,
                                "tool_calls": None
                            }]
                        }
                    }
                
                self.logger.info(f"CrewAI completed execution in {execution_time:.2f}s")
                
            finally:
                # Restore original _run methods
                for tool in self.agent.tools:
                    tool_name = tool.name
                    if tool_name in original_run_methods:
                        tool._run = original_run_methods[tool_name]
                    
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"CrewAI execution failed after {execution_time:.2f}s: {e}")
            yield {
                "agent": {
                    "messages": [{
                        "content": f"I encountered an error: {str(e)}",
                        "tool_calls": None
                    }]
                }
            }
 
    def _generate_final_message(self, tool_name: str, user_message: str) -> str:
        """Generate appropriate final message based on tool used"""
        messages = {
            "ExtractAndWriteUserInfo": "I've successfully extracted the user information and added them to the database.",
            "PurchaseRecordFetcher": "I've extracted the user information and retrieved their purchase history.",
            "Purchase": "I've processed the purchase by extracting both user and product information.",
            "ViewAllMembers": "Here are all the members in the database.",
            "ViewAllProducts": "Here are all the available products."
        }
        
        return messages.get(tool_name, "I've processed your request using CrewAI with event-driven extraction.")
    
    def get_framework_name(self) -> str:
        return "CrewAI (Event-Driven)"
    
    def get_framework_info(self) -> Dict[str, Any]:
        return {
            "name": "CrewAI", 
            "description": "Multi-agent collaboration framework with Azure OpenAI, event-driven execution tracking, and structured information extraction",
            "model": "Azure OpenAI GPT-4o with Event Streaming",
            "features": [
                "🎧 Real-Time Event Listening",
                "🔄 Event-Driven Tool Tracking", 
                "🤝 Multi-Agent Collaboration",
                "📋 Task-Based Execution with Structured Output", 
                "🔧 Custom Tool Integration",
                "📊 SQLite Database Operations",
                "🎯 Role-Based Agent Design",
                "🧠 Intelligent Information Extraction",
                "🏗️ Azure OpenAI Integration",
                "📝 Pydantic Model Validation",
                "⚡ Streaming Response Support"
            ],
            "dependencies": ["crewai", "crewai-tools", "pydantic", "azure-openai"]
        }