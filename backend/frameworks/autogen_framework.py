import os
from typing import List, Any, Dict
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from .base_framework import BaseFramework

from dotenv import load_dotenv

load_dotenv()


class AutoGenFramework(BaseFramework):
    """AutoGen framework implementation with Azure OpenAI"""
    
    def __init__(self):
        self.agent = None
        self.model_client = None
    
    def _initialize_model_client(self):
        """Initialize Azure OpenAI model client"""
        if self.model_client is None:
            # Use same Azure OpenAI config as LangGraph for fair comparison
            self.model_client = AzureOpenAIChatCompletionClient(
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_API_KEY"),
                azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                api_version=os.getenv("AZURE_API_VERSION", "2025-04-01-preview"),
                model=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                temperature=0,
            )
        return self.model_client
    
    def _convert_langchain_tools_to_autogen(self, langchain_tools: List) -> List:
        """Convert LangChain tools to AutoGen tools"""
        autogen_tools = []
        
        for lc_tool in langchain_tools:
            try:
                # Extract function from LangChain StructuredTool
                func = lc_tool.func
                name = lc_tool.name
                description = lc_tool.description
                
                # Create AutoGen FunctionTool
                autogen_tool = FunctionTool(
                    func,
                    name=name,
                    description=description
                )
                autogen_tools.append(autogen_tool)
                
            except Exception as e:
                print(f"Warning: Could not convert tool {lc_tool.name}: {e}")
                continue
        
        return autogen_tools
    
    def create_agent(self, tools: List, system_prompt: str):
        """Create AutoGen agent instance with Azure OpenAI"""
        model_client = self._initialize_model_client()
        
        # Convert LangChain tools to AutoGen tools
        autogen_tools = self._convert_langchain_tools_to_autogen(tools)
        
        # Create AutoGen AssistantAgent with tools
        self.agent = AssistantAgent(
            name="sqlite_assistant",
            model_client=model_client,
            system_message=system_prompt,
            tools=autogen_tools,  # AutoGen supports tools directly
        )
        
        return self.agent
    
    def stream_execute(self, agent, message: str):
        """Execute and return streaming results"""
        try:
            # Use AutoGen's run method (synchronous)
            import asyncio
            
            # Create message and run agent
            message_obj = TextMessage(content=message, source="user")
            
            # Run the agent synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Simple execution without streaming for now
                response = loop.run_until_complete(
                    agent.on_messages([message_obj], CancellationToken())
                )
                
                # Convert response to compatible format
                if hasattr(response, 'chat_message') and hasattr(response.chat_message, 'content'):
                    content = response.chat_message.content
                    
                    # Yield in format expected by Demo.py
                    yield {
                        "agent": {
                            "messages": [{
                                "content": content,
                                "tool_calls": None
                            }]
                        }
                    }
                else:
                    # Fallback response
                    yield {
                        "agent": {
                            "messages": [{
                                "content": "AutoGen agent response",
                                "tool_calls": None
                            }]
                        }
                    }
                    
            finally:
                loop.close()
                
        except Exception as e:
            # Yield error in compatible format
            yield {
                "agent": {
                    "messages": [{
                        "content": f"Error: {str(e)}",
                        "tool_calls": None
                    }]
                }
            }
    
    def get_framework_name(self) -> str:
        """Get framework name"""
        return "AutoGen"
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get framework information"""
        return {
            "name": "AutoGen",
            "description": "Microsoft AutoGen AgentChat framework with Azure OpenAI",
            "model": "GPT-4 Turbo",
            "features": [
                "Native Tool Support",
                "Conversational Agents",
                "Azure OpenAI Integration",
                "Multi-Agent Collaboration"
            ],
            "dependencies": ["autogen-agentchat", "autogen-ext[openai,azure]"]
        }