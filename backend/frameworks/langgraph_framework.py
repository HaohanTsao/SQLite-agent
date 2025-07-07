import os
from typing import Annotated, List, Any, Dict
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from .base_framework import BaseFramework
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import get_usage_metadata_callback
import time

# Import our unified logging system
from backend.utils.simple_logger import (
    setup_simple_logger, 
    ExecutionTimer, 
    log_execution_time,
    log_tokens,
    log_user_input
)


class LangGraphState(TypedDict):
    """LangGraph state definition"""
    messages: Annotated[list, add_messages]


class LangGraphFramework(BaseFramework):
    """LangGraph framework implementation with Azure OpenAI"""
    
    def __init__(self):
        # Setup unified logger
        self.logger = setup_simple_logger("LangGraph")
        self.llm = None
        self.logger.info("LangGraph framework initialized")
    
    def _initialize_llm(self):
        """Initialize Azure OpenAI LLM"""
        if self.llm is None:
            with ExecutionTimer(self.logger, "LLM initialization"):
                self.llm = AzureChatOpenAI(
                    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                    api_key=os.getenv("AZURE_API_KEY"),
                    azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                    api_version=os.getenv("AZURE_API_VERSION", "2025-04-01-preview"),
                    temperature=0,
                    stream_usage=True,
                )
        return self.llm
    
    @log_execution_time("Agent Creation")
    def create_agent(self, tools: List, system_prompt: str):
        """Create LangGraph agent instance with Azure OpenAI"""
        self.logger.info("LangGraph starting agent creation")
        
        llm = self._initialize_llm()
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
        
        # Create React Agent
        agent = create_react_agent(
            model=llm_with_tools, 
            tools=tools,
            prompt=system_prompt
        )
        
        self.logger.info("LangGraph agent created successfully")
        return agent

    def stream_execute(self, agent, message: str):
        start_time = time.time()
        log_user_input(self.logger, message)
        self.logger.info("LangGraph starting stream execution")
        
        try:
            step_count = 0
            
            # Use LangChain's official usage metadata callback
            with get_usage_metadata_callback() as cb:
                for step in agent.stream(
                    {"messages": [HumanMessage(content=message)]}, 
                    stream_mode="updates"
                ):
                    step_count += 1
                    yield step
                
                # Get token usage from LangChain callback
                if cb.usage_metadata:
                    total_prompt_tokens = 0
                    total_completion_tokens = 0
                    total_tokens = 0
                    
                    for model_name, usage in cb.usage_metadata.items():
                        total_prompt_tokens += usage.get("input_tokens", 0)
                        total_completion_tokens += usage.get("output_tokens", 0)
                        total_tokens += usage.get("total_tokens", 0)
                    
                    if total_tokens > 0:
                        log_tokens(
                            "LangGraph",
                            prompt_tokens=total_prompt_tokens,
                            completion_tokens=total_completion_tokens,
                            total_tokens=total_tokens
                        )
                    else:
                        self.logger.warning("LangGraph usage metadata contains no tokens")
                else:
                    self.logger.warning("No usage metadata captured by LangGraph")
            
            execution_time = time.time() - start_time
            self.logger.info(f"LangGraph completed execution in {execution_time:.2f}s ({step_count} steps)")
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"LangGraph execution failed after {execution_time:.2f}s: {e}")
            raise

    def get_framework_name(self) -> str:
        """Get framework name"""
        return "LangGraph"
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get framework information"""
        return {
            "name": "LangGraph", 
            "description": "LangChain's graph-based agent framework with Azure OpenAI",
            "model": "GPT-4 Turbo",
            "features": [
                "React Agent Pattern", 
                "Streaming Support", 
                "State Management",
                "Tool Integration",
                "Azure OpenAI"
            ],
            "dependencies": ["langchain-openai", "langgraph"]
        }