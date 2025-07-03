from typing import Annotated, List, Any, Dict
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from .base_framework import BaseFramework
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class LangGraphState(TypedDict):
    """LangGraph state definition"""
    messages: Annotated[list, add_messages]


class LangGraphFramework(BaseFramework):
    """LangGraph framework implementation with Azure OpenAI"""
    
    def __init__(self):
        self.llm = None
    
    def _initialize_llm(self):
        """Initialize Azure OpenAI LLM"""
        if self.llm is None:
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_API_KEY"),
                azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4o"),
                api_version=os.getenv("AZURE_API_VERSION", "2025-04-01-preview"),
                temperature=0,
            )
        return self.llm
    
    def create_agent(self, tools: List, system_prompt: str):
        """Create LangGraph agent instance with Azure OpenAI"""
        llm = self._initialize_llm()
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
        
        # Create React Agent
        agent = create_react_agent(
            model=llm_with_tools, 
            tools=tools,
            prompt=system_prompt
        )
        return agent
    
    def stream_execute(self, agent, message: str):
        """Execute and return streaming results"""

        logger = logging.getLogger(__name__)
        logger.info(f"LangGraph executing message: {message[:100]}...")
        
        try:
            for step in agent.stream(
                {"messages": [HumanMessage(content=message)]}, 
                stream_mode="updates"
            ):
                logger.debug(f"LangGraph step: {step}")
                yield step
        except Exception as e:
            logger.error(f"LangGraph stream error: {e}", exc_info=True)
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