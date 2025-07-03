from abc import ABC, abstractmethod
from typing import List, Any, Dict

class BaseFramework(ABC):
    """Base framework abstraction class"""
    
    @abstractmethod
    def create_agent(self, tools: List, system_prompt: str):
        """Create agent instance with framework-specific LLM initialization"""
        pass
    
    @abstractmethod
    def stream_execute(self, agent, message: str):
        """Execute and return streaming results"""
        pass
    
    @abstractmethod
    def get_framework_name(self) -> str:
        """Get framework name"""
        pass
    
    @abstractmethod
    def get_framework_info(self) -> Dict[str, Any]:
        """Get framework information"""
        pass