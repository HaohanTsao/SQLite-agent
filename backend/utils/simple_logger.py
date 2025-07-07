"""
Simple unified logging configuration
Supports execution time and token consumption tracking
"""
import logging
import time
import functools
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_simple_logger(name: str = "sqlite_agent") -> logging.Logger:
    """
    Create a simple unified logger
    All frameworks share one configuration but can be distinguished by name
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate configuration
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)  # Only log INFO level and above
    
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # File output - all frameworks share one file
    file_handler = logging.FileHandler(
        log_dir / "sqlite_agent.log", 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # Console output - only show important information
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    
    # Simple format - includes framework name
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class ExecutionTimer:
    """Simple execution time timer"""
    
    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            execution_time = time.time() - self.start_time
            if exc_type is None:
                self.logger.info(f"Completed {self.operation_name} in {execution_time:.2f}s")
            else:
                self.logger.error(f"Failed {self.operation_name} after {execution_time:.2f}s")


def log_execution_time(operation_name: str):
    """
    Decorator: automatically log function execution time
    Usage: @log_execution_time("Agent Creation")
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            logger = getattr(self, 'logger', setup_simple_logger())
            
            start_time = time.time()
            logger.info(f"Starting {operation_name}")
            
            try:
                result = func(self, *args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Completed {operation_name} in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Failed {operation_name} after {execution_time:.2f}s: {e}")
                raise
        return wrapper
    return decorator


class TokenUsageTracker:
    """
    Token usage tracker
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.total_tokens = 0
        self.session_tokens = {}  # Track by framework name
    
    def log_token_usage(self, framework_name: str, prompt_tokens: int = 0, 
                       completion_tokens: int = 0, total_tokens: int = None):
        """Log token usage with enhanced validation"""
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        
        # Enhanced validation
        if total_tokens <= 0:
            self.logger.warning(f"Zero or negative token count for {framework_name}: {total_tokens}")
            return
        
        if prompt_tokens < 0 or completion_tokens < 0:
            self.logger.warning(f"Negative token values for {framework_name}: prompt={prompt_tokens}, completion={completion_tokens}")
            return
        
        # Update totals
        self.total_tokens += total_tokens
        
        # Update framework-specific counts
        if framework_name not in self.session_tokens:
            self.session_tokens[framework_name] = 0
        self.session_tokens[framework_name] += total_tokens
        
        # Log entry with detailed breakdown
        if prompt_tokens > 0 and completion_tokens > 0:
            self.logger.info(
                f"Token usage {framework_name}: {prompt_tokens} input + {completion_tokens} output = {total_tokens} "
                f"(session total: {self.session_tokens[framework_name]})"
            )
        else:
            self.logger.info(
                f"Token usage {framework_name}: {total_tokens} total (session total: {self.session_tokens[framework_name]})"
            )
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session token usage summary"""
        return {
            "total_tokens": self.total_tokens,
            "by_framework": self.session_tokens.copy(),
            "timestamp": datetime.now().isoformat()
        }
    
    def log_session_summary(self):
        """Log session summary"""
        if self.total_tokens > 0:
            self.logger.info(f"Session token summary - Total: {self.total_tokens}")
            for framework, tokens in self.session_tokens.items():
                percentage = (tokens / self.total_tokens) * 100
                self.logger.info(f"  {framework}: {tokens} tokens ({percentage:.1f}%)")


# Global token tracker instance
_global_token_tracker: Optional[TokenUsageTracker] = None

def get_token_tracker() -> TokenUsageTracker:
    """Get global token tracker"""
    global _global_token_tracker
    if _global_token_tracker is None:
        logger = setup_simple_logger("TokenTracker")
        _global_token_tracker = TokenUsageTracker(logger)
    return _global_token_tracker


def log_tokens(framework_name: str, **kwargs):
    """Convenience function: log token usage"""
    tracker = get_token_tracker()
    tracker.log_token_usage(framework_name, **kwargs)


# Convenience logging functions
def log_framework_start(logger: logging.Logger, framework_name: str, operation: str):
    """Log framework operation start"""
    logger.info(f"{framework_name} starting {operation}")

def log_framework_success(logger: logging.Logger, framework_name: str, operation: str, duration: float = None):
    """Log framework operation success"""
    if duration:
        logger.info(f"{framework_name} completed {operation} in {duration:.2f}s")
    else:
        logger.info(f"{framework_name} completed {operation}")

def log_framework_error(logger: logging.Logger, framework_name: str, operation: str, error: Exception, duration: float = None):
    """Log framework operation failure"""
    if duration:
        logger.error(f"{framework_name} failed {operation} after {duration:.2f}s: {error}")
    else:
        logger.error(f"{framework_name} failed {operation}: {error}")

def log_user_input(logger: logging.Logger, message: str, max_length: int = 50):
    """Log user input"""
    truncated = message[:max_length] + "..." if len(message) > max_length else message
    logger.info(f"User input: {truncated}")

def log_tool_usage(logger: logging.Logger, tool_name: str, input_text: str = "", result: str = ""):
    """Log tool usage"""
    if input_text:
        truncated_input = input_text[:30] + "..." if len(input_text) > 30 else input_text
        logger.info(f"Tool called: {tool_name} - {truncated_input}")
    else:
        logger.info(f"Tool called: {tool_name}")
    
    if result and "error" not in result.lower():
        logger.info(f"Tool completed: {tool_name}")
    elif result and "error" in result.lower():
        logger.error(f"Tool failed: {tool_name}")