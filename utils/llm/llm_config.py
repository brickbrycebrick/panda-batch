import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration class for LLM provider and model settings"""
    
    provider: str = "litellm"
    model: str = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None
    client: Optional[Any] = None  # For custom clients (e.g., OpenAI client with custom base_url)
    
    def __post_init__(self):
        """Ensure a model is provided, otherwise list available configurations."""
        if self.model is None:
            # Get available configurations from the LLMConfigs class
            available_configs = [
                func for func in dir(LLMConfigs) 
                if callable(getattr(LLMConfigs, func)) and not func.startswith("_")
            ]
            
            error_message = (
                "No model specified. Please provide a model name directly or use one of the predefined configurations from `LLMConfigs`.\n"
                f"Available configurations: {', '.join(available_configs)}\n"
                "Example: `config = LLMConfigs.openai(model='gpt-4o-mini')`"
            )
            raise ValueError(error_message)
    
    def get_decorator_kwargs(self, response_model=None) -> Dict[str, Any]:
        """Get kwargs for the @llm.call decorator"""
        kwargs = {
            "provider": self.provider,
            "model": self.model
        }
        
        if response_model:
            kwargs["response_model"] = response_model
            
        # Use custom client if provided (takes precedence over base_url/api_key)
        if self.client:
            kwargs["client"] = self.client
        else:
            # Fall back to base_url and api_key for other providers
            if self.base_url:
                kwargs["base_url"] = self.base_url
                
            if self.api_key:
                kwargs["api_key"] = self.api_key
            
        if self.extra_params:
            kwargs.update(self.extra_params)
            
        return kwargs


# Predefined configurations for common setups
class LLMConfigs:
    """Collection of predefined LLM configurations"""
    
    @staticmethod
    def azure_openai(deployment_name: Optional[str] = None) -> LLMConfig:
        """Configuration for Azure OpenAI"""
        model_name = deployment_name or os.getenv("AZURE_DEPLOYMENT_NAME")
        if not model_name:
            raise ValueError("Deployment name must be provided or AZURE_DEPLOYMENT_NAME environment variable must be set")
        
        return LLMConfig(
            provider="litellm",
            model=f"azure/{model_name}"
        )
    
    @staticmethod
    def openai(model: str = "gpt-4o-mini") -> LLMConfig:
        """Configuration for OpenAI"""
        return LLMConfig(
            provider="openai",
            model=model
        )
    
    @staticmethod
    def anthropic(model: str = "claude-sonnet-4-20250514") -> LLMConfig:
        """Configuration for Anthropic"""
        return LLMConfig(
            provider="anthropic",
            model=model
        )
    
    @staticmethod
    def custom(provider: str, model: str, **kwargs) -> LLMConfig:
        """Configuration for custom provider/model combinations"""
        return LLMConfig(
            provider=provider,
            model=model,
            extra_params=kwargs
        )
    
    @staticmethod
    def openrouter(model: str = "openai/gpt-4o-mini", site_url: Optional[str] = None, site_name: Optional[str] = None) -> LLMConfig:
        """Configuration for OpenRouter (OpenAI-compatible API)"""
        
        # Create custom headers for OpenRouter rankings
        headers = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name
        
        custom_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            default_headers=headers if headers else None
        )
        
        return LLMConfig(
            provider="openai",
            model=model,
            client=custom_client
        )
