import asyncio
import os
import pandas as pd
from typing import Optional, Type, Any, List, Literal
from mirascope import llm
from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from ..rate_limiters import TokenRateLimiter
from .llm_config import LLMConfig, LLMConfigs

load_dotenv()


class BatchProcessor:
    """Simple batch processing class for LLMs with rate limiting"""
    
    def __init__(self, 
                 llm_config: Optional[LLMConfig] = None,
                 max_tokens_per_minute: int = 704000,
                 safety_margin: float = 0.9,
                 max_concurrent: int = 50):
        """
        Initialize BatchProcessor
        
        Args:
            llm_config: LLM configuration (defaults to Azure OpenAI from env)
            max_tokens_per_minute: Rate limit for tokens per minute
            safety_margin: Safety factor for rate limiting (0.9 = 90% of limit)
            max_concurrent: Maximum concurrent requests
        """
        self.llm_config = llm_config or LLMConfigs.azure_openai()
        self.rate_limiter = TokenRateLimiter(max_tokens_per_minute, safety_margin, max_concurrent)
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text (approximately 1.3 tokens per word)"""
        return max(10, int(len(text.split()) * 1.3))
    
    async def process_single(self, text: str, response_model: Optional[Type[BaseModel]] = None) -> Any:
        """
        Process a single text input with optional Pydantic response model
        
        Args:
            text: Input text to process
            response_model: Optional Pydantic model for structured output
            
        Returns:
            Processed result (string or Pydantic model instance)
        """
        estimated_tokens = self.estimate_tokens(text)
        await self.rate_limiter.acquire(estimated_tokens=estimated_tokens)
        
        # Get decorator kwargs from configuration
        decorator_kwargs = self.llm_config.get_decorator_kwargs(response_model)
        
        @llm.call(**decorator_kwargs)
        async def _process(prompt: str) -> response_model if response_model else str:
            return prompt
        
        result = await _process(text)
        
        # Convert to string if no response model was specified
        return result if response_model else str(result)
    
    async def process_batch(self, texts: List[str], response_model: Optional[Type[BaseModel]] = None) -> List[Any]:
        """
        Process a batch of texts with rate limiting and error handling
        
        Args:
            texts: List of input texts to process
            response_model: Optional Pydantic model for structured output
            
        Returns:
            List of processed results (may contain error strings for failed requests)
        """
        tasks = [self.process_single(text, response_model) for text in texts]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"Error: {str(result)}"
                    print(f"Error processing item {i}: {result}")
                    processed_results.append(error_msg)
                else:
                    processed_results.append(result)
            return processed_results
        except Exception as e:
            print(f"Batch processing error: {e}")
            return [f"Batch Error: {str(e)}"] * len(texts)
    
    async def process_dataframe(self, 
                              df: pd.DataFrame, 
                              prompt_column: str, 
                              response_model: Optional[Type[BaseModel]] = None,
                              output_format: Literal['dataframe', 'objects'] = 'dataframe',
                              output_column_name: str = 'llm_output') -> Any:
        """
        Process a DataFrame column and return results in the specified format.

        Args:
            df: Input DataFrame.
            prompt_column: Name of the column containing prompts/texts to process.
            response_model: Optional Pydantic model for structured output.
            output_format: 'dataframe' to return a flattened DataFrame (default), or
                           'objects' to return a list of Pydantic model instances.
                           If 'objects' is chosen with a response_model, each object will
                           contain the original DataFrame row data plus the extracted fields.
            output_column_name: Name of the output column if no response_model is used.

        Returns:
            A pandas DataFrame or a list of Pydantic objects, based on output_format.
        """
        texts = df[prompt_column].astype(str).tolist()
        results = await self.process_batch(texts, response_model)

        # Handle case where a response model is used
        if response_model and results and not isinstance(results[0], str):
            if output_format == 'objects':
                # Create Pydantic model
                df_fields = {col: (Any, None) for col in df.columns}
                EnrichedModel = create_model(
                    f'Enriched{response_model.__name__}',
                    **df_fields,
                    __base__=response_model
                )

                enriched_results = []
                df_records = df.to_dict('records')
                for i, llm_result in enumerate(results):
                    if isinstance(llm_result, BaseModel):
                        combined_data = {**df_records[i], **llm_result.model_dump()}
                        enriched_results.append(EnrichedModel(**combined_data))
                    else:
                        # If no response model, just add the result to the row
                        enriched_results.append({**df_records[i], output_column_name: llm_result})
                return enriched_results

            # output_format == 'dataframe'
            model_dicts = [r.model_dump(mode='json') if hasattr(r, 'model_dump') else {} for r in results]
            flattened_df = pd.json_normalize(model_dicts)
            return pd.concat([df.reset_index(drop=True), flattened_df], axis=1)

        # Handle case for raw text output (no response model)
        else:
            if output_format == 'objects':
                return results
            
            # output_format == 'dataframe'
            df_copy = df.copy()
            df_copy[output_column_name] = results
            return df_copy