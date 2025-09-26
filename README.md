# Async LLM Batch Processor

This project provides a simple and efficient toolkit for processing hundreds to thousands of LLM calls asynchronously. It is designed for data scientists and engineers who need to run rapid experiments on moderately sized datasets, with a focus on structured data extraction.

It uses [Mirascope](https://mirascope.com/) for interacting with LLMs, [Pydantic](https://docs.pydantic.dev/) for data validation, and Python's `asyncio` for concurrency.

## Key Features

- **Asynchronous Batch Processing**: Process lists of prompts or pandas DataFrames in parallel.
- **Smart Rate Limiting**: Automatically manages API calls to respect token-per-minute rate limits.
- **Structured Outputs**: Use Pydantic models to get clean, validated, and type-safe data back from the LLM.
- **Flexible Configurations**: Easily switch between LLM providers (OpenAI, Azure, Anthropic, OpenRouter) with a simple configuration class.
- **Pandas-Native Workflow**: Seamlessly enrich pandas DataFrames with structured LLM outputs.
- **Versatile Output Formats**: Get results back as a flattened DataFrame or a list of rich, type-safe Pydantic objects.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up the environment and install dependencies:**
   This project uses `uv` for package management.
   ```bash
   uv sync
   ```

3. **Launch Jupyter Lab:**
   ```bash
   uv run jupyter lab
   ```

## Quick Start

This example demonstrates how to enrich a pandas DataFrame with structured data extracted from text prompts.

```python
import pandas as pd
from pydantic import BaseModel
from utils.llm.batch_llm import BatchProcessor
from utils.llm.llm_config import LLMConfigs

# 1. Define your data and the structure you want to extract
class Book(BaseModel):
    title: str
    author: str
    year: int

df = pd.DataFrame({
    'id': [1, 2, 3],
    'prompts': [
        "To Kill a Mockingbird by Harper Lee (1960)",
        "The Great Gatsby Author: F. Scott Fitzgerald Publication Year: 1925",
        "Book: 1984 Writer: George Orwell Year: 1949",
    ]
})

# 2. Initialize the processor with your desired LLM config
config = LLMConfigs.openai(model="gpt-4o-mini")
processor = BatchProcessor(llm_config=config)

# 3. Process the DataFrame
# This runs all prompts in parallel, handles rate limits, and validates the output
results_df = await processor.process_dataframe(
    df, 
    'prompts', 
    response_model=Book
)

print(results_df)
```

**Output:**

```
   id                                            prompts                  title               author  year
0   1         To Kill a Mockingbird by Harper Lee (1960)  To Kill a Mockingbird           Harper Lee  1960
1   2  The Great Gatsby Author: F. Scott Fitzgerald P...       The Great Gatsby  F. Scott Fitzgerald  1925
2   3        Book: 1984 Writer: George Orwell Year: 1949                   1984        George Orwell  1949
```

## Usage

The `BatchProcessor` class provides three main methods:

- `process_single(text, response_model)`: Process a single string.
- `process_batch(texts, response_model)`: Process a list of strings.
- `process_dataframe(df, prompt_column, response_model, output_format)`: Process a pandas DataFrame column.

See `0_batch_llm_example.ipynb` for more detailed examples, including how to get a list of Pydantic objects for use in APIs or other applications.