import asyncio
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nest_asyncio

# Apply nest_asyncio to allow running asyncio in environments like Jupyter
nest_asyncio.apply()

from utils.llm.batch_llm import BatchProcessor
from utils.llm.llm_config import LLMConfigs

async def run_benchmark():
    """Runs a performance benchmark for different models and data sizes."""
    
    # 1. --- Data and Prompt Setup ---
    print("Loading dataset...")
    try:
        splits = {'train': 'main/train-00000-of-00001.parquet'}
        tasks_df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
    except Exception as e:
        print(f"Failed to load dataset from Hugging Face: {e}")
        print("Please ensure you have an internet connection and the necessary permissions.")
        return

    instructions = """
        You are a math problem solver. Solve this problem, showing your work step-by-step.
        Conclude with '#### [final_answer]' where final_answer is the numerical result.
        Problem: 
    """

    # 2. --- Benchmark Configuration ---
    models_to_test = {
        "Claude Sonnet 3.5": LLMConfigs.openrouter(model="anthropic/claude-3.5-sonnet"),
        "Gemini 2.5 Flash Lite": LLMConfigs.openrouter(model="google/gemini-2.5-flash-lite"),
        "Grok 4 Fast": LLMConfigs.openrouter(model="x-ai/grok-4-fast:free"),
    }
    
    sizes_to_test = [1, 10, 100, 1000]
    results = []

    print("\nStarting benchmark...")

    # 3. --- Run Benchmark Loop ---
    for model_name, config in models_to_test.items():
        for size in sizes_to_test:
            print(f"\nTesting model: '{model_name}' with {size} rows...")
            
            # Prepare the dataframe for this run
            df_slice = tasks_df.head(size).copy()
            df_slice['question'] = instructions + df_slice['question']
            
            processor = BatchProcessor(llm_config=config,max_tokens_per_minute=100000, max_concurrent=1000)
            
            start_time = time.perf_counter()
            
            try:
                await processor.process_dataframe(
                    df_slice, 
                    prompt_column='question',
                    output_column_name=model_name # Use model name for output column
                )
                end_time = time.perf_counter()
                duration = end_time - start_time
                
                results.append({
                    'model': model_name,
                    'size': size,
                    'time_seconds': duration
                })
                print(f"Finished in {duration:.2f} seconds.")

            except Exception as e:
                print(f"An error occurred during processing: {e}")
                # Optionally, log the error and continue
                results.append({
                    'model': model_name,
                    'size': size,
                    'time_seconds': float('inf') # Indicate failure
                })

    # 4. --- Visualize Results ---
    if not results:
        print("No results to plot.")
        return

    results_df = pd.DataFrame(results)
    print("\n--- Benchmark Results ---")
    print(results_df)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.barplot(data=results_df, x='size', y='time_seconds', hue='model', ax=ax)
    
    ax.set_title('LLM Batch Processing Performance', fontsize=16, weight='bold')
    ax.set_xlabel('Number of Rows Processed', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.legend(title='Model')
    
    # Add text labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fs', label_type='edge', fontsize=9, padding=3)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\nBenchmark complete. Plot saved to 'benchmark_results.png'.")
    plt.show()


if __name__ == "__main__":
    # This allows the async function to be run from a standard Python script
    asyncio.run(run_benchmark())
