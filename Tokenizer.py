import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import tiktoken
import numpy as np
from tqdm.auto import tqdm

def fast_batch_count(dataset, text_column="text", tokenizer_name="meta-llama/Llama-3-8B-Instruct", batch_size=1000):
    """Count tokens efficiently with support for both HF and OpenAI tokenizers"""
    # Initialize tokenizer
    if tokenizer_name in ["gpt-4", "gpt-3.5-turbo"]:
        encoder = tiktoken.get_encoding("cl100k_base")
        def tokenize_fn(texts):
            return [len(encoder.encode(text)) for text in texts]
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        def tokenize_fn(texts):
            return [len(enc) for enc in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    
    # Process with progress bar
    counts = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Token counting"):
        batch = dataset[i:i+batch_size][text_column]
        counts.extend(tokenize_fn(batch))
    
    return np.array(counts)

def main():
    parser = argparse.ArgumentParser(description='Count tokens in a Hugging Face dataset.')
    parser.add_argument('dataset', type=str, help='Hugging Face dataset path (e.g. "imdb")')
    parser.add_argument('--split', type=str, default='train', help='Dataset split (default: train)')
    parser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-3-8B-Instruct', 
                       help='Tokenizer name (HF model or "gpt-4"/"gpt-3.5-turbo")')
    parser.add_argument('--text-field', type=str, default='text', 
                       help='Field containing text (default: text)')
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}[{args.split}]")
    dataset = load_dataset(args.dataset, split=args.split)
    
    print(f"Counting tokens using {args.tokenizer}...")
    token_counts = fast_batch_count(
        dataset,
        text_column=args.text_field,
        tokenizer_name=args.tokenizer
    )
    
    print("\nResults:")
    print(f"Total examples: {len(dataset):,}")
    print(f"Total tokens: {token_counts.sum():,}")
    print(f"Average length: {token_counts.mean():.1f} tokens")
    print(f"Maximum length: {token_counts.max():,} tokens")
    print(f"95th percentile: {np.percentile(token_counts, 95):.1f} tokens")

if __name__ == "__main__":  # Fixed this line
    main()