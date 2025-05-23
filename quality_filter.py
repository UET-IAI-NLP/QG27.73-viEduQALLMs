import fasttext
from datasets import Dataset, load_dataset
import re
import argparse
def filter_quality_data_hf(dataset, model_path, text_field="text",
                           target_label="__label__hq", prob_threshold=0.5,
                           batch_size=64):
    """
    Filters a Hugging Face Dataset using a FastText quality classifier.

    Args:
        dataset (datasets.Dataset): A Hugging Face Dataset containing the text data.
        model_path (str): Path to the trained FastText model (e.g., 'quality_classifier.bin').
        text_field (str): The field in the dataset containing the text.
        target_label (str): The label representing high-quality text.
        prob_threshold (float): Minimum confidence to accept the prediction.
        batch_size (int): Batch size for the map() function.
        
    Returns:
        datasets.Dataset: A filtered Hugging Face Dataset with only examples that pass quality filtering.
    """
    # Load the FastText model
    model = fasttext.load_model(model_path)
    dataset = load_dataset(dataset, split='train')
    def replace_newlines(text: str) -> str:
        return re.sub("\n+", " ", text)
    # Define a mapping function that computes the "pass_filter" column.
    def quality_filter(batch):
        texts = batch[text_field]
        texts = [replace_newlines(text) for text in texts]
        # Get predictions for all texts in the batch
        labels, probs = model.predict(texts)
        # Create a boolean list: True if the prediction meets the target label and threshold.
        pass_filter = [(lbl[0] == target_label) and (pr[0] >= prob_threshold) 
                       for lbl, pr in zip(labels, probs)]
        batch["pass_filter"] = pass_filter
        return batch

    # Use map() to add the "pass_filter" column to the dataset.
    # Using num_proc=1 to avoid multiprocessing issues with FastText.
    dataset_with_filter = dataset.map(
        quality_filter,
        batched=True,
        batch_size=batch_size,
        num_proc=1
    )
    
    # Filter the dataset based on the "pass_filter" flag.
    filtered_dataset = dataset_with_filter.filter(lambda ex: ex["pass_filter"])
    return filtered_dataset

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Example script with argument parser.')

    # Define expected arguments
    parser.add_argument('dataset', type=str, help='huggingface dataset path')
    parser.add_argument('--model_path', type=str, help='path to the trained FastText model')
    parser.add_argument('--text_field', type=str, default='text', help='field in the dataset containing the text')
    parser.add_argument('--target_label', type=str, default='__label__hq', help='label representing high-quality text')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    result = filter_quality_data_hf(args.dataset, args.model_path, text_field=args.text_field, target_label=args.target_label)
    result.push_to_hub(args.dataset)
if __name__ == '__main__':
    main()
