import fasttext
from datasets import Dataset
import re
def filter_quality_data_hf(dataset, model_path, text_field="text",
                           target_label="__label__high_quality", prob_threshold=0.5,
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
