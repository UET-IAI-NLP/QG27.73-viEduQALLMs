import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import login
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class BCNN(nn.Module):
    def __init__(self, embedding_dim, output_dim,
                 dropout,bidirectional_units,conv_filters):

        super().__init__()
        self.bert = phoBert
        #.fc_input = nn.Linear(embedding_dim,embedding_dim)
        self.bidirectional_lstm = nn.LSTM(
            embedding_dim, bidirectional_units, bidirectional=True, batch_first=True
        )
        self.conv1 = nn.Conv1d(in_channels=2*bidirectional_units, out_channels=conv_filters[0], kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=2*bidirectional_units, out_channels=conv_filters[1], kernel_size=5)

        self.fc = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,b_input_ids,b_input_mask):
        encoded = self.bert(b_input_ids,b_input_mask)[0]
        embedded, _ = self.bidirectional_lstm(encoded)
        embedded = embedded.permute(0, 2, 1)
        conved_1 = F.relu(self.conv1(embedded))
        conved_2 = F.relu(self.conv2(embedded))
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        #pooled_n = [batch size, n_fibatlters]

        cat = self.dropout(torch.cat((pooled_1, pooled_2), dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]

        result =  self.fc(cat)

        return result
# Replace 'your_api_token' with your actual Hugging Face API token
login(token="...",add_to_git_credential=True)
repo_id = '...'
# 1. Load your dataset (replace 'your_dataset' and 'split' as appropriate)
dataset = load_dataset(repo_id, split="train")  # assumes a column "text"

# 2. Load the model and tokenizer (adjust the model name as needed)

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2', cache_dir = 'True')
model = torch.load(r'/workspace/thviet/LLMs/Monolingual/toxic/toxic.pt')
model.eval()  # set the model in inference mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Tokenize the dataset (assumes your text column is named "text")
def tokenize_function(examples):
    return tokenizer(examples["text"],add_special_tokens=True,
            max_length=256,
            padding=True,
            truncation=True,return_attention_mask=True,
            return_tensors='pt')

tokenized_dataset = dataset.map(tokenize_function, num_proc= 18, batched=True, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 4. Create a DataLoader for batch inference
batch_size = 256
dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)

# 5. Run batched inference to collect predictions
all_predictions = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        # Move batch tensors to the same device as the model
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        all_predictions.extend(preds.cpu().numpy())

# 6. Add the predictions as a new column to the tokenized dataset.
#    (Ensure the order is preserved between your DataLoader and dataset.)
dataset = dataset.add_column("predictions", all_predictions)

# 7. Filter the dataset based on model predictions.
#    For instance, keep only examples where the model predicted label 1.
filtered_dataset = dataset.filter(lambda x: x["predictions"] == 0)
print("Original dataset length:", len(tokenized_dataset))
print("Filtered dataset length (only prediction==0):", len(filtered_dataset))
# e.g., "your_username/your_dataset_name"
# Push dataset to Hugging Face Hub
filtered_dataset.push_to_hub(repo_id)
toxic_dataset = dataset.filter(lambda x: x["predictions"] == 1)
toxic_dataset.push_to_hub('...')

'''import fasttext
import pandas as pd

def filter_quality_data(df, model_path, text_field="text",
                        target_label="__label__high_quality", prob_threshold=0.5):
    """
    Filters a Pandas DataFrame using a FastText quality classifier.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        model_path (str): Path to the trained FastText model (e.g., 'quality_classifier.bin').
        text_field (str): The column name in df that contains the text to classify.
        target_label (str): The label that represents high-quality text.
        prob_threshold (float): The minimum confidence required to accept the prediction.
        
    Returns:
        pd.DataFrame: A DataFrame filtered to only include rows where the classifier 
                      predicts the target label with a probability above the threshold.
    """
    # Load the FastText model
    model = fasttext.load_model(model_path)
    
    # Get the texts from the DataFrame column
    texts = df[text_field].tolist()
    
    # Get predictions for the list of texts (model.predict returns lists of labels and probabilities)
    labels, probs = model.predict(texts)
    
    # Create a boolean mask based on predictions and confidence scores
    mask = [
        (label[0] == target_label) and (prob[0] >= prob_threshold)
        for label, prob in zip(labels, probs)
    ]
    
    # Filter the DataFrame based on the mask
    filtered_df = df[mask].reset_index(drop=True)
    return filtered_df

# Example usage:
if __name__ == "__main__":
    # Create a sample DataFrame
    sample_data = {
        "text": [
            "This dataset is very clean and useful.",
            "There are many errors and noise in this dataset.",
            "The text is well-structured and informative.",
            "Garbage content with irrelevant words."
        ]
    }
    df = pd.DataFrame(sample_data)
    
    # Path to your trained FastText model (adjust as needed)
    model_path = "quality_classifier.bin"
    
    # Filter the DataFrame
    filtered_df = filter_quality_data(df, model_path,
                                      text_field="text",
                                      target_label="__label__high_quality",
                                      prob_threshold=0.5)
    print(filtered_df)
'''
