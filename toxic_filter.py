import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import login
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import argparse
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
parser = argparse.ArgumentParser(description='Example script with argument parser.')

# Define expected arguments
parser.add_argument('dataset', type=str, help='huggingface dataset path')
args = parser.parse_args()
# Replace 'your_api_token' with your actual Hugging Face API token
login(token="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",add_to_git_credential=True)
repo_id = args.dataset
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
new_repo_id = repo_id + '_clean'
print(new_repo_id)
filtered_dataset.push_to_hub(repo_id)
#toxic_dataset = dataset.filter(lambda x: x["predictions"] == 1)
#toxic_dataset.push_to_hub('...')



