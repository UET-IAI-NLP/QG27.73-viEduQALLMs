import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin

class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

# Setup configuration and model
config = AutoConfig.from_pretrained("nvidia/multilingual-domain-classifier")
tokenizer = AutoTokenizer.from_pretrained("nvidia/multilingual-domain-classifier")
model = CustomModel.from_pretrained("nvidia/multilingual-domain-classifier")
model.to('cuda')
model.eval()
import pandas as pd
from datasets import load_dataset
ds = load_dataset('group2sealion/8mil_last', split = 'data')
#model.to('cuda')
# Prepare and process inputs
# ['Sports', 'News']
from tqdm import tqdm, trange
def tokenize_function(examples):
    return tokenizer(examples["text"],add_special_tokens=True,
            max_length=256,
            padding=True,
            truncation=True,return_attention_mask=True,
            return_tensors='pt')

tokenized_dataset = ds.map(tokenize_function, num_proc= 18, batched=True, remove_columns=ds.column_names)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 4. Create a DataLoader for batch inference
batch_size = 256
from torch.utils.data import DataLoader
dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)

# 5. Run batched inference to collect predictions
all_predictions = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        # Move batch tensors to the same device as the model
        input_ids = batch["input_ids"].to('cuda')
        attention_mask = batch["attention_mask"].to('cuda')
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        all_predictions.extend([config.id2label[class_idx.item()] for class_idx in preds.cpu().numpy()])
