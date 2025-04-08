import torch
from torch import nn
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import argparse
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

#model.to('cuda')
# Prepare and process inputs
# ['Sports', 'News']

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"],add_special_tokens=True,
            max_length=256,
            padding=True,
            truncation=True,return_attention_mask=True,
            return_tensors='pt')

def main():
    parser = argparse.ArgumentParser(description='Example script with argument parser.')

    # Define expected arguments
    parser.add_argument('dataset', type=str, help='huggingface dataset path')
    parser.add_argument('--split', type=str, default='data', help='split of hf dataset')
    parser.add_argument('--device', type=str, default='cuda', help='device to infer model')
    parser.add_argument('--text_field', type=str, default='text', help='field in the dataset containing the text')
    parser.add_argument('--batch_size', type=int, default=256, help='field in the dataset containing the text')
    parser.add_argument('--repo', type=str, help='push dataset to hf')
    parser.add_argument('--num_proc', type=int, default=16, help='push dataset to hf')

    # Parse the command-line arguments
    args = parser.parse_args()
    # Setup configuration and model
    config = AutoConfig.from_pretrained("nvidia/multilingual-domain-classifier")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/multilingual-domain-classifier")
    model = CustomModel.from_pretrained("nvidia/multilingual-domain-classifier")
    model.to(args.device)
    model.eval()
    ds = load_dataset(args.dataset, split = args.split)
    tokenized_dataset = ds.map(tokenize_function, args.num_proc, batched=True, remove_columns=ds.column_names)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # 4. Create a DataLoader for batch inference
    batch_size = args.batch_size
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
    ds = ds.add_column("domains", all_predictions)
    ds.push_to_hub(args.repo)
if __name__ == '__main__':
    main()