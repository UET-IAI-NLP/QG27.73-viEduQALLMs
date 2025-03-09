from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import Dataset, DatasetDict, load_dataset
from accelerate import Accelerator
import huggingface_hub
import torch
import json
import os
import wandb
import random
import yaml
from datetime import datetime



def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
config = load_config("config.yml")


huggingface_hub.login(config['huggingface_access_key'])

# Hyper parameter
model_name = config['model_name']
data_path = config['data_path']
cache_dir = config['cache_dir']
BLOCK_SIZE = config['BLOCK_SIZE']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device =',device)


model = AutoModelForCausalLM.from_pretrained( model_name,torch_dtype=torch.bfloat16, cache_dir=cache_dir)
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# change the padding tokenizer value
tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
model.config.pad_token_id = tokenizer.pad_token_id 
tokenizer.padding_side = 'right' 

# def read_data(path):
#     data = []
#     with open(path, 'r', encoding='utf-8') as file:
#         for line in file:
#             data.append(json.loads(line))
#     return data
# path = "/workspace/nmquy/LLM_QA/CPT/uet_iai_nlp_data_for_llms/final_filtered/c4_part1_nontoxic_dedup_stats.jsonl"
# data = read_data(path)


# import pandas as pd 
# file_path = "/workspace/nmquy/LLM_QA/CPT/uet_iai_nlp_data_for_llms/sentence_split/part1.parquet"

# df = pd.read_parquet(file_path, engine='pyarrow')
# df.rename(columns={"Sentence" : "text"}, inplace=True)
# df.head()

# dataset_train = Dataset.from_pandas(df, preserve_index=False)
# dataset = DatasetDict({
#     "train":dataset_train
# })

dataset = load_dataset(data_path)

# Tokenizer data
def tokenizer_function(samples):
#    text = str(samples["text"])
#    text = text if text is not None else ""
    return tokenizer(samples["text"])

tokenizer_data = dataset.map(tokenizer_function,
                             batched=True,
                             remove_columns=dataset["train"].column_names)


# Preprocess tokenizer
def group_texts(samples):
    # samples chua input_ids va attention_mask, ta se noi tat ca input_ids, attention_mask thanh 1 list duy nhat
    concatenated_examples = {k: sum(samples[k], []) for k in samples.keys()}
    # Tinh so luong block de chia input_ids thanh cac block co kich thuoc block_size
    total_length = len(concatenated_examples[list(samples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Chia input_ids ban dau thanh cac block voi ca attention_ids
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    # Xac dinh labels chinh la input_ids trong bai toan CausalLM
    result["labels"] = result["input_ids"]
    return result

print("Process data")
clm_dataset = tokenizer_data.map( group_texts, batched=True, num_proc=config['num_proc'])
# Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

output_dir = config['output_path'] + "output_edu_" + datetime.now().strftime("%Y-%m-%d_%H-%M")

# Train args
args = TrainingArguments(
    report_to='wandb',
    bf16=True,
    deepspeed=config['deepspeed'],
    warmup_steps=config['warmup_steps'],
    per_device_train_batch_size=config['train_batch_size'],
    per_device_eval_batch_size=config['eval_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    gradient_checkpointing=config['gradient_checkpointing'],
    weight_decay=config['weight_decay'],
    num_train_epochs=config['num_train_epochs'],
    learning_rate=config['learning_rate'],
    logging_steps=config['logging_steps'],
    save_strategy=config['save_strategy'],
    save_steps=200,
    save_total_limit=config['save_total_limit'],
    output_dir=output_dir,
    evaluation_strategy=config['evaluation_strategy'],
    hub_private_repo=False,
    push_to_hub=True,
    hub_model_id=config["hub_model_id"]
)

# Training
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=clm_dataset["train"],
    eval_dataset=clm_dataset["train"],
    data_collator=data_collator,
)
trainer.train()

save_path = config['save_path'] + "save_model_pretrain_" + datetime.now().strftime("%Y-%m-%d_%H-%M")

trainer.save_model()
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
