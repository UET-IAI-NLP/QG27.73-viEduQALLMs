# Usage
Tokenizer
```shell
python Tokenizer.py  --h
usage: Tokenizer.py [-h] [--split SPLIT] [--tokenizer TOKENIZER] [--text-field TEXT_FIELD] dataset

Count tokens in a Hugging Face dataset.

positional arguments:
  dataset               Hugging Face dataset path (e.g. "imdb")

optional arguments:
  -h, --help            show this help message and exit
  --split SPLIT         Dataset split (default: train)
  --tokenizer TOKENIZER
                        Tokenizer name (HF model or "gpt-4"/"gpt-3.5-turbo")
  --text-field TEXT_FIELD
                        Field containing text (default: text)
```
Toxic filter
```shell
python toxic_filter.py --h
usage: toxic_filter.py [-h] dataset

Example script with argument parser.

positional arguments:
  dataset     huggingface dataset path

optional arguments:
  -h, --help  show this help message and exit
```
Quality filter
```shell
python quality_filter.py --h
usage: quality_filter.py [-h] [--model_path MODEL_PATH] [--text_field TEXT_FIELD] [--target_label TARGET_LABEL] dataset

Example script with argument parser.

positional arguments:
  dataset               huggingface dataset path

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to the trained FastText model
  --text_field TEXT_FIELD
                        field in the dataset containing the text
  --target_label TARGET_LABEL
                        label representing high-quality text
```
Deduplication
```shell
python dedup.py --h
usage: dedup.py [-h] --base-path BASE_PATH --data-file DATA_FILE --output-path OUTPUT_PATH [--split SPLIT]
                [--cache-dir CACHE_DIR] [--column COLUMN] [--num-perm NUM_PERM] [--threshold THRESHOLD] [--b B] [--r R]
                [--batch-size BATCH_SIZE]

Run text_dedup.minhash for a single file.

optional arguments:
  -h, --help            show this help message and exit
  --base-path BASE_PATH
                        Base directory containing the data file (passed to text_dedup --path).
  --data-file DATA_FILE
                        The specific input data file name (relative to base-path, e.g., "train-00001-of-00016.parquet").
  --output-path OUTPUT_PATH
                        The full path for the output results (e.g., "/content/drive/MyDrive/dedup/output_file_1.jsonl").
  --split SPLIT         Dataset split to process.
  --cache-dir CACHE_DIR
                        Directory for caching intermediate results.
  --column COLUMN       Column containing text data in the Parquet file.
  --num-perm NUM_PERM   Number of permutations for MinHash.
  --threshold THRESHOLD
                        Jaccard similarity threshold for deduplication.
  --b B                 Number of bands for LSH.
  --r R                 Number of rows per band for LSH.
  --batch-size BATCH_SIZE
                        Processing batch size for text_dedup.
```
# Pretraining dataset
Data sources come from the following categories:
## Dataset Summary

| Dataset Name | Original Size (Storage) | Original Size (Samples) | Pre-Processed |
|--------------|----------------|----------------|---------------|
| [VNU web crawl](https://huggingface.co/datasets/group2sealion/vnu_crawl_clean)     |  113MB(json)        | 19653      | Yes           |
|  [CC100 vi](https://huggingface.co/datasets/statmt/cc100)   | 28GB(txt.gz)         | 100M         | No            |
| [C4_vi](https://huggingface.co/datasets/allenai/c4)    | 116GB(parquet)         |  78M       |      26%      |

1. Web crawler dataset:  
- Website UET (ĐH Công nghệ): tuyensinh.uet.vnu.edu.vn; new.uet.vnu.edu.vn
- Website HUS (ĐH KHTN): hus.vnu.edu.vn
- Website EUB (ĐH Kinh tế): ueb.vnu.edu.vn
- Website IS (ĐH Quốc tế): is.vnu.edu.vn
- Website Eduacation (ĐH Giáo dục): education.vnu.edu.vn
- Website NXB ĐHQG: press.vnu.edu.vn   
[List domain web crawler](https://docs.google.com/spreadsheets/d/1zbkltkSPRm6f48Lb1Jo3Njq1-LrSd8H6/edit?gid=409337688#gid=409337688)  
2. CC100:  
[link to CC100 vi](https://huggingface.co/datasets/statmt/cc100)  
3. C4_vi:  
  [link to C4_vi](https://huggingface.co/datasets/allenai/c4)
# Tokenizer  
We use tokenizer from [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)  
# Training models  
We apply continual pretraining to [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) on our processed dataset. The training process last 10 days on 2 Nvidia A100 GPUs and we achieve the average training loss of 1.9
# Filtering models  
1. [Domain classification model](https://huggingface.co/nvidia/multilingual-domain-classifier):  
- Model type: deberta-v2
- Params: 278 M
- Size: 1.11 GB
2. [Quality classification model](https://huggingface.co/zerostratos/quality_classification) :
- Model type: fasttext
- Size: 2.02 GB
3. [Toxic detection model](https://huggingface.co/zerostratos/lstm)
- Model type: RoBERTa with classification layers
- Params: 136 M
- Size: 544 MB
# Deduplication
Locality Sensitive Hashing: Minhash  
# Usage:  
-Heuristic Filtering using Data-juicer: run processing_data.ipynb  
-Run Model-based filtering: python toxic_filter.py dataset     
# Result and Domain distribution:  
![image](https://github.com/user-attachments/assets/59ccdc70-5f0d-473b-b070-72939a7b6251)

![domains (2)](https://github.com/user-attachments/assets/f9ea0ec0-8b25-4577-939c-6ad1e02b8108)

