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
# Usage:  
-Heuristic Filtering using Data-juicer: run processing_data.ipynb  
-Run Model-based filtering: python toxic_filter.py 

# Deduplication
Locality Sensitive Hashing: Minhash
