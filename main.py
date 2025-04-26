#########      Importing all the required libraries    ##########
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import tiktoken as ttk
from tqdm import tqdm
import os
import urllib.request
from dataclasses import dataclass, field, asdict, replace
from typing import Dict


# Getting Raw Text 

if not os.path.exists("the-verdict.txt"):
    url  = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 512
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = float

    def to_dict(self) -> dict:
        return asdict(self)
    
    def __repr__(self) -> str:
        config_dict = self.to_dict()
        formatted_items = [f'"{key}": {repr(value)}' for key, value in config_dict.items()]
        return "GPT_CONFIG_124M = {\n    " + ",\n    ".join(formatted_items) + "\n}"
        
@dataclass
class DataConfig:
    dataPath: str = r'the-verdict.txt'
    max_lenght: int = GPTConfig.context_length
    batch_size: int = 64
    train_ratio: float = 0.90
    stride: int = GPTConfig.context_length

DataConfig = DataConfig()
GPTConfig = GPTConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding= "utf-8") as f:
            raw_text = f.read()
        return raw_text
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
    
def text_to_tokens_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())