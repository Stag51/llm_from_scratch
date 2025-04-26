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








#############     Stage-I Data preparation and sampling        ############

class LLMDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length: int, stride: int):
        self.tokenizer = tokenizer
        token_ids = tokenizer.encode(txt)
        self.input_ids = []
        self.target_ids = []

        for i in tqdm(range(0, len(token_ids) - max_length, stride)):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def LLM_DataLoader(txt, tokenizer, batch_size: int, max_length: int, stride: int,
                   shuffle: bool = True, drop_last: bool = True):
    llmdataset = LLMDataset(txt, tokenizer, max_length, stride)
    llmdataloader = DataLoader(llmdataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return llmdataloader

# Loading the verdict.txt 

raw_data = read_txt(DataConfig.dataPath)
tokenizer = ttk.get_encoding("gpt2")

total_token = len(tokenizer.encode(raw_data))
print(f"-> Number of Characters : {len(raw_data)}\n-> Number of Tokens : {total_token}")

# Splitting Data into training and validation

train_ratio = DataConfig.train_ratio
split_idxs = int(train_ratio * len(raw_data))
train_data = raw_data[:split_idxs]
val_data = raw_data[split_idxs:]

print(f"Length of Training Data : {len(train_data)}")
print(f"Length of Validation Data: {len(val_data)}")

# Sanity Check 
if total_token * (train_ratio) < GPTConfig.context_length:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPTConfig.context_length or "
          "increase the `training_ratio`")
if total_token * (1-train_ratio) < GPTConfig.context_length:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPTConfig.context_length` or "
          "decrease the `training_ratio`")
    
# Processing data to use in LLm as input

train_dataloader = LLM_DataLoader(
    txt = train_data,
    tokenizer= tokenizer,
    max_length= DataConfig.max_lenght,
    batch_size= DataConfig.batch_size,
    stride= DataConfig.stride,
    shuffle= False,
    drop_last= False
)

print("View Example:")
dataiter = iter(train_dataloader)
firstbatch = next(dataiter)
print(f"Inputs: \n{firstbatch[0]} \ntarget: \n{firstbatch[1]}")
print(firstbatch[0].shape)

val_dataloader = LLM_DataLoader(
    txt = val_data,
    tokenizer= tokenizer,
    max_length= DataConfig.max_lenght,
    batch_size= DataConfig.batch_size,
    stride= DataConfig.stride,
    shuffle= False,
    drop_last= False
)

dataiter = iter(val_dataloader)
firstbatch = next(dataiter)
print(firstbatch[0].shape)











######       Creating Attention Mechanism  #########

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out:int, context_length: float, dropout: float,num_heads: int, qkv_bias: bool = False):
        super(MultiHeadAttention, self).__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dims = d_out // num_heads

        self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            'mask',
            torch.trill(torch.ones(context_length, context_length)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        batches, num_tokens, dim_in = x.shape
        # Linear Projections
        queries = self.w_queries(x)
        keys = self.w_keys(x)
        values = self.w_values(x)

        # Reshaping and Transposing for multihead Attenstion

        queries = queries.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention Score Calculations

        attn_score = (queries @ keys.transpose(2, 3)) / (self.head_dims ** 0.5)

         # Applying Mask
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :num_tokens, :num_tokens] == 0, float('-inf'))

        # Softmax to get attention weights

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batches, num_tokens, self.d_out)


        # Final Linear Prediction
        context_vec = self.out_proj(context_vec)
        return context_vec