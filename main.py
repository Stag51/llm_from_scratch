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
    


#######     Building LLM         ######


# Createing Normalization Layer
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super(LayerNorm, self).__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# GELU Activation

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5 * x * (1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
# Feed Forward NetWork

class FeedForwardGELU(nn.Module):
    def __init__(self, cfg):
        super(FeedForwardGELU, self).__init__()
        emb_dim = cfg.emb_dim

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
    def forward(self, x):
        return self.layers(x)
    

# Transfer Block

class TransferBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg.emb_dim,
            d_out = cfg.emb_dim,
            context_length= cfg.context_length,
            num_heads= cfg.n_heads,
            dropout= cfg.dropout,
            qkv_bias= cfg.qkv_bias
        )
        self.ff = FeedForwardGELU(cfg)
        self.norm1 = LayerNorm(self.emb_dim)
        self.norm2 = LayerNorm(self.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)


    def forward(self, x):
        resid_conn = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + resid_conn

        resid_conn = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + resid_conn
        return x
    
# Building the main GPT Architecture
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.dropout_emb = nn.Dropout(cfg.drop_rate)
        self.transformer_block = nn.Sequential( *[TransferBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_ff = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias = False)

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        tok_embeds = self.tok_emb(idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device = idx.device))

        x = tok_embeds + pos_embeds
        x = self.dropout_emb
        x = self.transformer_block
        x = self.final_norm
        logits = self.out_ff(x)
        return logits
    
#############           Training loop       ############

# Generate Function
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        # Getting the prediction
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim= -1, keepdim= True)
        idx = torch.cat((idx, idx_next), dim = 1)

    return idx

def generate(model, idx, max_new_tokens, context_size, temperature = 0.0, top_k = None, eos_id = None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim= -1)
            idx_next = torch.multinomial(probs, num_samples=-1)

        else: 
            idx_next = torch.argmax(logits, dim = -1, keepdim= True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim = 1)


    return idx

def generate_print_sample(model, tokenizer, device, start_context, temperature, top_k, eos_id):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_tokens_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model= model,
            idx = encoded,
            max_new_tokens= 50,
            context_size= context_size,
            temperature= temperature,
            top_k= top_k,
            eos_id= eos_id
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


# Calculating the training and validation losses

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()

        else:
            break

    return total_loss / num_batches

# Training and Evaluation Methode

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, temperature, top_k, eos_id):
    train_losses , val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:03d}): "
                  f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        # Printing the sample text
        print("Example: ")
        generate_print_sample(
            model, 
            tokenizer,
            device,
            start_context,
            temperature,
            top_k,
            eos_id
        )
        print('-*-'*10)

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss