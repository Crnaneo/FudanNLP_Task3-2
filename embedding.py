from transformers import AutoTokenizer
from torch import nn;
import torch;
import math;


class Embedding(nn.Module):
    def __init__(self,max_len=1024,dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True);
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = 1e9
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]', '<SOS>']})
        self.dim = dim;
        self.token_embedding = nn.Embedding(len(self.tokenizer),self.dim);
        self.position_embedding = nn.Embedding(max_len,self.dim);

    def tokenize(self,data,device=torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))):
        tokens = self.tokenizer.encode(data, add_special_tokens=True)
        return tokens;

    def forward(self,tokens,device=torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))):
        batch_size, seq_len = tokens.size();
        positions = torch.arange(0,seq_len,device=device).expand(batch_size,seq_len)
        out = self.token_embedding(tokens)*math.sqrt(self.dim)+self.position_embedding(positions)
        return out;