"""
Embed

This module contains encoding functions to get embeddings
"""
import torch
import random
from numpy import dot
from numpy.linalg import norm
import numpy as np

device = "cuda:1" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

def cos(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0].cpu().detach().numpy()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).cpu().detach().numpy()

#CLS Pooling - Take output from first token
def cls_pooling(model_output):
    return model_output.last_hidden_state[:,0].cpu().detach().numpy()


def encode(query,tokenizer,encoder,model_name):
    """Encoding query with encoder of model_name and tokenizer"""
    set_seed(1234)
    q_token = tokenizer(query,padding=True,truncation=True,return_tensors="pt").to(device)
    set_seed(1234)        
    with torch.no_grad():
        model_output = encoder(**q_token)
    if "LaBSE" in model_name:
        q_emb = model_output.pooler_output
        q_emb = torch.nn.functional.normalize(q_emb).cpu().detach().numpy()
    elif "cocondenser" in model_name or "cocodr" in model_name:
        q_emb = max_pooling(model_output, q_token['attention_mask'])
        #q_emb = m_plus(q_emb)
    elif "multi-qa-mpnet" in model_name:
        q_emb = cls_pooling(model_output)
    else:      
        q_emb = mean_pooling(model_output, q_token['attention_mask'])
    return q_emb