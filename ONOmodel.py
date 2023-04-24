import math
import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.nn import functional as F
from torch.nn import GELU, ReLU, Tanh, Sigmoid
from torch.nn.utils.rnn import pad_sequence
from sympy.matrices import Matrix, GramSchmidt

from torchvision.models.vision_transformer import MLPBlock
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinearSelfAttention
from timm.models.layers import trunc_normal_


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


def orthogonal_tensor(X):
    n, k = X.size(-2), X.size(-1)
    transposed = n < k
    if transposed:
        X = X.mT
        n, k = k, n
    # Here n > k and X is a tall matrix
    # We just need n x k - k(k-1)/2 parameters
    # 
    X = X.tril()
    if n != k:
        # Embed into a square matrix
        X = torch.cat([X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1)
    A = X - X.mH
    # A is skew-symmetric (or skew-hermitian)
    Q = torch.matrix_exp(A)

    if n != k:
        Q = Q[..., :k]
    # Q is now the size of the X (albeit perhaps transposed)
    if transposed:
        Q = Q.mT
    return Q



class ONOBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        #mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        #head_dim: int,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        ortho = False
    ):
        super().__init__()

        self.attention_dropout = attention_dropout
        self.ortho = ortho
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, hidden_dim, dropout)

        self.attn = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dim_head = hidden_dim // num_heads)
        
        self.act = nn.Sigmoid()

        #self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)  

    def forward(self, input: torch.Tensor , fx):
        #torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y
        
        if self.ortho :
            x = orthogonal_tensor(x)
        #fx = fx.view(fx.shape[0],fx.shape[1],1)
        fx = torch.matmul(torch.matmul(x,x.transpose(-2,-1)),fx)
        fx = self.act(fx)
    
        return x , fx


class ONO(nn.Module):
    def __init__(self,
                 space_dim=2,
                 n_layers=5,
                 n_hidden=64,
                 ffn_dropout=0,
                 attn_dropout=0,
                 n_head=8,
                 Time_Input = False,
                 ortho = False,
                 #n_experts = 2,
                 #n_inner = 4,
                 #attn_type='linear',
                 #act = 'gelu',
                 ):
        super(ONO, self).__init__()

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        
        self.preprocess = nn.Linear(space_dim , n_hidden)

        self.blocks = nn.Sequential(*[ONOBlock(num_heads = n_head, hidden_dim= n_hidden, dropout= ffn_dropout, attention_dropout= attn_dropout , ortho = ortho) for _ in range(n_layers)])
        self.blocks[-1].act=nn.Identity()
        
        # self.apply(self._init_weights)

        self.__name__ = 'ONO'

    # x: B , N*N , space_dim fx: B , N*N , 1  , T : B , 1
    def forward(self, x, fx ,T = None):

        x = self.preprocess(x)
        
        if self.Time_Input == False:
            for block in self.blocks:
                x ,fx = block(x, fx)
        else :
            Time_emb = timestep_embedding(T , self.n_hidden).repeat(1,x.shape[1],1)
            for block in self.blocks:
                x ,fx = block(x + Time_emb, fx)
            

        return fx
    
