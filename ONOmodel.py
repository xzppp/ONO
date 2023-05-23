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


def orthogonal_qr(x):
    # shape = x.shape
    # reshaped_x = x.reshape(-1, shape[-2], shape[-1])
    Q, _ = torch.linalg.qr(x,mode='reduced')  # 使用 PyTorch 的 QR 分解函数
    return Q

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



class ONOBlock_linear_attn(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            # mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            # head_dim: int,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            ortho=False,
            act='gelu'
    ):
        super().__init__()

        self.attention_dropout = attention_dropout
        self.ortho = ortho
        self.n_heads = num_heads
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.ln_f = norm_layer(hidden_dim)
        # self.mlp1 = MLPBlock(hidden_dim, hidden_dim, dropout)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        # self.mlp2 = MLPBlock(hidden_dim, hidden_dim, dropout)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim)
        )

        self.query1 = nn.Linear(hidden_dim, hidden_dim)
        self.key1 = nn.Linear(hidden_dim, hidden_dim)
        self.value1= nn.Linear(hidden_dim, hidden_dim)

        self.query2 = nn.Linear(hidden_dim, hidden_dim)
        self.key2 = nn.Linear(hidden_dim, hidden_dim)
        self.value2 = nn.Linear(hidden_dim, hidden_dim)

        self.attn = LinearSelfAttention(hidden_dim, causal=False, heads=num_heads, dim_head=hidden_dim // num_heads)
        if act == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Sigmoid()

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.002)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def linear_attention(self, q, k, v, n_head=1):
        B, T1, C = q.size()
        _, T2, _ = k.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = q.view(B, T1, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T2, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T2, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)  #
        k_cumsum = k.sum(dim=-2, keepdim=True)
        D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)  # normalized

        context = k.transpose(-2, -1) @ v
        y = (q @ context) * D_inv

        # output projection
        y = rearrange(y, 'b h n d -> b n (h d)')

        return y


    def forward(self, input: torch.Tensor, fx):
        # torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        q1, k1, v1 = self.query1(x), self.key1(x), self.value1(x)
        x = self.mlp1(self.linear_attention(q1, k1, v1,self.n_heads)) + x

        # x = self.dropout(x)


        x = self.ln_2(x)

        fx = self.ln_f(fx)
        q2 = self.query2(fx)
        x = orthogonal_qr(x) if self.ortho else x
        k2, v2 =  self.key2(x), self.value2(x)

        fx = self.mlp2(self.linear_attention(q2, k2, v2, self.n_heads)) + fx


        return x, fx



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
        ortho = False,
        act = 'gelu'
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

        self.mlp2 = MLPBlock(hidden_dim, hidden_dim, dropout)
        self.attn = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dim_head = hidden_dim // num_heads)
        if act == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Sigmoid()

        self.initialize_weights()
    
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
        # x = torch.matmul(x, torch.matmul(x.transpose(-2,-1), x))
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y
        
        x_ = orthogonal_qr(x)  if self.ortho else x
        #fx = fx.view(fx.shape[0],fx.shape[1],1)
        fx = torch.matmul(x_,torch.matmul(x_.transpose(-2,-1),fx))

        fx = self.act(fx)
    
        return x , fx



class ONO(nn.Module):
    def __init__(self,
                 space_dim=2,
                 f_dim = 1,
                 out_dim = 1,
                 n_layers=4,
                 n_hidden=128,
                 ffn_dropout=0.0,
                 attn_dropout=0.0,
                 n_head=1,
                 Time_Input = False,
                 ortho = False,
                 act = 'gelu',
                 #n_experts = 2,
                 #n_inner = 4,
                 #attn_type='linear',
                 ):
        super(ONO, self).__init__()

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.f_dim = f_dim
        # self.preprocess = nn.Linear(space_dim , n_hidden)
        self.preprocess = nn.Sequential(nn.Linear(space_dim + f_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden,n_hidden))

        self.f_mlp = nn.Sequential(nn.Linear(f_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_hidden))
        self.out_mlp = nn.Sequential(nn.Linear(n_hidden, n_hidden),nn.GELU(),nn.Linear(n_hidden,out_dim))

        self.blocks = nn.Sequential(*[ONOBlock_linear_attn(num_heads = n_head, hidden_dim= n_hidden, dropout= ffn_dropout, attention_dropout= attn_dropout , ortho = ortho , act = act) for _ in range(n_layers)])
        # self.blocks[-1].act=nn.Identity()
        
        # self.apply(self._init_weights)

        self.__name__ = 'ONO'

    # x: B , N*N , space_dim fx: B , N*N , 1  , T : B , 1
    def forward(self, x, fx ,T = None):

        x = self.preprocess(torch.cat([x, fx],dim=-1))
        x = self.preprocess(x)
        # fx = self.f_mlp(fx)
        fx = x
        if self.Time_Input == False:
            for block in self.blocks:
                x ,fx = block(x, fx)
        else :
            Time_emb = timestep_embedding(T , self.n_hidden).repeat(1,x.shape[1],1)
            for block in self.blocks:
                x ,fx = block(x + Time_emb, fx)
            
        fx = self.out_mlp(fx)
        return fx
    
