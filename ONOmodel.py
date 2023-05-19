import math
import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.nn import functional as F
from torch.nn import GELU, ReLU, Tanh, Sigmoid
from torch.nn.utils.rnn import pad_sequence
from sympy.matrices import Matrix, GramSchmidt
from torch.autograd import Variable
from torchvision.models.vision_transformer import MLPBlock
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinearSelfAttention
from timm.models.layers import trunc_normal_

#n_hidden:128 burger: 0.00976(x as query) 0.00932(fx as query)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=421*421):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/2, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


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
        
                     
class Attn(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        #mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        #head_dim: int,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        space_dim = 1
    ):
        super().__init__()
        self.attention_dropout = attention_dropout
        
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(2*hidden_dim, hidden_dim)
        )
        self.PE = PositionalEncoding(d_model=hidden_dim, dropout=0.0)
        self.attn = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dim_head = hidden_dim // num_heads)
        self.space_dim = space_dim
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

    def forward(self, input, freq_x = None, freq_y = None):

        x = self.ln_1(input)
        if self.space_dim == 1:  
            x = self.PE(x)

        elif self.space_dim == 2:
            x = apply_2d_rotary_pos_emb(x, freq_x, freq_y)

        x = self.attn(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y

        return x             


class ONOBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        attention_dropout: float,
        ortho = False,
        act = 'gelu',
        space_dim = 1,
    ):
        super().__init__()

        self.attention_dropout = attention_dropout
        self.ortho = ortho
        self.nums_block = space_dim
        self.space_dim = space_dim

        self.Attn = Attn(num_heads = num_heads, hidden_dim=hidden_dim, dropout= dropout, attention_dropout= attention_dropout, space_dim=space_dim)
        
        self.act = nn.GELU()
        self.register_parameter("mu", nn.Parameter(torch.zeros(hidden_dim)))

    def forward(self, x, fx, freq_x = None, freq_y = None):

        x = self.Attn(x, freq_x = freq_x, freq_y = freq_y)
        
        x_ = orthogonal_qr(x)  if self.ortho else x
        
        fx = (x_*torch.nn.functional.softplus(self.mu))@(x_.transpose(-2,-1)@fx)

        fx = self.act(fx)

        return x , fx



class ONO(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 ffn_dropout=0.0,
                 attn_dropout=0.0,
                 n_head=8,
                 Time_Input = False,
                 ortho = False,
                 act = 'gelu',
                 res = 256,
                 ):
        super(ONO, self).__init__()

        self.space_dim = space_dim
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        
        self.preprocess_x = FeedForward(dim = space_dim, hidden_dim= n_hidden, out_dim = n_hidden//2)
        self.preprocess_f = FeedForward(dim = 1, hidden_dim= n_hidden, out_dim = n_hidden//2)

        self.blocks = nn.Sequential(*[ONOBlock(num_heads = n_head, hidden_dim=n_hidden, dropout= ffn_dropout, attention_dropout= attn_dropout , ortho = ortho , act = act, space_dim=space_dim) for _ in range(n_layers)])
        self.blocks[-1].act=nn.Identity()

        self.emb_module_f = RotaryEmbedding(n_hidden // space_dim, min_freq=1.0/res)

        self.__name__ = 'ONO'
        
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.1)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)  

    # x: B , N*N , space_dim fx: B , N*N , 1  , T : B , 1
    def forward(self, x, fx ,T = None):

        # 1d情况下分开升维再concate比concate后升维效果更好,1d情况升维不需要mlp只需要单层linear便可达到需要的效果
        if self.space_dim == 1:
            x = self.preprocess_x(x)
            Input_f = self.preprocess_f(fx)
            x = torch.cat((x, Input_f),-1)

        elif self.space_dim == 2 :
            freq_x = self.emb_module_f.forward(x[..., 0], x.device)
            freq_y = self.emb_module_f.forward(x[..., 1], x.device)
            
            x = self.preprocess_x(x)
            Input_f = self.preprocess_f(fx)
            x = torch.cat((x, Input_f),-1)   
            
        
        if self.Time_Input == False:
            for block in self.blocks:
                if self.space_dim == 2:
                    x ,fx = block(x, fx, freq_x = freq_x, freq_y = freq_y)
                elif self.space_dim == 1:
                    x ,fx = block(x, fx)

        else :
            Time_emb = timestep_embedding(T , self.n_hidden).repeat(1,x.shape[1],1)
            Time_emb = self.tim_fc(Time_emb)
            for block in self.blocks:
                x ,fx = block(x, fx)   
                    
        return fx
    
