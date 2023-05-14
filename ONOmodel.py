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

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

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

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride=1
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
        
    def forward(self, x):
        out = self.conv1(x)
        out = nn.functional.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = nn.functional.relu(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)
        return nn.functional.relu(x+out, True)

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


def linear_attention(query, key, value,
                     mask=None, dropout=None):
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = query.size(-2)

    query = query.softmax(dim=-1)
    key = key.softmax(dim=-2)
    scores = torch.matmul(key.transpose(-2, -1), value)

    if mask is not None:
        raise RuntimeError("linear attention does not support casual mask.")

    p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(query, p_attn)
    return out, p_attn

class LinearCrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd = 64, n_head = 8, n_inputs = 2, attn_pdrop = 0.0):
        super(LinearCrossAttention, self).__init__()
        
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.query = nn.Linear(n_embd, n_embd)
        self.keys = nn.ModuleList([nn.Linear(n_embd, n_embd) for _ in range(n_inputs)])
        self.values = nn.ModuleList([nn.Linear(n_embd, n_embd) for _ in range(n_inputs)])
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.n_inputs = n_inputs

        self.attn_type = 'l1'

    '''
        Linear Attention and Linear Cross Attention (if y is provided)
    '''
    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.size()
        #freq_x = repeat(freq_x, 'b n d -> b h n d', h = self.n_head)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        #q = apply_rotary_pos_emb(q, freq_x)
        q = q.softmax(dim=-1)
        out = q
        for i in range(self.n_inputs):
            _, T2, _ = y[i].size()
            k = self.keys[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            #k = apply_rotary_pos_emb(k, freq_x)
            v = self.values[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            #v = apply_rotary_pos_emb(v, freq_x)
            k = k.softmax(dim=-1)  #
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)  # normalized
            out = out +  1 * (q @ (k.transpose(-2, -1) @ v)) * D_inv


        # output projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out

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

class CrossAttnBlock(nn.Module):

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
        
        self.preprocess = nn.Sequential(nn.Linear(1, hidden_dim))
        self.proprocess = nn.Sequential(nn.Linear(hidden_dim, 1))
        
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, hidden_dim, dropout)
        
        self.ln_3 = norm_layer(hidden_dim)
        self.ln_4 = norm_layer(hidden_dim)
        self.mlp_2 = MLPBlock(hidden_dim, hidden_dim, dropout)        
        

        self.attn = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dim_head = hidden_dim // num_heads)
        self.cattn = LinearCrossAttention(n_embd = hidden_dim, n_head = 8, n_inputs = 2, attn_pdrop = attention_dropout)

        #self.initialize_weights()
        self.register_parameter("mu", nn.Parameter(torch.zeros(hidden_dim)))

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

    def forward(self, input, Input_f):
        #torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        #Input_f = self.ln_1(Input_f)

        x = self.ln_1(input)
        
        cross_input = torch.stack([x,Input_f], 0)
        x = self.cattn(Input_f,cross_input)
        
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y
        
        return x        
             
             
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
        self.PE = PositionalEncoding(d_model=hidden_dim, dropout=0.0)
        self.attn = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dim_head = hidden_dim // num_heads)

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

    def forward(self, input):
        #torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        #Input_f = self.ln_1(Input_f)

        x = self.ln_1(input)
        x = self.PE(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y
        
        return x             


class vit(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        #mlp_dim: int,
        dropout = 0.0,
        attention_dropout = 0.0,
        #head_dim: int,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.attention_dropout = attention_dropout
        self.resblock = residual_block(in_channel = 1, out_channel = hidden_dim, same_shape=True)
        
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, hidden_dim, dropout)
              
        self.attn = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dim_head = hidden_dim // num_heads)
        #self.PE = PositionalEncoding(d_model=hidden_dim, dropout=0.0)
        
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

    def forward(self, input, freq_x, freq_y):
        
        input = self.resblock(input.unsqueeze(1))
        input = rearrange(input,'b h x y -> b (x y) h')
        
        x = self.ln_1(input)
        x = apply_2d_rotary_pos_emb(x, freq_x, freq_y)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y        
        
        return input  

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
        act = 'gelu',
        space_dim = 1
    ):
        super().__init__()

        self.attention_dropout = attention_dropout
        self.ortho = ortho
        self.space_dim = space_dim
        
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, hidden_dim, dropout)
        
        self.ln_3 = norm_layer(hidden_dim)
        
        self.mlp_1 = MLPBlock(hidden_dim, hidden_dim, dropout)

        self.cattn = LinearCrossAttention(n_embd = hidden_dim, n_head = num_heads, n_inputs = 2, attn_pdrop = 0.0)
        self.PE = PositionalEncoding(d_model=hidden_dim, dropout=0.0)

        self.attn = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dim_head = hidden_dim // num_heads)
        self.preprocess = nn.Linear(1,hidden_dim)
        self.preprocess_f = nn.Linear(1,hidden_dim)
        self.proprocess = nn.Linear(hidden_dim,1)
        
        self.act = nn.LeakyReLU()

        self.initialize_weights()
        self.register_parameter("mu", nn.Parameter(torch.zeros(hidden_dim)))
    
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

    def forward(self, input, fx, Input_f, freq_x = None, freq_y = None):
        #torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        #Input_f = self.ln_1(Input_f)


        if self.space_dim == 1:    
            x = self.PE(input)
            Input_f = self.PE(Input_f)
            
        elif self.space_dim == 2:
            x = apply_2d_rotary_pos_emb(input, freq_x, freq_y)
            Input_f = apply_2d_rotary_pos_emb(Input_f, freq_x, freq_y)
            
        x = self.ln_1(x)
        Input_f = self.ln_2(Input_f)            
        #Input_f = self.preprocess_f(Input_f)
        cross_input = torch.stack([Input_f,x],0)
        x = self.cattn(Input_f ,cross_input)

        #x = self.attn(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_3(x)
        y = self.mlp(y)
        x = x + y

        x_ = orthogonal_qr(x)  if self.ortho else x

        #fx = self.preprocess(fx)
        fx = (x_*torch.nn.functional.softplus(self.mu))@(x_.transpose(-2,-1)@fx)
        #fx = self.proprocess(fx)
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
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 #n_experts = 2,
                 #n_inner = 4,
                 #attn_type='linear',
                 ):
        super(ONO, self).__init__()

        self.space_dim = space_dim
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        
        #self.preprocess_pos = nn.Linear(space_dim, n_hidden)
        self.preprocess_x = nn.Linear(space_dim, n_hidden)
        self.preprocess_f = nn.Linear(1, n_hidden)
        
        self.posmlp = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_hidden))
        self.fmlp = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_hidden))
        #self.posattn = Attn(hidden_dim=n_hidden, num_heads= n_head, dropout=ffn_dropout, attention_dropout= attn_dropout)
        self.fattn = Attn(hidden_dim=n_hidden, num_heads= n_head, dropout=ffn_dropout, attention_dropout= attn_dropout)

        self.blocks = nn.Sequential(*[ONOBlock(num_heads = n_head, hidden_dim= n_hidden, dropout= ffn_dropout, attention_dropout= attn_dropout , ortho = ortho , act = act, space_dim=space_dim) for _ in range(n_layers)])
        self.blocks[-1].act=nn.Identity()
        
        self.PE = PositionalEncoding(d_model=n_hidden,dropout= 0.0)
        #self.finalprocess = nn.Sequential(nn.Linear(1 , n_hidden), nn.GELU(), nn.Linear(n_hidden, n_hidden), nn.GELU(), nn.Linear(n_hidden,1))

        self.emb_module_f = RotaryEmbedding(n_hidden // space_dim, min_freq=1.0/res)
        self.tim_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden,n_hidden))
        # self.apply(self._init_weights)
        
        self.vit = vit(hidden_dim=n_hidden,num_heads=n_head)
        
        self.ln = norm_layer(n_hidden)

        self.__name__ = 'ONO'

    # x: B , N*N , space_dim fx: B , N*N , 1  , T : B , 1
    def forward(self, x, fx ,T = None):

        if self.space_dim == 1:
            Input_f = self.preprocess_f(fx)
            Input_f = self.fattn(Input_f)

        #Cross_input = torch.stack([pos, Input_f],0)
        #Input_f = fx
        elif self.space_dim == 2 :   
            freq_x = self.emb_module_f.forward(x[..., 0], x.device)
            freq_y = self.emb_module_f.forward(x[..., 1], x.device)
            Input_f = self.vit(fx.squeeze(-1), freq_x, freq_y)         
            fx = rearrange(fx,'b x y d-> b (x y) d')
            #Input_f = apply_2d_rotary_pos_emb(Input_f, freq_x, freq_y)
                      
        x = self.preprocess_x(x)
        
        if self.Time_Input == False:
            for block in self.blocks:
                if self.space_dim == 1:
                    x ,fx = block(x, fx, Input_f)
                elif self.space_dim == 2 :
                    x ,fx = block(x, fx, Input_f, freq_x, freq_y)

        else :
            Time_emb = timestep_embedding(T , self.n_hidden).repeat(1,x.shape[1],1)
            Time_emb = self.tim_fc(Time_emb)
            for block in self.blocks:
                if self.space_dim == 1:
                    x ,fx = block(x, fx, Input_f)
                elif self.space_dim == 2 :
                    x ,fx = block(x, fx, Input_f, freq_x, freq_y)        
                    
        #fx = self.finalprocess(fx)
        return fx
    
