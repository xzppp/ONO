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

from linear_attention_transformer.linear_attention_transformer import SelfAttention
from timm.models.layers import trunc_normal_


ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}


# n_hidden:128 burger: 0.00976(x as query) 0.00932(fx as query)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



'''
    A simple MLP class, includes at least 2 layers and n hidden layers
'''
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu',res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Linear(n_input, n_hidden)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])

        # self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])



    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            if self.res:
                x = self.act(self.linears[i](x)) + x
            else:
                x = self.act(self.linears[i](x))
            # x = self.act(self.bns[i](self.linears[i](x))) + x

        x = self.linear_post(x)
        return x




class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=421 * 421):
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
    def __init__(self, dim, min_freq=1 / 2, scale=1.):
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
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d // 2], t[..., d // 2:]

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
    Q, _ = torch.linalg.qr(x, mode='reduced')  # 使用 PyTorch 的 QR 分解函数
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




class LinearAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head):
        super(LinearAttention, self).__init__()
        # assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(0.0)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head

        self.attn_type = 'l1'

    '''
        Linear Attention and Linear Cross Attention (if y is provided)
    '''
    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)


        if self.attn_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)   #
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)       # normalized
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)  #
            D_inv = 1. / T2                                           # galerkin
        elif self.attn_type == "l2":                                   # still use l1 normalization
            q = q / q.norm(dim=-1,keepdim=True, p=1)
            k = k / k.norm(dim=-1,keepdim=True, p=1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)  # normalized
        else:
            raise NotImplementedError

        context = k.transpose(-2, -1) @ v
        y = self.attn_drop((q @ context) * D_inv + q)

        # output projection
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.proj(y)
        return y



class Attn(nn.Module):

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            # mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            # head_dim: int,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            space_dim=1
    ):
        super(Attn, self).__init__()
        self.attention_dropout = attention_dropout

        # Attention block
        # self.ln_1 = norm_layer(hidden_dim)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        # self.ln_2 = norm_layer(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.PE = PositionalEncoding(d_model=hidden_dim, dropout=0.0)
        # self.attn = SelfAttention(hidden_dim, causal=False, heads=num_heads, dim_head=hidden_dim // num_heads)
        # self.attn = LinearCrossAttention(hidden_dim, num_heads)
        self.attn = LinearAttention(hidden_dim, num_heads)
        self.space_dim = space_dim
        # self.initialize_weights()

    # def initialize_weights(self):
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y=None, freq_x=None, freq_y=None):

        x = self.ln_1(x)

        if self.space_dim == 1:
            x = self.PE(x)

        elif self.space_dim == 2:
            if (freq_x != None) and (freq_y != None) :
                x = apply_2d_rotary_pos_emb(x, freq_x, freq_y)


        x = self.attn(x, y) + x
        # x = self.dropout(x)
        # x = x + input

        # x = self.ln_2(x)
        x = self.mlp1(self.ln_2(x)) + x
        # x = x + y

        return x


class MultipleTensors():
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)


    def __getitem__(self, item):
        return self.x[item]

class CrossAttentionBlock(nn.Module):
    def __init__(self, n_embd, n_heads,n_inputs = 1):
        super(CrossAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2_branch = nn.ModuleList([nn.LayerNorm(n_embd) for _ in range(n_inputs)])
        self.n_inputs = n_inputs
        self.ln3 = nn.LayerNorm(n_embd)
        self.ln4 = nn.LayerNorm(n_embd)
        self.ln5 = nn.LayerNorm(n_embd)

        # self.ln6 = nn.LayerNorm(config.n_embd)      ## for ab study
        # if config.attn_type == 'linear':
        print('Using Linear Attention')
        self.selfattn = LinearAttention(n_embd, n_head=n_heads)
        # self.crossattn = LinearCrossAttention(n_embd, n_head=n_heads)
        self.crossattn = LinearAttention(n_embd, n_head=n_heads)

            # self.selfattn_branch = LinearAttention(config)


        # if config.act == 'gelu':
        self.act = GELU


        self.resid_drop1 = nn.Dropout(0.0)
        self.resid_drop2 = nn.Dropout(0.0)
        self.mlp1 = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            self.act(),
            nn.Linear(4* n_embd, n_embd),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(n_embd, 4 *n_embd),
            self.act(),
            nn.Linear(4* n_embd, n_embd),
        )


    def ln_branchs(self, y):
        return MultipleTensors([self.ln2_branch[i](y[i]) for i in range(self.n_inputs)])


    def forward(self, x, y):
        # x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln_branchs(y)))
        x = x + self.resid_drop1(self.crossattn(self.ln1(x)))
        x = x + self.mlp1(self.ln3(x))
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        x = x + self.mlp2(self.ln5(x))

        return x




class ONOBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            attention_dropout: float,
            ortho=False,
            act='gelu',
            space_dim=1,
    ):
        super().__init__()

        self.attention_dropout = attention_dropout
        self.ortho = ortho
        self.nums_block = space_dim
        self.space_dim = space_dim

        self.Attn1 = Attn(num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout,
                         attention_dropout=attention_dropout, space_dim=space_dim)

        self.Attn2 = Attn(num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout,
                         attention_dropout=attention_dropout, space_dim=space_dim)
        self.mlp = MLP(hidden_dim, hidden_dim, hidden_dim, n_layers=2)
        self.act = nn.GELU()
        self.register_parameter("mu", nn.Parameter(torch.zeros(hidden_dim)))

    def forward(self, x, fx, freq_x=None, freq_y=None):
        # x = self.Attn1(x, y=None, freq_x=freq_x, freq_y=freq_y)
        # x = orthogonal_qr(x) if self.ortho else x
        # # fx = (x_ * torch.nn.functional.softplus(self.mu)) @ (x_.transpose(-2, -1) @ fx)
        # x = self.Attn2(x, y=None, freq_x=freq_x, freq_y=freq_y)

        fx = self.Attn1(fx, y=x, freq_x=freq_x, freq_y=freq_y)
        x = orthogonal_qr(x) if self.ortho else x
        # fx = (x_ * torch.nn.functional.softplus(self.mu)) @ (x_.transpose(-2, -1) @ fx)
        fx = self.Attn2(fx, y=None, freq_x=freq_x, freq_y=freq_y)
        x = self.mlp(x) + x
        return x, fx


class ONO2(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 ffn_dropout=0.0,
                 attn_dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 ortho=False,
                 act='gelu',
                 res=256,
                 ):
        super(ONO2, self).__init__()

        self.space_dim = space_dim
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden

        self.preprocess_x = FeedForward(dim=space_dim, hidden_dim=n_hidden, out_dim=n_hidden // 2)
        self.preprocess_f = FeedForward(dim=1, hidden_dim=n_hidden, out_dim=n_hidden // 2)
        # self.preprocess_f2 = FeedForward(dim=1+space_dim, hidden_dim=n_hidden, out_dim=n_hidden )
        self.preprocess_x2 = MLP(1 + space_dim, n_hidden, n_hidden, n_layers=2,act=act)
        self.preprocess_f2 = MLP(1, n_hidden, n_hidden, n_layers=2, act=act)

        self.decode_f = MLP(n_hidden, n_hidden, 1, n_layers=2, act=act)
        self.blocks = nn.ModuleList([ONOBlock(num_heads=n_head, hidden_dim=n_hidden, dropout=ffn_dropout, attention_dropout=attn_dropout,ortho=ortho, act=act, space_dim=space_dim) for _ in range(n_layers)])
        # self.blocks = nn.ModuleList([CrossAttentionBlock(n_hidden, n_head) for _ in range(n_layers)])
        # self.blocks[-1].act = nn.Identity()

        self.emb_module_f = RotaryEmbedding(n_hidden // space_dim, min_freq=1.0 / res)

        self.__name__ = 'ONO'

        # self.initialize_weights()

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

    def forward(self, x, fx, T=None):

        # 1d情况下分开升维再concate比concate后升维效果更好,1d情况升维不需要mlp只需要单层linear便可达到需要的效果
        if self.space_dim == 1:
            x = self.preprocess_x(x)
            Input_f = self.preprocess_f(fx)
            x = torch.cat((x, Input_f), -1)

        elif self.space_dim == 2:
            freq_x = self.emb_module_f.forward(x[..., 0], x.device)
            freq_y = self.emb_module_f.forward(x[..., 1], x.device)

            # x = self.preprocess_x(x)
            # input_f = self.preprocess_f(fx)
            # x = torch.cat((x, input_f), -1)

            x = torch.cat((x, fx), -1)
            x = self.preprocess_x2(x)
            # fx = self.preprocess_f2(fx)
            fx = x
            # x = fx
            # fx = MultipleTensors([x])

        if self.Time_Input == False:
            for i, block  in enumerate(self.blocks):
                if self.space_dim == 2:
                    x, fx = block(x, fx, freq_x=None, freq_y=None)
                    # x = self.blocks[i](x, fx)

                elif self.space_dim == 1:
                    x, fx = block(x, fx)

            fx = self.decode_f(fx)

        else:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.tim_fc(Time_emb)
            for block in self.blocks:
                x, fx = block(x, fx)

        return fx

