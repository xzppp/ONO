import math
import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.nn import functional as F
from functools import partial
from typing import Callable
import warnings

from timm.models.layers import trunc_normal_
# 
from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinearSelfAttention

# pip install performer-pytorch
from performer_pytorch import SelfAttention as PerformerSelfAttention

# pip install nystrom-attention
from nystrom_attention import NystromAttention

# pip install reformer_pytorch
from reformer_pytorch import LSHSelfAttention

# pip install linformer
from linformer import LinformerSelfAttention

ACTIVATION = {'gelu':nn.GELU,'tanh':nn.Tanh,'sigmoid':nn.Sigmoid,'relu':nn.ReLU,'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus,'ELU':nn.ELU,'silu':nn.SiLU}

'''
    A simple MLP class, includes at least 2 layers and n hidden layers
'''
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])
        # self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
            # x = self.act(self.bns[i](self.linears[i](x))) + x
        x = self.linear_post(x)
        return x

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

class ONOBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            attention_dropout: float,
            act='gelu',
            mlp_ratio=4,
            orth=False,
            attn_type=None,
            last_layer=False,
            momentum=0.9,
            psi_dim=64,
    ):
        super().__init__()
        self.orth = orth
        self.momentum = momentum
        if self.orth:
            self.register_buffer("feature_cov", None)

        self.ln_1 = nn.LayerNorm(hidden_dim)
        if attn_type == 'performer':
            self.Attn = PerformerSelfAttention(hidden_dim, causal = False, heads = num_heads, dropout=dropout, no_projection=True) # this is not preformer now
        elif attn_type == 'nystrom':
            self.Attn = NystromAttention(hidden_dim, heads = num_heads, dim_head =hidden_dim//num_heads, dropout=dropout)
        elif attn_type == 'reformer':
            self.Attn = LSHSelfAttention(hidden_dim, heads = num_heads, bucket_size = 85, n_hashes = 8, causal = False)
        elif attn_type == 'linformer':
            self.Attn = LinformerSelfAttention(hidden_dim, 7225, heads=num_heads, dropout=dropout)
        else:
            self.Attn = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dropout=dropout, attn_dropout=attention_dropout)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

        self.proj = MLP(hidden_dim, hidden_dim, psi_dim, n_layers=0, res=False, act=act)# if orth else nn.Identity()
        self.register_parameter("mu", nn.Parameter(torch.zeros(psi_dim)))
        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, 1) if last_layer else MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

    def forward(self, x, fx):
        x = self.Attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x

        x_ = self.proj(x)
        if self.orth:
            if self.training:
                batch_cov = torch.einsum("blc, bld->cd", x_, x_) / x_.shape[0] / x_.shape[1]
                with torch.no_grad():
                    if self.feature_cov is None:
                        self.feature_cov = batch_cov
                    else:
                        self.feature_cov.mul_(self.momentum).add_(batch_cov, alpha=1-self.momentum)
            else:
                batch_cov = self.feature_cov
            L = psd_safe_cholesky(batch_cov)
            L_inv_T = L.inverse().transpose(-2, -1)
            x_ = x_ @ L_inv_T

        fx = (x_ * torch.nn.functional.softplus(self.mu)) @ (x_.transpose(-2, -1) @ fx) # + fx
        fx = self.mlp2(self.ln_3(fx)) # + fx
        return x, fx

class ONO2(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 attn_dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 attn_type=None,
                 mlp_ratio=4,
                 orth=False,
                 psi_dim=64
        ):
        super(ONO2, self).__init__()
        self.__name__ = 'ONO'

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden

        self.preprocess = MLP(1 + space_dim, n_hidden * mlp_ratio, n_hidden * 2, n_layers=0, act=act)
        self.blocks = nn.ModuleList([ONOBlock(num_heads=n_head, hidden_dim=n_hidden, 
                                              dropout=dropout, attention_dropout=attn_dropout,
                                              act=act, attn_type=attn_type, 
                                              mlp_ratio=mlp_ratio, orth=orth,
                                              psi_dim=psi_dim,
                                              last_layer = (_ == n_layers - 1))
                                        for _ in range(n_layers)])
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

    def forward(self, x, fx, T=None):
        x = torch.cat((x, fx), -1)
        x, fx = self.preprocess(x).chunk(2, dim=-1)
        if self.Time_Input == False:
            for i, block  in enumerate(self.blocks):
                x, fx = block(x, fx)
        else:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.tim_fc(Time_emb)
            for block in self.blocks:
                x, fx = block(x, fx)
        return fx


def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
	"""Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
	Args:
		:attr:`A` (Tensor):
			The tensor to compute the Cholesky decomposition of
		:attr:`upper` (bool, optional):
			See torch.cholesky
		:attr:`out` (Tensor, optional):
			See torch.cholesky
		:attr:`jitter` (float, optional):
			The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
			as 1e-6 (float) or 1e-8 (double)
	"""
	try:
		L = torch.linalg.cholesky(A, upper=upper, out=out)
		return L
	except RuntimeError as e:
		isnan = torch.isnan(A)
		if isnan.any():
			raise NanError(
				f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
			)

		if jitter is None:
			jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
		Aprime = A.clone()
		jitter_prev = 0
		for i in range(10):
			jitter_new = jitter * (10 ** i)
			Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
			jitter_prev = jitter_new
			try:
				L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
				warnings.warn(
					f"A not p.d., added jitter of {jitter_new} to the diagonal",
					RuntimeWarning,
				)
				return L
			except RuntimeError:
				continue
		# return torch.randn_like(A).tril()
		raise e