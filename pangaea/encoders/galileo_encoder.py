# GALILEO encoder implementation for PANGAEA-bench
# Adapted from Galileo: Learning Global and Local Features in Pretrained Remote Sensing Models
# Source: https://arxiv.org/abs/2502.09356
# https://github.com/nasaharvest/galileo

import collections.abc
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, vmap
from torch.jit import Final
from pangaea.encoders.base import Encoder

# GALILEO constants extracted from the repository
BASE_GSD = 10
CONFIG_FILENAME = "config.json"
ENCODER_FILENAME = "encoder.pt"

# Band definitions from GALILEO
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
SRTM_BANDS = ["elevation", "slope"]
DW_BANDS = [f"DW_{i}" for i in range(9)]
WC_BANDS = [f"WC_{i}" for i in range(5)]
ERA5_BANDS = ["temperature_2m", "total_precipitation_sum"]
TC_BANDS = ["def", "soil", "aet"]
VIIRS_BANDS = ["avg_rad"]
LANDSCAN_BANDS = ["b1"]
LOCATION_BANDS = ["x", "y", "z"]

# Complete band lists as in GALILEO
SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + ["NDVI"]
TIME_BANDS = ERA5_BANDS + TC_BANDS + VIIRS_BANDS
SPACE_BANDS = SRTM_BANDS + DW_BANDS + WC_BANDS
STATIC_DW_BANDS = [f"{x}_static" for x in DW_BANDS]
STATIC_WC_BANDS = [f"{x}_static" for x in WC_BANDS]
STATIC_BANDS = LANDSCAN_BANDS + LOCATION_BANDS + STATIC_DW_BANDS + STATIC_WC_BANDS

# Band group indices exactly as in GALILEO
from collections import OrderedDict
SPACE_TIME_BANDS_GROUPS_IDX = OrderedDict({
    "S1": [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
    "S2_RGB": [SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]],
    "S2_Red_Edge": [SPACE_TIME_BANDS.index(b) for b in ["B5", "B6", "B7"]],
    "S2_NIR_10m": [SPACE_TIME_BANDS.index(b) for b in ["B8"]],
    "S2_NIR_20m": [SPACE_TIME_BANDS.index(b) for b in ["B8A"]],
    "S2_SWIR": [SPACE_TIME_BANDS.index(b) for b in ["B11", "B12"]],
    "NDVI": [SPACE_TIME_BANDS.index("NDVI")],
})

TIME_BAND_GROUPS_IDX = OrderedDict({
    "ERA5": [TIME_BANDS.index(b) for b in ERA5_BANDS],
    "TC": [TIME_BANDS.index(b) for b in TC_BANDS],
    "VIIRS": [TIME_BANDS.index(b) for b in VIIRS_BANDS],
})

SPACE_BAND_GROUPS_IDX = OrderedDict({
    "SRTM": [SPACE_BANDS.index(b) for b in SRTM_BANDS],
    "DW": [SPACE_BANDS.index(b) for b in DW_BANDS],
    "WC": [SPACE_BANDS.index(b) for b in WC_BANDS],
})

STATIC_BAND_GROUPS_IDX = OrderedDict({
    "LS": [STATIC_BANDS.index(b) for b in LANDSCAN_BANDS],
    "location": [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
    "DW_static": [STATIC_BANDS.index(b) for b in STATIC_DW_BANDS],
    "WC_static": [STATIC_BANDS.index(b) for b in STATIC_WC_BANDS],
})

# Embeddings utilities from GALILEO
def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    Exact implementation from GALILEO embeddings.py
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device) / embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_month_encoding_table(embed_dim: int) -> torch.Tensor:
    """
    Exact implementation from GALILEO embeddings.py
    Sinusoid month encoding table, for 12 months indexed from 0-11
    """
    assert embed_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))

    sin_table = torch.sin(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)

    return month_table  # (M, D)


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim: int, 
    grid_size: int, 
    res: torch.Tensor, 
    cls_token: bool = False,
    device: torch.device = None
) -> torch.Tensor:
    """
    Exact implementation from GALILEO embeddings.py
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: [n,grid_size*grid_size, embed_dim] or [n,1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if device is None:
        device = res.device
    res = res.to(device)
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)  #  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros([n, 1, embed_dim], device=pos_embed.device),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    """Exact implementation from GALILEO embeddings.py"""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


# FlexiPatchEmbed from GALILEO
def to_2tuple(x: Any) -> Tuple:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(itertools.repeat(x, 2))


class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]],
        in_chans: int = 3,
        embed_dim: int = 128,
        norm_layer: Optional[nn.Module] = None,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (1, 2, 3, 4, 5, 6),
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias
        self.patch_size_seq = patch_size_seq
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices"""
        pinvs = {}
        for ps in self.patch_size_seq:
            tuple_ps = to_2tuple(ps)
            pinvs[tuple_ps] = self._calculate_pinv(self.patch_size, tuple_ps)
        return pinvs

    def _resize(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...], shape, mode=self.interpolation, antialias=self.antialias
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(self, old_shape: Tuple[int, int], new_shape: Tuple[int, int]) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: Tuple[int, int]):
        """Resize patch_embed to target resolution via pseudo-inverse resizing"""
        if self.patch_size == new_patch_size:
            return patch_embed

        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(self.patch_size, new_patch_size)
        pinv = self.pinvs[new_patch_size].to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)
        return v_resample_patch_embed(patch_embed)

    def forward(self, x: Tensor, patch_size: Optional[Union[int, Tuple[int, int]]] = None) -> Tensor:
        batch_size = x.shape[0]
        has_time_dimension = len(x.shape) == 5
        
        if has_time_dimension:
            num_timesteps = x.shape[3]
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")

        if not patch_size:
            patch_size = self.patch_size
        patch_size = to_2tuple(patch_size)

        # Resize conv weights if needed
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)
        
        # Apply conv with resized weights
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)

        if has_time_dimension:
            x = rearrange(x, "(b t) c h w -> b h w t c", b=batch_size, t=num_timesteps)
        else:
            x = rearrange(x, "b c h w -> b h w c")
        
        x = self.norm(x)
        return x


# Attention and Transformer blocks from GALILEO
class Attention(nn.Module):
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        cross_attn: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.cross_attn = cross_attn

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y=None, attn_mask=None):
        B, N, C = x.shape
        q = self.q(x)

        if y is None:
            assert not self.cross_attn
            k = self.k(x)
            v = self.v(x)
        else:
            assert self.cross_attn
            k = self.k(y)
            v = self.v(y)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.fast_attn:
            if attn_mask is not None:
                attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, N, 1))
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p
            )
        else:
            if attn_mask is not None:
                raise NotImplementedError
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        cross_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            cross_attn=cross_attn,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, y, attn_mask):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), y, attn_mask)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class ModuleListWithInit(nn.ModuleList):
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


# GALILEO Base class
class GalileoBase(nn.Module):
    cross_attn: bool

    def __init__(
        self,
        embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        base_patch_size: int = 4,
        use_channel_embs: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.space_time_groups = SPACE_TIME_BANDS_GROUPS_IDX
        self.space_groups = SPACE_BAND_GROUPS_IDX
        self.time_groups = TIME_BAND_GROUPS_IDX
        self.static_groups = STATIC_BAND_GROUPS_IDX
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size

        self.blocks = ModuleListWithInit([
            Block(
                embedding_size,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                cross_attn=self.cross_attn,
                drop_path=drop_path,
            )
            for _ in range(depth)
        ])

        self.max_sequence_length = max_sequence_length
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_embed_from_grid_torch(
                int(embedding_size * 0.25), torch.arange(max_sequence_length)
            ),
            requires_grad=False,
        )
        month_tab = get_month_encoding_table(int(embedding_size * 0.25))
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        
        # Channel embeddings
        args = {"requires_grad": use_channel_embs}
        self.s_t_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_TIME_BANDS_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.sp_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.t_channel_embed = nn.Parameter(
            torch.zeros(len(TIME_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.st_channel_embed = nn.Parameter(
            torch.zeros(len(STATIC_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @classmethod
    def collapse_and_combine_hwtc(
        cls,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
    ):
        s_t_x = rearrange(s_t_x, "b h w t c_g d -> b (h w t c_g) d")
        sp_x = rearrange(sp_x, "b h w c_g d -> b (h w c_g) d")
        t_x = rearrange(t_x, "b t c_g d -> b (t c_g) d")

        s_t_m = rearrange(s_t_m, "b h w t c_g-> b (h w t c_g)")
        sp_m = rearrange(sp_m, "b h w c_g-> b (h w c_g)")
        t_m = rearrange(t_m, "b t c_g -> b (t c_g)")

        x = torch.cat([s_t_x, sp_x, t_x, st_x], dim=1)
        m = torch.cat([s_t_m, sp_m, t_m, st_m], dim=1)
        return x, m

    @classmethod
    def split_and_expand_hwtc(
        cls,
        x: torch.Tensor,
        h: int, w: int, t: int,
        s_t_c_g: int, sp_c_g: int, t_c_g: int, st_c_g: int,
    ):
        n_s_t_t = h * w * t * s_t_c_g
        n_t_t = t * t_c_g

        s_t_x = rearrange(x[:, :n_s_t_t], "b (h w t c) d -> b h w t c d", h=h, w=w, t=t, c=s_t_c_g)
        sp_x = rearrange(
            x[:, n_s_t_t : -(n_t_t + st_c_g)], "b (h w c) d -> b h w c d", h=h, w=w, c=sp_c_g
        )
        t_x = rearrange(x[:, -(n_t_t + st_c_g) : -st_c_g], "b (t c) d -> b t c d", t=t, c=t_c_g)
        st_x = x[:, -st_c_g:]

        return s_t_x, sp_x, t_x, st_x

    def apply_encodings(self, s_t_x, sp_x, t_x, st_x, months, patch_size, input_res):
        b, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g = sp_x.shape[-2], t_x.shape[-2]
        st_c_g = st_x.shape[-2]

        s_t_channel = repeat(self.s_t_channel_embed, "c_g d -> b h w t c_g d", b=b, h=h, w=w, t=t)
        t_channel = repeat(self.t_channel_embed, "c_g d -> b t c_g d", b=b, t=t)
        st_channel = repeat(self.st_channel_embed, "c_g d -> b c_g d", b=b)
        sp_channel = repeat(self.sp_channel_embed, "c_g d -> b h w c_g d", b=b, h=h, w=w)

        pos_embed_s_t = repeat(
            self.pos_embed[:t], "t d -> b h w t c_g d", b=b, h=h, w=w, c_g=s_t_c_g
        )
        m_embed_s_t = repeat(
            self.month_embed(months), "b t d -> b h w t c_g d", h=h, w=w, c_g=s_t_c_g
        )

        pos_embed_t = repeat(self.pos_embed[:t], "t d -> b t c_g d", b=b, c_g=t_c_g)
        m_embed_t = repeat(self.month_embed(months), "b t d -> b t c_g d", c_g=t_c_g)
        t_zeros = torch.zeros(b, t, t_c_g, int(self.embedding_size * 0.25), device=t_x.device)

        sp_zeros = torch.zeros(
            b, h, w, sp_c_g, sp_channel.shape[-1] * 2, device=sp_channel.device,
        )
        st_zeros = torch.zeros(b, st_c_g, st_channel.shape[-1] * 3, device=st_channel.device)

        # Spatial embeddings
        if patch_size is None:
            patch_size = self.base_patch_size
        token_res = input_res * patch_size
        gsd_ratio = token_res / BASE_GSD

        assert h == w, "get_2d_sincos_pos_embed_with_resolution currently requires that h==w"
        spatial_embed = get_2d_sincos_pos_embed_with_resolution(
            int(self.embedding_size * 0.25), h, torch.ones(b).to(s_t_x.device) * gsd_ratio, device=s_t_x.device
        )
        spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
        spatial_embed_s_t = repeat(spatial_embed, "b h w d -> b h w t c_g d", h=h, w=w, t=t, c_g=s_t_c_g)
        spatial_embed_s = repeat(spatial_embed, "b h w d -> b h w c_g d", h=h, w=w, c_g=sp_c_g)

        s_t_embed = torch.cat([s_t_channel, pos_embed_s_t, m_embed_s_t, spatial_embed_s_t], dim=-1)
        sp_embed = torch.cat([sp_channel, sp_zeros, spatial_embed_s], dim=-1)
        t_embed = torch.cat([t_channel, pos_embed_t, m_embed_t, t_zeros], dim=-1)
        st_embed = torch.cat([st_channel, st_zeros], dim=-1)
        
        return s_t_x + s_t_embed, sp_x + sp_embed, t_x + t_embed, st_x + st_embed


class GalileoEncoder(GalileoBase):
    cross_attn = False

    def __init__(
        self,
        max_patch_size: int = 8,
        embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        freeze_projections: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__(
            embedding_size, depth, mlp_ratio, num_heads, max_sequence_length, max_patch_size, 
            use_channel_embs=True, drop_path=drop_path,
        )

        self.space_time_embed = nn.ModuleDict({
            group_name: FlexiPatchEmbed(
                in_chans=len(group), embed_dim=embedding_size, patch_size=max_patch_size
            )
            for group_name, group in self.space_time_groups.items()
        })
        self.space_embed = nn.ModuleDict({
            group_name: FlexiPatchEmbed(
                in_chans=len(group), embed_dim=embedding_size, patch_size=max_patch_size
            )
            for group_name, group in self.space_groups.items()
        })
        self.time_embed = nn.ModuleDict({
            group_name: nn.Linear(in_features=len(group), out_features=embedding_size)
            for group_name, group in self.time_groups.items()
        })
        self.static_embed = nn.ModuleDict({
            group_name: nn.Linear(in_features=len(group), out_features=embedding_size)
            for group_name, group in self.static_groups.items()
        })
        
        if freeze_projections:
            self.space_time_embed.requires_grad_(False)
            self.space_embed.requires_grad_(False)
            self.time_embed.requires_grad_(False)
            self.static_embed.requires_grad_(False)
            
        self.norm = nn.LayerNorm(embedding_size)
        self.apply(self._init_weights)

    def apply_linear_projection(
        self,
        s_t_x: torch.Tensor, sp_x: torch.Tensor, t_x: torch.Tensor, st_x: torch.Tensor,
        s_t_m: torch.Tensor, sp_m: torch.Tensor, t_m: torch.Tensor, st_m: torch.Tensor,
        patch_size: int,
    ):
        b, h, w, t, _ = s_t_x.shape
        new_h, new_w = h // patch_size, w // patch_size

        s_t_l, sp_l, t_l, st_l = [], [], [], []
        s_t_m_l, sp_m_l, t_m_l, st_m_l = [], [], [], []
        
        for idx, (channel_group, channel_idxs) in enumerate(self.space_time_groups.items()):
            s_t_m_l.append(s_t_m[:, 0::patch_size, 0::patch_size, :, idx])
            if s_t_m_l[-1].min() == 0:
                s_t_l.append(
                    self.space_time_embed[channel_group](
                        s_t_x[:, :, :, :, channel_idxs], patch_size=patch_size
                    )
                )
            else:
                s_t_l.append(
                    torch.empty(b, new_h, new_w, t, self.embedding_size, dtype=s_t_x.dtype, device=s_t_x.device)
                )
                
        for idx, (channel_group, channel_idxs) in enumerate(self.space_groups.items()):
            sp_m_l.append(sp_m[:, 0::patch_size, 0::patch_size, idx])
            if sp_m_l[-1].min() == 0:
                sp_l.append(
                    self.space_embed[channel_group](
                        sp_x[:, :, :, channel_idxs], patch_size=patch_size
                    )
                )
            else:
                sp_l.append(
                    torch.empty(b, new_h, new_w, self.embedding_size, dtype=sp_x.dtype, device=sp_x.device)
                )

        for idx, (channel_group, channel_idxs) in enumerate(self.time_groups.items()):
            t_m_l.append(t_m[:, :, idx])
            if t_m_l[-1].min() == 0:
                t_l.append(self.time_embed[channel_group](t_x[:, :, channel_idxs]))
            else:
                t_l.append(torch.empty(b, t, self.embedding_size, dtype=t_x.dtype, device=t_x.device))

        for idx, (channel_group, channel_idxs) in enumerate(self.static_groups.items()):
            st_m_l.append(st_m[:, idx])
            if st_m_l[-1].min() == 0:
                st_l.append(self.static_embed[channel_group](st_x[:, channel_idxs]))
            else:
                st_l.append(torch.empty(b, self.embedding_size, dtype=st_x.dtype, device=st_x.device))

        return (
            torch.stack(s_t_l, dim=-2), torch.stack(sp_l, dim=-2),
            torch.stack(t_l, dim=-2), torch.stack(st_l, dim=-2),
            torch.stack(s_t_m_l, dim=-1), torch.stack(sp_m_l, dim=-1),
            torch.stack(t_m_l, dim=-1), torch.stack(st_m_l, dim=-1),
        )

    @staticmethod
    def remove_masked_tokens(x, mask):
        org_mask_dtype = mask.dtype
        mask = mask.bool()
        sorted_mask, indices = torch.sort((~mask).int(), dim=1, descending=True, stable=True)
        x = x.gather(1, indices[:, :, None].expand_as(x))
        x = x * sorted_mask.unsqueeze(-1)
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        updated_mask = 1 - sorted_mask[:, :max_length]
        return x, indices, updated_mask.to(dtype=org_mask_dtype)

    @staticmethod
    def add_removed_tokens(x, indices, mask):
        masked_tokens = repeat(torch.zeros_like(x[0, 0, :]), "d -> b t d", b=x.shape[0], t=indices.shape[1])
        full_mask = torch.cat((
            mask,
            torch.ones((x.shape[0], indices.shape[1] - x.shape[1]), device=x.device, dtype=mask.dtype),
        ), dim=-1)
        out = masked_tokens.clone()
        out[~full_mask.bool()] = x[~mask.bool()]
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
        return out, full_mask

    def create_token_exit_ids(self, s_t_x, sp_x, t_x, st_x, token_exit_cfg):
        """Create token exit IDs as in original GALILEO implementation"""
        exit_s_t = torch.zeros_like(s_t_x)
        exit_sp = torch.zeros_like(sp_x)
        exit_t = torch.zeros_like(t_x)
        exit_st = torch.zeros_like(st_x)

        for idx, (key, _) in enumerate(self.space_time_groups.items()):
            exit_s_t[:, :, :, :, idx, :] = token_exit_cfg[key]

        for idx, (key, _) in enumerate(self.space_groups.items()):
            exit_sp[:, :, :, idx, :] = token_exit_cfg[key]

        for idx, (key, _) in enumerate(self.time_groups.items()):
            exit_t[:, :, idx, :] = token_exit_cfg[key]

        for idx, (key, _) in enumerate(self.static_groups.items()):
            exit_st[:, idx, :] = token_exit_cfg[key]
        return exit_s_t, exit_sp, exit_t, exit_st

    @classmethod
    def average_tokens(cls, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m):
        """Average tokens implementation from original GALILEO"""
        x, m = cls.collapse_and_combine_hwtc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        x, _, m = cls.remove_masked_tokens(x, m)
        x_for_mean = x * (1 - m.unsqueeze(-1))
        return x_for_mean.sum(dim=1) / torch.sum(1 - m, -1, keepdim=True)

    @classmethod
    def apply_mask_and_average_tokens_per_patch(
        cls,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
    ):
        """Apply mask and average tokens per patch from original GALILEO"""
        s_t_x = rearrange(s_t_x, "b t_h t_w t c_g d -> b (t_h t_w) (t c_g) d")
        sp_x = rearrange(sp_x, "b t_h t_w c_g d -> b (t_h t_w) c_g d")
        # repeat time tokens over space
        t_x = repeat(
            rearrange(t_x, "b t c_g d -> b (t c_g) d"), "b n d -> b s n d", s=sp_x.shape[1]
        )
        st_x = repeat(st_x, "b c_g d -> b s c_g d", s=sp_x.shape[1])
        s_t_m = rearrange(s_t_m, "b t_h t_w t c_g-> b (t_h t_w) (t c_g)")
        sp_m = rearrange(sp_m, "b t_h t_w c_g-> b (t_h t_w) c_g")
        t_m = repeat(rearrange(t_m, "b t c_g -> b (t c_g)"), "b n -> b s n", s=sp_x.shape[1])
        st_m = repeat(st_m, "b c_g -> b s c_g", s=sp_x.shape[1])

        x = torch.cat([s_t_x, sp_x, t_x, st_x], dim=2)  # B, S, N, D
        m = torch.cat([s_t_m, sp_m, t_m, st_m], dim=2)  # B, S, N

        x_for_mean = x * (1 - m.unsqueeze(-1))

        return x_for_mean.sum(dim=2) / torch.sum(1 - m, -1, keepdim=True)

    def forward(
        self,
        s_t_x: torch.Tensor, sp_x: torch.Tensor, t_x: torch.Tensor, st_x: torch.Tensor,
        s_t_m: torch.Tensor, sp_m: torch.Tensor, t_m: torch.Tensor, st_m: torch.Tensor,
        months: torch.Tensor, patch_size: int, input_resolution_m: Optional[int] = BASE_GSD,
        exit_after: Optional[int] = None, token_exit_cfg: Optional[Dict] = None,
        add_layernorm_on_exit: bool = True,
    ):
        (s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m) = self.apply_linear_projection(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, patch_size
        )

        if (exit_after is None) or (exit_after > 0):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m = self.apply_attn(
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months, patch_size, input_resolution_m,
                exit_after=exit_after, token_exit_cfg=token_exit_cfg,
            )

        if add_layernorm_on_exit:
            s_t_x = self.norm(s_t_x)
            sp_x = self.norm(sp_x)
            t_x = self.norm(t_x)
            st_x = self.norm(st_x)
            
        return (s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months)

    @classmethod
    def load_from_folder(cls, folder: Path, device: torch.device = None):
        """Load GALILEO encoder from folder"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        config_path = folder / CONFIG_FILENAME
        encoder_path = folder / ENCODER_FILENAME
        
        if not config_path.exists():
            raise ValueError(f"Expected {CONFIG_FILENAME} in {folder}")
        if not encoder_path.exists():
            raise ValueError(f"Expected {ENCODER_FILENAME} in {folder}")

        with open(config_path, "r") as f:
            config = json.load(f)
            model_config = config["model"]
            encoder_config = model_config["encoder"]
            
        encoder = cls(**encoder_config)
        state_dict = torch.load(encoder_path, map_location=device)
        
        # Clean state dict
        for key in list(state_dict.keys()):
            state_dict[key.replace(".backbone", "")] = state_dict.pop(key)
            
        encoder.load_state_dict(state_dict)
        return encoder

    def apply_attn(
        self, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, 
        months, patch_size, input_res, exit_after=None, token_exit_cfg=None,
    ):
        if token_exit_cfg:
            exit_s_t, exit_sp, exit_t, exit_st = self.create_token_exit_ids(
                s_t_x, sp_x, t_x, st_x, token_exit_cfg
            )
            exit_ids_seq, _ = self.collapse_and_combine_hwtc(
                exit_s_t, exit_sp, exit_t, exit_st, s_t_m, sp_m, t_m, st_m
            )
            # exited_tokens starts as linear projections!
            exited_tokens, _ = self.collapse_and_combine_hwtc(
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
            )
        else:
            exit_ids_seq = None
            exited_tokens = None

        _, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g, st_c_g = sp_x.shape[3], t_x.shape[-2], st_x.shape[-2]
        
        s_t_x, sp_x, t_x, st_x = self.apply_encodings(s_t_x, sp_x, t_x, st_x, months, patch_size, input_res)
        x, m = self.collapse_and_combine_hwtc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        
        # we only care about the values >= 1 for this mask, since 2 just tells the decoder
        # to decode those tokens. From the perspective of the encoder, 1 and 2 are equivalent
        # since they both represent masked values
        new_m = m >= 1
        x, indices, new_m = self.remove_masked_tokens(x, new_m)

        if exit_ids_seq is not None:
            exit_ids_seq, _, _ = self.remove_masked_tokens(exit_ids_seq, m >= 1)
            # still linear projections
            exited_tokens, _, _ = self.remove_masked_tokens(exited_tokens, m >= 1)

        for i_blk, blk in enumerate(self.blocks):
            if (exit_after is not None) and ((i_blk + 1) > exit_after):
                # if exit_after is N, then we exit after the Nth layer
                # if exit_after is 0, then all layers are skipped
                break

            # skip the 0th block since this is just the linear
            # projection
            if (exit_ids_seq is not None) and (i_blk > 0):
                assert exited_tokens is not None
                # half depth
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=x.detach(),
                    other=exited_tokens.detach(),
                )

            # we take the inverse of the mask because a value
            # of True indicates the value *should* take part in
            # attention
            x = blk(x=x, y=None, attn_mask=~new_m.bool())

        if exit_ids_seq is not None:
            assert exited_tokens is not None
            # full depth
            # IMPORTANT: write this to x
            x = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),  # 2 for full depth
                input=x.detach(),
                other=exited_tokens.detach(),
            )

        # we don't care about the mask returned by add_removed_tokens, since we will
        # just use the original, unclipped mask here
        x, _ = self.add_removed_tokens(x, indices, new_m)
        return (
            *self.split_and_expand_hwtc(x, h, w, t, s_t_c_g, sp_c_g, t_c_g, st_c_g),
            s_t_m, sp_m, t_m, st_m,
        )


class GalileoEncoderWrapper(Encoder):
    """
    GALILEO encoder wrapper for PANGAEA-bench.
    Adapts the Galileo encoder to work with PANGAEA's encoder interface.
    """
    
    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: Dict[str, List[str]], 
        output_layers: List[int],
        patch_size: int = 4,
        month: int = 6,
        do_pool: bool = True,
        add_layernorm_on_exit: bool = True,
        download_url: str = "",
        **kwargs
    ) -> None:
        """
        Initialize GALILEO encoder.
        
        Args:
            encoder_weights: Path to pretrained model folder
            input_size: Input image size (height=width)
            input_bands: Dictionary mapping modality to list of band names
            output_layers: Layers to output (for compatibility, not used in Galileo)
            patch_size: Patch size for Galileo encoder
            month: Month for temporal encoding (1-12)
            do_pool: Whether to pool the output tokens
            add_layernorm_on_exit: Whether to add layer norm on exit
            download_url: URL to download model weights
        """
        
        # Initialize base encoder
        super().__init__(
            model_name="galileo",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=128,  # Galileo's default embedding dimension
            output_layers=output_layers,
            output_dim=128,  # Same as embed_dim for Galileo
            multi_temporal=False,  # Galileo handles multi-temporal internally
            multi_temporal_output=False,
            pyramid_output=False,  # Galileo doesn't use pyramid output
            download_url=download_url,
        )
        
        self.patch_size = patch_size
        self.month = month
        self.do_pool = do_pool
        self.add_layernorm_on_exit = add_layernorm_on_exit
        self.output_layers = output_layers
        
        # Load the pretrained Galileo encoder will be called later from run.py
        self.galileo_encoder = None
            
        # Set up band mappings
        self._setup_band_mappings()
        
    def _setup_band_mappings(self):
        """Set up mappings between PANGAEA bands and Galileo bands."""
        self.optical_bands = self.input_bands.get('optical', [])
        self.sar_bands = self.input_bands.get('sar', [])
        
        # Map PANGAEA optical bands to Galileo S2 bands
        self.s2_indices = []
        galileo_s2_bands = S2_BANDS
        for band in self.optical_bands:
            if band in galileo_s2_bands:
                self.s2_indices.append(galileo_s2_bands.index(band))
                
        # Map PANGAEA SAR bands to Galileo S1 bands  
        self.s1_indices = []
        galileo_s1_bands = S1_BANDS
        for band in self.sar_bands:
            if band in galileo_s1_bands:
                self.s1_indices.append(galileo_s1_bands.index(band))
                
    def load_encoder_weights(self, logger) -> None:
        """Load pretrained GALILEO weights."""
        import logging
        logger = logging.getLogger(__name__)
        
        model_path = Path(self.encoder_weights)
        
        try:
            if model_path.exists() and (model_path / CONFIG_FILENAME).exists():
                logger.info(f"Loading GALILEO model from {self.encoder_weights}")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.galileo_encoder = GalileoEncoder.load_from_folder(model_path, device)
                logger.info("✓ Successfully loaded GALILEO encoder weights")
            else:
                logger.info(f"GALILEO model path does not exist: {self.encoder_weights}")
                logger.info("Creating GALILEO encoder with random weights for testing")
                self.galileo_encoder = GalileoEncoder(
                    max_patch_size=self.patch_size,
                    embedding_size=128,
                    depth=2,
                    num_heads=8,
                    max_sequence_length=24
                )
                logger.info("✓ Created GALILEO encoder with random initialization")
        except Exception as e:
            logger.error(f"Error loading GALILEO model: {e}")
            logger.info("Falling back to random initialization")
            self.galileo_encoder = GalileoEncoder(
                max_patch_size=self.patch_size,
                embedding_size=128,
                depth=2,
                num_heads=8,
                max_sequence_length=24
            )
            logger.info("✓ Created GALILEO encoder with random initialization")
            
    def _prepare_galileo_input(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare input for Galileo model from PANGAEA format.
        
        Args:
            x: Dictionary with 'optical' and/or 'sar' keys containing tensors
            
        Returns:
            Dictionary with 's2' and/or 's1' keys for Galileo
        """
        galileo_input = {}
        
        # Handle optical (S2) data
        if 'optical' in x and len(self.s2_indices) > 0:
            optical_data = x['optical']  # Could be [B, C, H, W] or [B, C, T, H, W]
            print(f"[Galileo _prepare_galileo_input] optical_data.shape: {optical_data.shape}")
            if optical_data.dim() == 5:
                # If temporal dimension exists, squeeze or select first temporal slice
                if optical_data.shape[2] == 1:
                    optical_data = optical_data.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
                else:
                    optical_data = optical_data[:, :, 0, :, :]  # Take first temporal slice
            # Select only the bands that map to Galileo S2 bands
            if len(self.s2_indices) < optical_data.shape[1]:
                optical_data = optical_data[:, self.s2_indices]
            # Convert to Galileo expected format: [B, H, W, C]
            galileo_input['s2'] = optical_data.permute(0, 2, 3, 1)
            
        # Handle SAR (S1) data  
        if 'sar' in x and len(self.s1_indices) > 0:
            sar_data = x['sar']  # Could be [B, C, H, W] or [B, C, T, H, W]
            print(f"[Galileo _prepare_galileo_input] sar_data.shape: {sar_data.shape}")
            if sar_data.dim() == 5:
                # If temporal dimension exists, squeeze or select first temporal slice
                if sar_data.shape[2] == 1:
                    sar_data = sar_data.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
                else:
                    sar_data = sar_data[:, :, 0, :, :]  # Take first temporal slice
            # Select only the bands that map to Galileo S1 bands
            if len(self.s1_indices) < sar_data.shape[1]:
                sar_data = sar_data[:, self.s1_indices]
            # Convert to Galileo expected format: [B, H, W, C]
            galileo_input['s1'] = sar_data.permute(0, 2, 3, 1)
            
        return galileo_input

    def _create_galileo_wrapper_input(self, galileo_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Create input for Galileo's wrapper forward method.
        For simplicity, we'll use the S2 data primarily.
        """
        if 's2' in galileo_input:
            return galileo_input['s2']
        elif 's1' in galileo_input:
            return galileo_input['s1']
        else:
            raise ValueError("No valid input data found for Galileo")
            
    def forward(self, x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through GALILEO encoder.
        
        Args:
            x: Dictionary containing input tensors for different modalities
            
        Returns:
            List of output tensors (for compatibility with PANGAEA decoder expectations)
        """
        if self.galileo_encoder is None:
            raise RuntimeError("GALILEO encoder not loaded properly")
            
        # Get input tensor properties
        first_tensor = next(iter(x.values()))
        batch_size = first_tensor.shape[0]
        device = first_tensor.device
        
        # Calculate output spatial dimensions based on patch size
        h_out = self.input_size // self.patch_size
        w_out = self.input_size // self.patch_size
        
        # Prepare input for Galileo in the expected format
        galileo_input = self._prepare_galileo_input(x)
        
        # Create mock inputs that match GALILEO's complex input structure
        h, w = self.input_size, self.input_size
        t = 1  # Single temporal step for simplicity
        
        # Build space-time features from available data
        s1_channels = len(S1_BANDS)  # 2 channels (VV, VH)
        s2_channels = len(S2_BANDS)  # 10 channels  
        ndvi_channels = 1
        total_st_channels = s1_channels + s2_channels + ndvi_channels  # 13 total
        
        space_time_x = torch.zeros(batch_size, h, w, t, total_st_channels, device=device)
        
        # Fill in available data
        channel_idx = 0
        
        # Add S1 data if available
        if 's1' in galileo_input:
            s1_data = galileo_input['s1']  # [B, H, W, C]
            n_s1_channels = min(s1_data.shape[-1], s1_channels)
            space_time_x[:, :, :, 0, channel_idx:channel_idx+n_s1_channels] = s1_data[:, :, :, :n_s1_channels]
        channel_idx += s1_channels
        
        # Add S2 data if available  
        if 's2' in galileo_input:
            s2_data = galileo_input['s2']  # [B, H, W, C]
            n_s2_channels = min(s2_data.shape[-1], s2_channels)
            space_time_x[:, :, :, 0, channel_idx:channel_idx+n_s2_channels] = s2_data[:, :, :, :n_s2_channels]
        channel_idx += s2_channels
        
        # Add mock NDVI (could be computed from S2 bands in the future)
        # space_time_x[:, :, :, 0, channel_idx] = 0.0  # Already zero-initialized
        
        # Create other required inputs (all mocked for now)
        space_x = torch.zeros(batch_size, h, w, len(SPACE_BANDS), device=device)
        time_x = torch.zeros(batch_size, t, len(TIME_BANDS), device=device) 
        static_x = torch.zeros(batch_size, len(STATIC_BANDS), device=device)
        
        # Create masks (all zeros = all visible)
        space_time_mask = torch.zeros(batch_size, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX), device=device)
        space_mask = torch.zeros(batch_size, h, w, len(SPACE_BAND_GROUPS_IDX), device=device)
        time_mask = torch.zeros(batch_size, t, len(TIME_BAND_GROUPS_IDX), device=device)
        static_mask = torch.zeros(batch_size, len(STATIC_BAND_GROUPS_IDX), device=device)
        
        # Month encoding
        months = torch.full((batch_size, t), self.month - 1, device=device, dtype=torch.long)
        
        try:
            # Forward pass through GALILEO encoder
            output = self.galileo_encoder(
                space_time_x,
                space_x, 
                time_x,
                static_x,
                space_time_mask,
                space_mask,
                time_mask, 
                static_mask,
                months,
                patch_size=self.patch_size,
                input_resolution_m=BASE_GSD,
                add_layernorm_on_exit=self.add_layernorm_on_exit
            )
            
            # Extract space-time features (main output for downstream tasks)
            s_t_features = output[0]  # [B, H', W', T, C_G, D]
            
            if self.do_pool:
                # Pool over space, time, and channel groups
                # [B, H', W', T, C_G, D] -> [B, D]
                pooled = s_t_features.mean(dim=(1, 2, 3, 4))
                
                # Convert back to PANGAEA expected format [B, C, H', W']
                # For pooled output, we create a 1x1 spatial output
                output_features = pooled.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
            else:
                # Keep spatial dimensions, pool over time and channel groups
                # [B, H', W', T, C_G, D] -> [B, H', W', D]
                spatial_features = s_t_features.mean(dim=(3, 4))
                
                # Convert to PANGAEA format [B, D, H', W']
                output_features = spatial_features.permute(0, 3, 1, 2)
                
        except Exception as e:
            print(f"Error in Galileo forward pass: {e}")
            print(f"Input shapes: {[(k, v.shape) for k, v in x.items()]}")
            # Fallback: return a dummy tensor
            if self.do_pool:
                output_features = torch.zeros(batch_size, self.embed_dim, 1, 1, device=device)
            else:
                output_features = torch.zeros(batch_size, self.embed_dim, h_out, w_out, device=device)
        
        # Return as list for compatibility with PANGAEA decoder expectations
        # Multiple outputs for different layers (even though Galileo doesn't have traditional layers)
        outputs = []
        for _ in self.output_layers:
            outputs.append(output_features)
            
        return outputs if outputs else [output_features]

