import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from functools import partial

from .utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    # in_features=324, hidden_features=324 * 2
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)   
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  
        self.drop = nn.Dropout(drop)

    def forward(self, x): 
        x = self.fc1(x)      
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)      
        x = self.drop(x)
        
        return x 


class Attention(nn.Module):
    def __init__(self, dim=324, num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):   
        B, N, C = x.shape 

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)    
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim=324, num_heads=1, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                        drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                        norm_layer=partial(nn.LayerNorm, eps=1e-6)):             
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio) 
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False): 
        y, attn = self.attn(self.norm1(x)) 
        if return_attention:
            return attn
        
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class CorrSelfAttention(nn.Module):  
    def __init__(self, embed_dim=324, depth=1, num_heads=1, mlp_ratio=2., 
                    qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)):                            
        super().__init__()
        
        num_patches = 1*1
        
        self.corr_global_conv = nn.Conv2d(324, 324, kernel_size=1, stride=1) 
   
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.ModuleList( [ Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                                            attn_drop=attn_drop_rate, drop_path=dpr[i], 
                                            norm_layer=norm_layer) 
                                        for i in range(depth) ] )
                    
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        
        self.apply(self._init_weights)

     
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def interpolate_pos_encoding(self, x, h, w):     
        npatch = x.shape[1]                          
        N = self.pos_embed.shape[1]                 
        if npatch == N and w == h:
            return self.pos_embed
           
        patch_pos_embed = self.pos_embed   
        dim = x.shape[-1]                 
        
        h0 = h 
        w0 = w 
        h0, w0 = h0 + 0.1, w0 + 0.1  
        
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic')
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return patch_pos_embed


    def prepare_tokens(self, x):  
        h, w = (9, 9)
        x = x + self.interpolate_pos_encoding(x, h, w)

        return self.pos_drop(x)


    def get_corr_selfattention(self, corr, n=1): 
        batch, dim, h1, w1 = corr.shape
        
        corr = self.corr_global_conv(corr) 
  
        corr = corr.permute(0,2,3,1).contiguous().reshape(batch*h1*w1, 1, dim)
        corr = self.prepare_tokens(corr)    
        
        output = []
        for i, blk in enumerate(self.blocks):      
            corr = blk(corr)  
            if len(self.blocks) - i <= n:                     
                output.append(self.norm(corr))  

        output = output[0].reshape(batch, h1, w1, dim).permute(0,3,1,2).contiguous()
        
        return output  

