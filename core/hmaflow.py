import os
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from update import BasicUpdateBlock
from extractor import BasicEncoder
from corr import CorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

from csa.self_attention import CorrSelfAttention


class HMAFlow(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = [10,8,6,4] 

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=192)
            
        self.align_corr = nn.Sequential(nn.Conv2d(980, 980, 2, 2, padding=0, groups=980),
                                        nn.ReLU(inplace=True))
                    
        self.corr_rd = nn.Sequential(nn.Conv2d(1960, 324, kernel_size=1, stride=1),
                                        nn.ReLU(inplace=True))
            
        self.cnet_rd = nn.Conv2d(384*2, 384, kernel_size=1, stride=1)

        self.atten = CorrSelfAttention(embed_dim=324, depth=1, num_heads=1, mlp_ratio=2., 
                                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
       
                                            
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        
        # [b,2, 110,256]
        coords10 = coords_grid(N, H//4, W//4, device=img.device)
        coords11 = coords_grid(N, H//4, W//4, device=img.device)
        
        # [b,2, 55,128]
        coords20 = coords_grid(N, H//8, W//8, device=img.device)
        coords21 = coords_grid(N, H//8, W//8, device=img.device)

        return coords10, coords11, coords20, coords21


    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination 
            flow:[b,2, 55,128], mask:[b, 64*9, 55,128]
        """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def up_hierarchical_flow2(self, flow, mask):
        """  
            flow:[b,2, 55,128], mask:[b, 4*9, 55,128]
        """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 2*H, 2*W)


    def forward(self, images1, images2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        images1 = 2 * (images1 / 255.0) - 1.0
        images2 = 2 * (images2 / 255.0) - 1.0

        images1 = images1.contiguous()
        images2 = images2.contiguous()
    
        # feature network
        fmap_groups = self.fnet([images1, images2])  

        corr_fn = CorrBlock(fmap_groups, radius=self.args.corr_radius)

        # context network
        cnet = self.cnet(images1, context=True)  # [b,384*2,55,128]           
        cnet = self.cnet_rd(cnet)      
        net, inp = torch.split(cnet, [192, 192], dim=1)  
        net = torch.tanh(net)
        inp = torch.relu(inp)


        coords10, coords11, coords20, coords21 = self.initialize_flow(images1)

        if flow_init is not None:
            coords21 = coords21 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords11, coords21 = [coords11.detach(), coords21.detach()]
            
            corr1, corr2 = corr_fn([coords11, coords21])    

            # motion field alignment
            corr1 = self.align_corr(corr1).float()
            corr = torch.cat([corr1, corr2], dim=1) 
            corr = self.corr_rd(corr).float()  

            # csa module
            corr = self.atten.get_corr_selfattention(corr, n=1)
            corr = torch.relu(corr).float()
            
            # compute flow
            flow = coords21 - coords20

            # refine optical flow 
            net, up_mask2, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords21 = coords21 + delta_flow
            
            coords11 = coords10 + self.up_hierarchical_flow2(coords21 - coords20, up_mask2)

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords21 - coords20)
            else:    
                flow_up = self.upsample_flow(coords21 - coords20, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords21 - coords20, flow_up

        return flow_predictions

