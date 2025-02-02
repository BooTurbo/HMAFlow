import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid
  
        
class CorrBlock:
    def __init__(self, fmap_groups, num_levels=2, radius=[10,8,6,4]):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        
        (fmap10,fmap20), (fmap11,fmap21) = fmap_groups
        fmap10 = fmap10.float() 
        fmap20 = fmap20.float() 
        fmap11 = fmap11.float() 
        fmap21 = fmap21.float() 

        corr1 = CorrBlock.corr(fmap10, fmap20)     
        corr2 = CorrBlock.corr(fmap11, fmap21)     
        
        batch, h1, w1, dim, h2, w2 = corr1.shape        
        corr1 = corr1.reshape(batch*h1*w1, dim, h2, w2)      

        batch, h1, w1, dim, h2, w2 = corr2.shape
        corr2 = corr2.reshape(batch*h1*w1, dim, h2, w2)   

        self.corr_pyramid = [corr1, corr2]


    def __call__(self, coords):
        """ coords1: [b,2, 110,256],  coords2: [b,2, 55,128] """
        r1 = self.radius[0]  
        r2 = self.radius[1]  
        r3 = self.radius[2] 
        r4 = self.radius[3] 

        out_pyramid = []
        
        for i in range(self.num_levels):
            corr_i = self.corr_pyramid[i]  

            coords_i = coords[i].permute(0, 2, 3, 1)
            batch, h, w, _ = coords_i.shape    

            # r=10 -> 21*21
            dx1 = torch.linspace(-r1, r1, 2*r1+1, device=coords_i.device)
            dy1 = torch.linspace(-r1, r1, 2*r1+1, device=coords_i.device)
            delta1 = torch.stack(torch.meshgrid(dy1, dx1), axis=-1)
            delta1 = delta1.view(1, 2*r1+1, 2*r1+1, 2)   

            # r=8 -> 17*17
            dx2 = torch.linspace(-r2, r2, 2*r2+1, device=coords_i.device)
            dy2 = torch.linspace(-r2, r2, 2*r2+1, device=coords_i.device)
            delta2 = torch.stack(torch.meshgrid(dy2, dx2), axis=-1)
            delta2 = delta2.view(1, 2*r2+1, 2*r2+1, 2)  

            # r=6 -> 13*13
            dx3 = torch.linspace(-r3, r3, 2*r3+1, device=coords_i.device)
            dy3 = torch.linspace(-r3, r3, 2*r3+1, device=coords_i.device)
            delta3 = torch.stack(torch.meshgrid(dy3, dx3), axis=-1)
            delta3 = delta3.view(1, 2*r3+1, 2*r3+1, 2)   

            # r=4 -> 9*9
            dx4 = torch.linspace(-r4, r4, 2*r4+1, device=coords_i.device)
            dy4 = torch.linspace(-r4, r4, 2*r4+1, device=coords_i.device)
            delta4 = torch.stack(torch.meshgrid(dy4, dx4), axis=-1)
            delta4 = delta4.view(1, 2*r4+1, 2*r4+1, 2)   

            # construct grid
            centroid = coords_i.reshape(batch*h*w, 1, 1, 2) 
            coords_i_delta1 = centroid + delta1      
            coords_i_delta2 = centroid + delta2     
            coords_i_delta3 = centroid + delta3
            coords_i_delta4 = centroid + delta4

            # sampling
            corr_i0 = bilinear_sampler(corr_i, coords_i_delta1)  
            corr_i1 = bilinear_sampler(corr_i, coords_i_delta2)  
            corr_i2 = bilinear_sampler(corr_i, coords_i_delta3)  
            corr_i3 = bilinear_sampler(corr_i, coords_i_delta4)  

            corr_i0 = corr_i0.view(batch, h, w, -1)      
            corr_i1 = corr_i1.view(batch, h, w, -1)      
            corr_i2 = corr_i2.view(batch, h, w, -1)
            corr_i3 = corr_i3.view(batch, h, w, -1)

            out_pyramid.append(corr_i0)
            out_pyramid.append(corr_i1)
            out_pyramid.append(corr_i2)
            out_pyramid.append(corr_i3)

        
        out1 = torch.cat(out_pyramid[:4], dim=-1)     
        out2 = torch.cat(out_pyramid[4:], dim=-1)    

        out1 = out1.permute(0, 3, 1, 2).contiguous().float()  
        out2 = out2.permute(0, 3, 1, 2).contiguous().float()  

        return out1, out2

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

