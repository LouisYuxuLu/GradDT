# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from einops import rearrange

import numbers


class DSPNet(nn.Module):
	def __init__(self):
		super(DSPNet,self).__init__()

		self.en = En_Decoder(4,16)
         
	def forward(self,x,sx,sy,lap):
        
		xsout,xyout,lapout,x_out = self.en(x,sx,sy,lap)
      
		return xsout,xyout,lapout,x_out

class MTRB(nn.Module):# Edge-oriented Residual Convolution Block
	def __init__(self,channel,norm=False):                                
		super(MTRB,self).__init__()

		self.conv_1_1 = nn.Conv2d(channel*1, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_1_2 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_1_3 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_1_4 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_2_1 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2_2 = nn.Conv2d(channel*3, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2_3 = nn.Conv2d(channel*3, channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_3_1 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_3_2 = nn.Conv2d(channel*3, channel,kernel_size=3,stride=1,padding=1,bias=False)
 
		self.conv_4_1 = nn.Conv2d(channel*1, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_2 = nn.Conv2d(channel*3, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_3 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)

		self.conv_5_1 = nn.Conv2d(channel*1, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_5_2 = nn.Conv2d(channel*3, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_5_3 = nn.Conv2d(channel*3, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_5_4 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)


		self.conv_1_sobelx = nn.Conv2d(1, channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_1_sobely = nn.Conv2d(1, channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_1_lap    = nn.Conv2d(1, channel,kernel_size=1,stride=1,padding=0,bias=False)        


		self.conv_sobelx_in = nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_sobely_in = nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_lap_in    = nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0,bias=False)  
		self.conv_x_in      = nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0,bias=False)


		self.conv_sobelx_out = nn.Conv2d(channel, 1,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_sobely_out = nn.Conv2d(channel, 1,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_lap_out    = nn.Conv2d(channel, 1,kernel_size=1,stride=1,padding=0,bias=False)  
		self.conv_x_out      = nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0,bias=False)
        
        
		self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)
		self.sig= nn.Sigmoid()

		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
		self.norm =nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')


	def forward(self,x,sobelx,sobely,laplacian):    

		sobelx_in = self.act(self.norm(self.conv_sobelx_in(self.maxpool(self.maxpool(self.conv_1_sobelx(sobelx))))))
		sobely_in = self.act(self.norm(self.conv_sobely_in(self.maxpool(self.maxpool(self.conv_1_sobely(sobely))))))
		lap_in    = self.act(self.norm(self.conv_x_in(self.maxpool(self.maxpool(self.conv_1_lap(laplacian))))))
		x_in      =  self.act(self.norm(self.conv_x_in(x)))        
        
        
		x_1_1 = self.act(self.norm(self.conv_1_1(sobelx_in)))
		x_1_2 = self.act(self.norm(self.conv_1_2(torch.cat((sobely_in,x_1_1),1))))
		x_1_3 = self.act(self.norm(self.conv_1_3(torch.cat((lap_in,x_1_2),1))))
		x_1_4 = self.act(self.norm(self.conv_1_4(torch.cat((x_in,x_1_3),1))))
        
        
		x_2_1 = self.act(self.norm(self.conv_2_1(torch.cat((x_1_1 , x_1_2),1))))
		x_2_2 = self.act(self.norm(self.conv_2_2(torch.cat((x_1_2 , x_1_3 , x_2_1),1))))
		x_2_3 = self.act(self.norm(self.conv_2_3(torch.cat((x_1_3 , x_1_4 , x_2_2),1))))
        

		x_3_1 = self.act(self.norm(self.conv_3_1(torch.cat((x_2_1 , x_2_2),1))))
		x_3_2 = self.act(self.norm(self.conv_3_2(torch.cat((x_2_2 , x_2_3 , x_3_1),1))))
   
     
		x_4_1 = self.act(self.norm(self.conv_4_1(x_3_1)))
		x_4_2 = self.act(self.norm(self.conv_4_2(torch.cat((x_3_1 , x_3_2 , x_4_1),1))))
		x_4_3 = self.act(self.norm(self.conv_4_3(torch.cat((x_3_2 , x_4_2),1))))
        
        
		x_5_1 = self.act(self.norm(self.conv_5_1(x_4_1)))
		x_5_2 = self.act(self.norm(self.conv_5_2(torch.cat((x_4_1 , x_4_2 , x_5_1),1))))
		x_5_3 = self.act(self.norm(self.conv_5_3(torch.cat((x_4_2 , x_4_3 , x_5_2),1))))
		x_5_4 = self.act(self.norm(self.conv_5_4(torch.cat((x_4_3 , x_5_3),1))))
        
		sobelx_out = self.conv_sobelx_out(x_5_1)
		sobely_out = self.conv_sobely_out(x_5_2)
		lap_out    = self.conv_lap_out(x_5_3)
		x_out      = x_5_1+x_5_2+x_5_3+x_5_4


		return	sobelx_out , sobely_out , lap_out  , x_out
  
class MTRB01(nn.Module):# Edge-oriented Residual Convolution Block
	def __init__(self,channel,norm=False):                                
		super(MTRB01,self).__init__()

		self.conv_1_1 = nn.Conv2d(channel*1, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_1_2 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_1_3 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_1_4 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_2_1 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2_2 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2_3 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_3_1 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_3_2 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
 
		self.conv_4_1 = nn.Conv2d(channel*1, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_2 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_3 = nn.Conv2d(channel*1, channel,kernel_size=3,stride=1,padding=1,bias=False)

		self.conv_5_1 = nn.Conv2d(channel*1, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_5_2 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_5_3 = nn.Conv2d(channel*2, channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_5_4 = nn.Conv2d(channel*1, channel,kernel_size=3,stride=1,padding=1,bias=False)


		self.conv_1_sobelx = nn.Conv2d(1, channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_1_sobely = nn.Conv2d(1, channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_1_lap    = nn.Conv2d(1, channel,kernel_size=1,stride=1,padding=0,bias=False)        


		self.conv_sobelx_in = nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_sobely_in = nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_lap_in    = nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0,bias=False)  
		self.conv_x_in      = nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0,bias=False)


		self.conv_sobelx_out = nn.Conv2d(channel, 1,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_sobely_out = nn.Conv2d(channel, 1,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_lap_out    = nn.Conv2d(channel, 1,kernel_size=1,stride=1,padding=0,bias=False)  
		self.conv_x_out      = nn.Conv2d(channel, channel,kernel_size=1,stride=1,padding=0,bias=False)
        
        
		self.conv_out = nn.Conv2d(4*channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)
		self.sig= nn.Sigmoid()

		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
		self.norm =nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')


	def forward(self,x,sobelx,sobely,laplacian):    

		sobelx_in = self.act(self.norm(self.conv_sobelx_in(self.maxpool(self.maxpool(self.conv_1_sobelx(sobelx))))))
		sobely_in = self.act(self.norm(self.conv_sobely_in(self.maxpool(self.maxpool(self.conv_1_sobely(sobely))))))
		lap_in    = self.act(self.norm(self.conv_lap_in(self.maxpool(self.maxpool(self.conv_1_lap(laplacian))))))
		x_in      =  self.act(self.norm(self.conv_x_in(x)))        
        
        
		x_1_1 = self.act(self.norm(self.conv_1_1(sobelx_in)))
		x_1_2 = self.act(self.norm(self.conv_1_2(torch.cat((sobely_in,x_1_1),1))))
		x_1_3 = self.act(self.norm(self.conv_1_3(torch.cat((lap_in,x_1_2),1))))
		x_1_4 = self.act(self.norm(self.conv_1_4(torch.cat((x_in,x_1_3),1))))
        
        
		x_2_1 = self.act(self.norm(self.conv_2_1(torch.cat((x_1_1 , x_1_2),1))))
		x_2_2 = self.act(self.norm(self.conv_2_2(torch.cat((x_1_2 , x_1_3),1))))
		x_2_3 = self.act(self.norm(self.conv_2_3(torch.cat((x_1_3 , x_1_4),1))))
        

		x_3_1 = self.act(self.norm(self.conv_3_1(torch.cat((x_2_1 , x_2_2),1))))
		x_3_2 = self.act(self.norm(self.conv_3_2(torch.cat((x_2_2 , x_2_3),1))))
   
     
		x_4_1 = self.act(self.norm(self.conv_4_1(x_3_1)))
		x_4_2 = self.act(self.norm(self.conv_4_2(torch.cat((x_3_1 , x_3_2),1))))
		x_4_3 = self.act(self.norm(self.conv_4_3(x_3_2)))
        
        
		x_5_1 = self.act(self.norm(self.conv_5_1(x_4_1)))
		x_5_2 = self.act(self.norm(self.conv_5_2(torch.cat((x_4_1 , x_4_2),1))))
		x_5_3 = self.act(self.norm(self.conv_5_3(torch.cat((x_4_2 , x_4_3),1))))
		x_5_4 = self.act(self.norm(self.conv_5_4(x_4_3)))
        
		sobelx_out = self.conv_sobelx_out(x_5_1)
		sobely_out = self.conv_sobely_out(x_5_2)
		lap_out    = self.conv_lap_out(x_5_3)
		x_out      = self.act(self.norm(self.conv_out(torch.cat((x_5_1,x_5_2,x_5_3,x_5_4),1)))+ x)


		return	sobelx_out+self.maxpool(self.maxpool(sobelx)) , sobely_out+self.maxpool(self.maxpool(sobely)) , lap_out+self.maxpool(self.maxpool(laplacian)) , x_out      
        

class En_Decoder(nn.Module):
	def __init__(self,inchannel,channel):
		super(En_Decoder,self).__init__()
        


		self.el = TransformerBlock(channel, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
		self.em = TransformerBlock(channel*2, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
		self.es = nn.Sequential(TransformerBlock(channel*4, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),TransformerBlock(channel*4, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'))
		self.ds = nn.Sequential(TransformerBlock(channel*4, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),TransformerBlock(channel*4, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'))
		self.dm = TransformerBlock(channel*2, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
		self.dl = TransformerBlock(channel, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
        
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)   

		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)  

        
		self.conv_in = nn.Conv2d(inchannel,channel,kernel_size=1,stride=1,padding=0,bias=False)        
		self.conv_out = nn.Conv2d(channel,1,kernel_size=1,stride=1,padding=0,bias=False)    
    		

		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.mtrb = MTRB01(4*channel)


	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')

	def forward(self,x,sx,sy,lap):
        
		x_elin = self.conv_in(torch.cat((x,sx,sy,lap),1))

		elout = self.el(x_elin)
        
		x_emin = self.conv_eltem(self.maxpool(elout))
        
		emout = self.em(x_emin)
        
		x_esin = self.conv_emtes(self.maxpool(emout))        
        
		esout = self.es(x_esin)
        
		xsout,xyout,lapout,mtrbout = self.mtrb(esout,sx,sy,lap)
        
		dsout = self.ds(mtrbout)
        
		x_dmin = self._upsample(self.conv_dstdm(dsout),emout) + emout
        
		dmout = self.dm(x_dmin)

		x_dlin = self._upsample(self.conv_dmtdl(dmout),elout) + elout
        
		dlout = self.dl(x_dlin)
        
		x_out = self.conv_out(dlout)+x

        
		return xsout,xyout,lapout,x_out 
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        #print(x.shape,mu.shape)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
     
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
        