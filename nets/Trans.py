# --------------------------------------------------------
# Swin Transformer + ACmix
# Copyright (c) 2021 Xuran Pan
# Written by Xuran Pan
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import matplotlib.pyplot as plt
from .Dwt import DWTForward
import math
import numpy as np
from torchvision.utils import save_image


class GELU(nn.Module):

    def forward(self, x):
        return F.gelu(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D

        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        return x

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, in_channel = 3, out_channel = 64, depth = 1, head = 4, drop_path = 0.2):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.query = MultiHeadDense(out_channel, bias=False)
        self.key = MultiHeadDense(out_channel, bias=False)
        self.value = MultiHeadDense(out_channel, bias=False)
        self.qkv = nn.Linear(out_channel, out_channel * 3, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute2D(out_channel)
        self.Conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel*3, out_channels=out_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),)
        self.depth = depth
        self.norm1 = nn.LayerNorm(out_channel)
        self.norm2 = nn.LayerNorm(out_channel)
        self.drop_path = DropPath(drop_path)
        self.mlp = Mlp(in_features=out_channel, hidden_features=out_channel*2, act_layer=GELU)

    def forward(self, x):
        B, C, _, H, W = x.shape
        HL = torch.chunk(x, dim=2, chunks=3)
        lh = HL[0].view(B, C, H, W)
        hh = HL[1].view(B, C, H, W)
        hl = HL[2].view(B, C, H, W)
        # print("...",lh.shape)
        contact = torch.cat((lh, hh, hl), 1)
        #print("...", contact.shape)
        x = self.Conv0(contact)
        #print("...", x.shape)
        b, c, h, w = x.shape
        input = x
        pe = self.pe(input)
        input = input + pe

        for i in range(0, self.depth):

            input = input.reshape(b, c, h * w).permute(0, 2, 1)  #[b, h*w, c
            #print("...:", Q.shape, A.shape)
            qkv = self.norm1(input)
            Q = self.query(qkv)
            K = self.key(qkv)
            A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))  #[b, h*w, h*w]
            V = self.value(qkv)
            #print("...:", A.shape, V.shape)
            attn = torch.bmm(A, V)

            #FFN
            attn = input + self.drop_path(attn)
            input = attn + self.drop_path(self.mlp(self.norm2(attn)))
            input = input.permute(0, 2, 1).reshape(b, c, h, w)
            x = input
        #print("sucessful")
        return x

class CALayer(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CABlock(nn.Module):
    """ Residual attention block
    """

    def __init__(self, in_size=3, out_size=64):
        super(CABlock, self).__init__()

        self.Conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.CA = CALayer(out_size)

        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_size*3, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=out_size, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_size * 3, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


    def forward(self, x):
        #print("...", x.shape)

        x1 = self.Conv0(x)
        x2 = self.Conv1(x1)
        x3 = self.CA(x2)
        contact = torch.cat((x1, x2, x3), 1)
        out = self.Conv2(contact)
        feature_map = self.Conv3(contact)
        return out, feature_map

#Dual-domain Fusion Block(DFB)
class DFB(nn.Module):
    def __init__(self, in_channel = 64, bias = False):
        super(DFB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel*3, in_channel, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, in_channel, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.Sigmoid(),
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 8, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(in_channel // 8),
            nn.Conv2d(in_channel // 8, in_channel, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.Up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.conv_finnay = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=in_channel//4, out_channels = 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),)

    def forward(self, h_img, p_img, middle_img = None):
        if middle_img == None:
            middle_img = h_img

        img = torch.cat((h_img, p_img, middle_img), 1)
        x1 = self.conv1(img)
        out = self.conv_finnay(x1)

        x2 = self.conv2(out)
        x3 = self.conv3(x1)
        x4 = x3 * x2 + x1
        out_img = self.Up_sample(x4)

        return out, out_img

class Pixel_Restruction(nn.Module):
    """ Pixel_Restruction Module
    """

    def __init__(self, out_size=64):
        super(Pixel_Restruction, self).__init__()

        self.DFB1 = DFB(in_channel=64)
        self.DFB2 = DFB(in_channel=64)
        self.DFB3 = DFB(in_channel=64)
        self.DFB4 = DFB(in_channel=64)

        self.Conv256 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.Conv128 = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.Conv64 = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.Conv32 = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.Conv16 = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.conv_finnay = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=out_size//4, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, D_128, D_64, D_32, D_16):
        '''Pixel Domain Module(PDM)'''
        P_256 = self.Conv256(x)
        P_128 = self.Conv128(P_256)
        P_64 = self.Conv64(P_128)
        P_32 = self.Conv32(P_64)
        P_16 = self.Conv16(P_32)

        '''Reconstruction Module(REM)'''
        Out_16, img_32 = self.DFB1(D_16, P_16)
        Out_32, img_64 = self.DFB2(D_32, P_32, img_32)
        Out_64, img_128 = self.DFB3(D_64, P_64, img_64)
        Out_128, img_256 = self.DFB4(D_128, P_128, img_128)


        img_256 = img_256 + P_256
        out_256 = self.conv_finnay(img_256)
        return out_256, Out_128, Out_64, Out_32, Out_16


class WPFNet(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()

        '''......Discrete wavelet transform......'''
        self.dwt1 = DWTForward(J=1, wave='db1', mode='zero')
        self.dwt2 = DWTForward(J=1, wave='db1', mode='zero')
        self.dwt3 = DWTForward(J=1, wave='db1', mode='zero')
        self.dwt4 = DWTForward(J=1, wave='db1', mode='zero')


        # T_Block
        self.MHSA1 = MultiHeadSelfAttention(in_channel = 3, out_channel = embed_dim, depth=1)
        self.MHSA2 = MultiHeadSelfAttention(in_channel = 3, out_channel = embed_dim, depth=2)
        self.MHSA3 = MultiHeadSelfAttention(in_channel = 3, out_channel = embed_dim, depth=4)
        self.MHSA4 = MultiHeadSelfAttention(in_channel = 3, out_channel = embed_dim, depth=8)


        self.conv128 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.conv64 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv32 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        '''......Global average pooling......'''
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        '''......High-Frequency......'''
        self.L_128 = CABlock(3, embed_dim)
        self.L_64 = CABlock(3, embed_dim)
        self.L_32 = CABlock(3, embed_dim)
        self.L_16 = CABlock(3, embed_dim)

        '''......Pixel-Reconstruction block......'''
        self.pix_rec = Pixel_Restruction()

        self.l1 = torch.nn.SmoothL1Loss().cuda()  # similarity loss (l1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        ''''....First DWT....'''
        ll_128, lh_128 = self.dwt1(x)
        # Low_frequency
        ll_128, l_128 = self.L_128(ll_128)  #8,3,128,128

        # High_frequency
        h_128 = self.MHSA1(lh_128[0])

        ''''....Second DWT....'''
        ll_64, lh_64 = self.dwt2(ll_128)

        # Low_frequency
        ll_64, l_64 = self.L_64(ll_64)  # 8,3,128,128
        # High_frequency
        h_64 = self.MHSA2(lh_64[0])

        ''''....Third DWT....'''
        ll_32, lh_32 = self.dwt3(ll_64)

        # Low_frequency
        ll_32, l_32 = self.L_32(ll_32)  # 8,3,128,128
        # High_frequency
        h_32 = self.MHSA3(lh_32[0])

        ''''....Fourth DWT....'''
        ll_16, lh_16 = self.dwt4(ll_32)

        # Low_frequency
        ll_16, l_16 = self.L_16(ll_16)  # 8,3,128,128
        # High_frequency
        h_16 = self.MHSA4(lh_16[0])

        D_128 = l_128 + h_128
        D_64 = l_64 + h_64
        D_32 = l_32 + h_32
        D_16 = l_16 + h_16

        out_256, out_128, out_64, out_32, out_16 = self.pix_rec(x, D_128, D_64, D_32, D_16)
        out = [out_256, out_128, out_64, out_32, out_16]
        ll = [ll_128, ll_64, ll_32, ll_16]
        return out, ll

    def forward(self, x):
        x = self.forward_features(x)
        return x
