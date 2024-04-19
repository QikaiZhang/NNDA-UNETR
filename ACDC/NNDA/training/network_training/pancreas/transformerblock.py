import torch.nn as nn
import torch
# from unetr_pp.network_architecture.dynunet_block import UnetResBlock
from nnformer.training.network_training.synapse.dynunet_block import UnetResBlock
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import torch
from torch import nn, einsum
# from synapse.model_components import PreNormResidual,Attention,FeedForward,Attentionlayer
from typing import Type, Callable, Tuple, Optional, Set, List
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            window_size=(7, 7, 7)
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        w1, w2, w3 = window_size
        # 初始化相对位置索引矩阵[2*H-1,2*W-1,2*D-1,num_heads]
        self.rel_pos_bias = nn.Embedding((2 * w1 - 1) * (2 * w2 - 1) * (2 * w3 - 1), self.heads)
        pos1 = torch.arange(w1)
        pos2 = torch.arange(w2)
        pos3 = torch.arange(w3)
        # 首先我们利用torch.arange和torch.meshgrid函数生成对应的坐标，[3,H,W,D] 然后堆叠起来，展开为一个二维向量，得到的是绝对位置索引。
        grid = torch.stack(torch.meshgrid(pos1, pos2, pos3, indexing='ij'))
        grid = rearrange(grid, 'c i j k -> (i j k) c')
        # 广播机制，分别在第一维，第二维，插入一个维度，进行广播相减，得到 3, whd*ww, whd*ww的张量
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos[..., 0] += w1 - 1
        rel_pos[..., 1] += w2 - 1
        rel_pos[..., 2] += w3 - 1
        # 做了个乘法操作，以进行区分,最后一维上进行求和，展开成一个一维坐标   a*x1 + b*x2 + c*x3  (a= hd b=d c =1)
        rel_pos_indices = (rel_pos * torch.tensor([(2 * w2 - 1) * (2 * w3 - 1), (2 * w3 - 1), 1])).sum(dim=-1)

        # 注册为一个不参与网络学习的变量
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, depth, window_height, window_width, window_depth, _, device, h = *x.shape, x.device, self.heads
        # flatten
        x = rearrange(x, 'b x y z w1 w2 w3 d -> (b x y z) (w1 w2 w3) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h), (q, k, v))
        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2 w3) d -> b w1 w2 w3 (h d)', w1=window_height, w2=window_width, w3=window_depth)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y z) ... -> b x y z ...', x=height, y=width, z=depth)

def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation

class MBConv3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm3d,
            drop_path: float = 0.
    ) -> None:
        super(MBConv3D, self).__init__()
        self.drop_path_rate: float = drop_path
        if not downscale:
            assert in_channels == out_channels, "If downscaling is utilized input and output channels must be equal."
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1, 1)),

            # Replace DepthwiseSeparableConv with 3D Convolutional layers
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1),
            norm_layer(out_channels),
            act_layer(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1, 1))
        )
        self.skip_path = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # Pooling operation for downsampling
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1))
        ) if downscale else nn.Identity()

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Clp_2(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 dim = None,
                 head = None,
                 windowsize = None, 
                 act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.head = head
        self.window_size = windowsize

        self.in_features = in_features
        self.out_features = in_features
        self.hidden_features = hidden_features
        self.hidc = hidden_features//4
        self.fc1 = nn.Conv3d(self.in_features, self.hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(self.hidden_features, self.out_features, 1)
        self.dw1 = nn.Conv3d(self.hidc, self.hidc, 1, groups=self.hidc)
        self.dw3 = nn.Conv3d(self.hidc, self.hidc, 3, padding=1, groups=self.hidc)
        self.dw5 = nn.Conv3d(self.hidc, self.hidc, 5, padding=2, groups=self.hidc)
        self.dw7 = nn.Conv3d(self.hidc, self.hidc, 7, padding=3, groups=self.hidc)
        self.conv = nn.Conv3d(self.hidden_features, self.hidden_features, 1)


    def forward(self, x):
        # x = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)',w1=self.window_size[0], w2=self.window_size[1],
        #                w3=self.window_size[2])(x)
        x = self.fc1(x)
        x = self.act(x)
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        x1 = self.dw1(x1)
        x2 = self.dw3(x2)
        x3 = self.dw5(x3)
        x4 = self.dw7(x4)
        y = x+self.conv(torch.cat([x1,x2,x3,x4], dim=1))
        y = self.fc2(y)
        # y = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d',w1=self.window_size[0], w2=self.window_size[1],
        #                w3=self.window_size[2])(y)
        return y
    
class Clp_1(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 dim = None,
                 head = None,
                 windowsize = None, 
                 act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.head = head
        self.window_size = windowsize

        self.in_features = in_features
        self.out_features = in_features
        self.hidden_features = hidden_features
        self.hidc = hidden_features//4
        self.fc1 = nn.Conv3d(self.in_features, self.hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(self.hidden_features, self.out_features, 1)
        self.dw1 = nn.Conv3d(self.hidc, self.hidc, 1, groups=self.hidc)
        self.dw3 = nn.Conv3d(self.hidc, self.hidc, 3, padding=1, groups=self.hidc)
        self.dw5 = nn.Conv3d(self.hidc, self.hidc, 5, padding=2, groups=self.hidc)
        self.dw7 = nn.Conv3d(self.hidc, self.hidc, 7, padding=3, groups=self.hidc)
        self.conv = nn.Conv3d(self.hidden_features, self.hidden_features, 1)


    def forward(self, x):
        x = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)',w1=self.window_size[0], w2=self.window_size[1],
                       w3=self.window_size[2])(x)
        x = self.fc1(x)
        x = self.act(x)
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        x1 = self.dw1(x1)
        x2 = self.dw3(x2)
        x3 = self.dw5(x3)
        x4 = self.dw7(x4)
        y = x+self.conv(torch.cat([x1,x2,x3,x4], dim=1))
        y = self.fc2(y)
        y = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d',w1=self.window_size[0], w2=self.window_size[1],
                       w3=self.window_size[2])(y)
        return y

class Conv3d_Forward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.,window_size=None,):
        super().__init__()
        # inner_dim = int(dim * mult)
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim,1),
            # nn.BatchNorm3d(dim,dim),
            nn.GELU(),
            # nn.Dropout(dropout),
            # nn.BatchNorm3d(dim,dim),
            nn.Conv3d(dim, dim,1),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        x = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)',w1=self.window_size[0], w2=self.window_size[1],
                       w3=self.window_size[2])(x)
        x = self.net(x)
        x = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d',w1=self.window_size[0], w2=self.window_size[1],
                w3=self.window_size[2])(x)
        return x

class Attentionlayer(nn.Module):
    def __init__(self, dim, head, window_size):
        super(Attentionlayer, self).__init__()
        self.dim = dim
        self.head = head
        self.window_size = window_size

    def forward(self, x):
        x1 = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
                       w3=self.window_size[2]).cuda()(x)
        x2 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0,
                                                 window_size=self.window_size).cuda()).cuda()(x1)
        x3 = PreNormResidual(self.dim, Conv3d_Forward(dim=self.dim, dropout=0.,window_size=self.window_size)).cuda()(x2)
        x4 = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)').cuda()(x3)

        x5 = Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
                       w3=self.window_size[2]).cuda()(x4)
        x6 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0,
                                                 window_size=self.window_size).cuda()).cuda()(x5)
        x7 = PreNormResidual(self.dim, Conv3d_Forward(dim=self.dim, dropout=0.,window_size=self.window_size).cuda()).cuda()(x6)
        x8 = Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)').cuda()(x7)

        return x8
        
# class Attentionlayer(nn.Module):
#     def __init__(self, dim, head, window_size):
#         super(Attentionlayer, self).__init__()
#         self.dim = dim
#         self.head = head
#         self.window_size = window_size

#     def forward(self, x):
#         x1 = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
#                        w3=self.window_size[2]).cuda()(x)
#         x2 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0,
#                                                  window_size=self.window_size).cuda()).cuda()(x1)
#         x3 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.)).cuda()(x2)
#         x4 = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)').cuda()(x3)

#         x5 = Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
#                        w3=self.window_size[2]).cuda()(x4)
#         x6 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0,
#                                                  window_size=self.window_size).cuda()).cuda()(x5)
#         x7 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.).cuda()).cuda()(x6)
#         x8 = Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)').cuda()(x7)

#         return x8


# class Global_Attentionlayer(nn.Module):
#     def __init__(self, dim, head, window_size):
#         super(Global_Attentionlayer, self).__init__()
#         self.dim = dim
#         self.head = head
#         self.window_size = window_size

#     def forward(self, x):
#         residual = x 
#         x = Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
#                       w3=self.window_size[2]).cuda()(x)
#         x = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0.1,
#                                                 window_size=self.window_size).cuda()).cuda()(x)
#         x = PreNormResidual(self.dim, Clp_1(in_features=self.dim,hidden_features=self.dim*4,dim=self.dim,head=self.head,windowsize=self.window_size).cuda()).cuda()(x)
#         x = Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)').cuda()(x)

#         return x + residual
    
class Global_Attentionlayer(nn.Module):
    def __init__(self, dim, head, window_size):
        super(Global_Attentionlayer, self).__init__()
        self.dim = dim
        self.head = head
        self.window_size = window_size

    def forward(self, x):
        # residual = x 
        x = Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
                      w3=self.window_size[2]).cuda()(x)
        x = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0.1,
                                                window_size=self.window_size).cuda()).cuda()(x)
        x = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.1).cuda()).cuda()(x)
        x = Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)').cuda()(x)

        return x 


class Local_Attentionlayer(nn.Module):
    def __init__(self, dim, head, window_size):
        super(Local_Attentionlayer, self).__init__()
        self.dim = dim
        self.head = head
        self.window_size = window_size

    def forward(self, x):
        x1 = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
                       w3=self.window_size[2]).cuda()(x)
        x2 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0.1,
                                                 window_size=self.window_size).cuda()).cuda()(x1)
        x3 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.2)).cuda()(x2)
        x4 = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)').cuda()(x3)

        return x4
    
# class Attentionlayer(nn.Module):
#     def __init__(self, dim, head, window_size):
#         super(Attentionlayer, self).__init__()
#         self.dim = dim
#         self.head = head
#         self.window_size = window_size

#     def forward(self, x):
#         x1 = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
#                        w3=self.window_size[2]).cuda().half()(x)
#         x2 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0.1,
#                                                  window_size=self.window_size).cuda().half()).cuda().half()(x1)
#         x3 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.1)).cuda().half()(x2)
#         x4 = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)').cuda().half()(x3)

#         x5 = Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
#                        w3=self.window_size[2]).cuda().half()(x4)
#         x6 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0.1,
#                                                  window_size=self.window_size).cuda().half()).cuda().half()(x5)
#         x7 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.1).cuda().half()).cuda().half()(x6)
#         x8 = Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)').cuda().half()(x7)

#         return x8

class GroupBatchnorm3d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm3d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, D, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(
        self,
        oup_channels: int,
        group_num: int = 16,
        gate_threshold: float = 0.5,
        torch_gn: bool = True,
    ):
        super().__init__()

        self.gn = (
            nn.GroupNorm(num_channels=oup_channels, num_groups=group_num)
            if torch_gn
            else GroupBatchnorm3d(c_num=oup_channels, group_num=group_num)
        )
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1, 1)
        reweights = self.sigmoid(gn_x * w_gamma)

        w1 = torch.where(reweights > self.gate_threshold, torch.ones_like(reweights), reweights)
        w2 = torch.where(reweights > self.gate_threshold, torch.zeros_like(reweights), reweights)
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):

    def __init__(
        self,
        op_channel: int,
        alpha: float = 1 / 2,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv3d(up_channel, up_channel // squeeze_ratio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv3d(low_channel, low_channel // squeeze_ratio, kernel_size=1, bias=False)
        self.GWC = nn.Conv3d(
            up_channel // squeeze_ratio,
            op_channel,
            kernel_size=group_kernel_size,
            stride=1,
            padding=group_kernel_size // 2,
            groups=group_size,
        )
        self.PWC1 = nn.Conv3d(up_channel // squeeze_ratio, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv3d(low_channel // squeeze_ratio, op_channel - low_channel // squeeze_ratio, kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(
        self,
        op_channel: int,
        group_num: int = 4,
        gate_threshold: float = 0.5,
        alpha: float = 1 / 2,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ):
        super().__init__()
        self.SRU = SRU(
            op_channel,
            group_num=group_num,
            gate_threshold=gate_threshold,
            torch_gn=False,
        )
        self.CRU = CRU(
            op_channel,
            alpha=alpha,
            squeeze_ratio=squeeze_ratio,
            group_size=group_size,
            group_kernel_size=group_kernel_size,
        )

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x
    
class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
            windowsize = (4,4,4),
            att_head_dim = 32,

    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_proj = nn.Conv3d(hidden_size,int(hidden_size)//2,1)
        self.att_proj = nn.Conv3d(hidden_size,int(hidden_size)//2,1)
        self.epa_block = EPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size,
                              num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        # self.clip2 = Clp_2(in_features=hidden_size,hidden_features=hidden_size*4,windowsize=windowsize,head=att_head_dim,dim=hidden_size)
        # self.mb_block = MBConv3D(in_channels=hidden_size,out_channels=hidden_size,downscale=False)
        # self.SCconv3d = ScConv(hidden_size)
        # self.Scconv3d_act = nn.SiLU()
        self.att_block = Attentionlayer(dim = hidden_size,window_size=windowsize,head = att_head_dim)
        # self.att_block = Global_Attentionlayer(dim = hidden_size,window_size=windowsize,head = att_head_dim)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape
        attn_skip = self.att_block(x)
        # attn_skip = self.att_linear(attn_skip)
        attn_skip = self.att_proj(attn_skip)

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))
        attn = Rearrange('B (H W D) C -> B C H W D ',B=B,C=C,H=H,W=W,D=D)(attn)
        attn = self.epa_proj(attn)
        # attn = self.clip2(attn)
        # attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        # attn_skip = self.att_block(self.mb_block(x))
        # attn_skip = self.Scconv3d_act(self.SCconv3d(x))
        # attn_skip = self.SCconv3d(x)

        attn_skip = torch.cat((attn, attn_skip), dim=1)
        # attn_skip = self.att_block(x)
        # attn_skip = self.att_block(self.SCconv3d(x))
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


class EPA(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1,):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        # self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        # self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        # self.out_proj = nn.Linear(hidden_size, hidden_size)
        # self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        # q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]
        q_shared, k_shared, v_CA = qkvv[0], qkvv[1], qkvv[2]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        # v_SA = v_SA.transpose(-2, -1)

        # k_shared_projected = self.E(k_shared)

        # v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        # x_CA = self.out_proj(x_CA)

        # attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        # attn_SA = attn_SA.softmax(dim=-1)
        # attn_SA = self.attn_drop_2(attn_SA)

        # x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        # x_SA = self.out_proj(x_SA)
        # x_CA = self.out_proj2(x_CA)
        # x = torch.cat((x_SA, x_CA), dim=-1)
        return x_CA

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
    

class EPA1(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
