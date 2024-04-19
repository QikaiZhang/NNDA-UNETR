import torch.nn as nn
import torch
from dynunet_block import UnetResBlock
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import torch
from torch import nn, einsum
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
        # grid = torch.stack(torch.meshgrid(pos1, pos2, pos3, indexing='ij'))
        grid = torch.stack(torch.meshgrid(pos1, pos2, pos3))
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


class Conv3d_Forward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.,window_size=None,):
        super().__init__()
        # inner_dim = int(dim * mult)
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim,1),
            nn.GELU(),
            nn.Conv3d(dim, dim,1),

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
        self.att_block = Attentionlayer(dim = hidden_size,window_size=windowsize,head = att_head_dim)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape
        attn_skip = self.att_block(x)
        attn_skip = self.att_proj(attn_skip)

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))
        attn = Rearrange('B (H W D) C -> B C H W D ',B=B,C=C,H=H,W=W,D=D)(attn)
        attn = self.epa_proj(attn)

        attn_skip = torch.cat((attn, attn_skip), dim=1)
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
        self.qkvv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(channel_attn_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA = qkvv[0], qkvv[1], qkvv[2]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)


        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        return x_CA

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
    