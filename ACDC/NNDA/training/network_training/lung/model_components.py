from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
# from unetr_pp.network_architecture.layers import LayerNorm
# from unetr_pp.network_architecture.synapse.transformerblock import TransformerBlock
# from unetr_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock
from nnformer.training.network_training.synapse.layers import LayerNorm
from nnformer.training.network_training.lung.transformerblock import TransformerBlock
from nnformer.training.network_training.synapse.dynunet_block import get_conv_layer, UnetResBlock
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import torch
from torch import nn, einsum
import torch as t

from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath
import torch.nn.functional as F
# from typing import Type, Callable, Tuple, Optional, Set, List, Union2
from typing import Type, Callable, Tuple, Optional, Set, List

einops, _ = optional_import("einops")


class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[32 * 48 * 48, 16 * 24 * 24, 8 * 12 * 12, 4 * 6 * 6], dims=[32, 64, 128, 256],
                 proj_size=[64, 64, 64, 32], depths=[3, 3, 3, 3], num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.0, transformer_dropout_rate=0.15,
                 windowsizes = [(8,8,8),(4,4,4),(2,2,2),(2,2,2)],att_head_dim = [16,16,16,16] , **kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    TransformerBlock(input_size=input_size[i], hidden_size=dims[i], proj_size=proj_size[i],
                                     num_heads=num_heads,
                                     dropout_rate=transformer_dropout_rate, pos_embed=True,att_head_dim=att_head_dim[i],
                                     windowsize=windowsizes[i]))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states

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

class SKFusion_3D(nn.Module):
    def __init__(self, channel, num_groups, M=4, reduction=4, L=4, G=4):
        '''

        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param M:  分支数
        :param reduction: 降维时的缩小比例
        :param L:  降维时全连接层 神经元的下界
         :param G:  组卷积
        '''
        super(SKFusion_3D, self).__init__()

        self.M = M
        self.channel = channel
        self.num_groups = num_groups
        self.ChannelInteraction = ChannelInteraction(self.num_groups)

        # 尺度不变
        self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(self.M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(
                nn.Conv3d(channel, channel, 3, 1, padding=1 + i, dilation=1 + i, groups=G, bias=False),
                nn.BatchNorm3d(channel),
                nn.ReLU(inplace=True))
            )
        self.fbap = nn.AdaptiveAvgPool3d(1)  # 三维自适应pool到指定维度    这里指定为1，实现 三维GAP
        d = max(channel // reduction, L)  # 计算向量Z 的长度d   下限为L
        self.fc1 = nn.Sequential(nn.Conv3d(in_channels=channel, out_channels=d, kernel_size=(1, 1, 1), bias=False),
                                 nn.BatchNorm3d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv3d(in_channels=d, out_channels=channel * M, kernel_size=(1, 1, 1), bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        batch_size, channel, _, _, _ = input.shape

        # split阶段
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))

        output = self.ChannelInteraction(output)  # --------分组

        # fusion阶段
        # U = output[0] + output[1]  + output[2] + output[3]# 逐元素相加生成 混合特征U
        U = torch.stack(output, dim=0)  # 逐元素相加生成 混合特征U
        U = torch.sum(U, dim=0)  # 逐元素相加生成 混合特征U

        s = self.fbap(U)
        z = self.fc1(s)  # S->Z降维
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b = a_b.reshape(batch_size, self.M, channel, 1, 1, 1)  # 调整形状，变为 两个全连接层的值
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax

        # selection阶段
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b = list(map(lambda x: t.squeeze(x, dim=1), a_b))  # 压缩第一维
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        # print(len(V))
        # print(V[0].shape)
        # V = V[0] + V[1] + V[2] + V[3] # 两个加权后的特征 逐元素相加

        # 逐元素相加
        V = torch.stack(V, dim=0)
        V = torch.sum(V, dim=0)
        return V + input


class ChannelInteraction(nn.Module):
    def __init__(self, num_groups):
        super(ChannelInteraction, self).__init__()
        self.num_groups = num_groups

    def forward(self, output):
        shapes = [x.shape for x in output]
        b, c, h, w, d = shapes[0]
        num_branches = len(output)

        group_channels = c // self.num_groups

        merged_channels = []
        for group_id in range(self.num_groups):
            start_channel = group_id * group_channels
            end_channel = (group_id + 1) * group_channels
            channels = [output[branch_id][:, start_channel:end_channel] for branch_id in range(num_branches)]
            merged_channel = torch.cat(channels, dim=1)
            merged_channels.append(merged_channel)

        # convolved_outputs = []
        # for merged_channel in merged_channels:
        #     convolved_output = nn.Conv3d(in_channels=merged_channels[0].shape[1], out_channels=c, kernel_size=1).cuda().half()(merged_channel)
        #     convolved_outputs.append(convolved_output)

        # concat_output = torch.cat(convolved_outputs, dim=1)
        # return concat_output
        return merged_channels


class UnetrPPEncoder2(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4], dims=[32, 64, 128, 256],
                 proj_size=[64, 64, 64, 32], depths=[2, 2, 2, 2], num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.0, transformer_dropout_rate=0.15, **kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(3, 3, 3), stride=(3, 3, 3),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        # self.stages = nn.ModuleList()
        # self.Channel_at21 = SKFusion_3D(channel=96, num_groups=3, M=3, reduction=4, L=4, G=3)
        # stage_blocks = []
        # self.at1 = Attentionlayer(dim=32, head=8, window_size= (6,6,6))
        # self.at2 = Attentionlayer(dim=64, head=8, window_size=(6, 6, 6))
        # self.at3 = Attentionlayer(dim=128, head=8, window_size=(4, 4, 4))
        # self.at4 = Attentionlayer(dim=256, head=8, window_size=(4, 4, 4))
        # stage_blocks.append(Attentionlayer(dim=32, head=8, window_size= (6,6,6)))
        # stage_blocks.append(Attentionlayer(dim=64, head=8, window_size=(6, 6, 6)))
        # stage_blocks.append(Attentionlayer(dim=128, head=8, window_size=(4, 4, 4)))
        # stage_blocks.append(Attentionlayer(dim=256, head=8, window_size=(4, 4, 4)))
        # self.stages.append(Attentionlayer(dim=32, head=32, window_size= (8, 8, 8)))
        # stage_blocks1,stage_blocks2,stage_blocks3 = [],[],[]
        # for i in range(2):
        #     stage_blocks1.append(Attentionlayer(dim=64, head=16, window_size=(4, 4, 4)))
        #     stage_blocks2.append(Attentionlayer(dim=128, head=16, window_size=(4, 4, 4)))
        #     stage_blocks3.append(Attentionlayer(dim=256, head=16, window_size=(3, 3, 3)))
        # for i in range(2):
        #     stage_blocks1.append(Attentionlayer(dim=64, head=16, window_size=(4, 4, 4)))
        # for i in range(3):
        #     stage_blocks2.append(Attentionlayer(dim=128, head=16, window_size=(4, 4, 4)))
        #     stage_blocks3.append(Attentionlayer(dim=256, head=16, window_size=(3, 3, 3)))
        # self.stages.append(nn.Sequential(*stage_blocks1))
        # self.stages.append(nn.Sequential(*stage_blocks2))
        # self.stages.append(nn.Sequential(*stage_blocks3))
        # self.stages.append(Attentionlayer(dim=64, head=16, window_size=(6, 6, 6)))
        # self.stages.append(Attentionlayer(dim=128, head=32, window_size=(4, 4, 4)))
        # self.stages.append(Attentionlayer(dim=256, head=64, window_size=(3, 3, 3)))
        # self.stages.append(nn.Sequential(*stage_blocks))

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads,
                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))

        # stage_blocks1, stage_blocks2, stage_blocks3, stage_blocks4 = [], [], [], []
        # for i in range(2):
        #     stage_blocks1.append(Attentionlayer(dim=32, head=16, window_size=(8, 8, 8)))
        #     # stage_blocks1.append(Attentionlayer(dim=32, head=8, window_size=(8, 8, 8)))
        # for i in range(2):
        #     stage_blocks2.append(Attentionlayer(dim=64, head=16, window_size=(4, 4, 4)))
        #     # stage_blocks3.append(Local_Attentionlayer(dim=128, head=32, window_size=(3, 3, 3)))
        #     stage_blocks3.append(Attentionlayer(dim=128, head=32, window_size=(4, 4, 4)))
        #     stage_blocks4.append(Attentionlayer(dim=256, head=64, window_size=(2, 2, 2)))
        # self.stages.append(nn.Sequential(*stage_blocks1))
        # self.stages.append(nn.Sequential(*stage_blocks2))
        # self.stages.append(nn.Sequential(*stage_blocks3))
        # self.stages.append(nn.Sequential(*stage_blocks4))

        # self.Mb_stages = nn.ModuleList()
        # MB_stage_blocks1, MB_stage_blocks2, MB_stage_blocks3, MB_stage_blocks4 = [], [], [], []
        # for i in range(1):
        #     MB_stage_blocks1.append(MBConv3D(in_channels = dims[0], out_channels=dims[0],downscale=False))
        #     MB_stage_blocks2.append(MBConv3D(in_channels=dims[1], out_channels=dims[1], downscale=False))
        #     MB_stage_blocks3.append(MBConv3D(in_channels=dims[2], out_channels=dims[2], downscale=False))
        #     MB_stage_blocks4.append(MBConv3D(in_channels=dims[3], out_channels=dims[3], downscale=False))
        # self.Mb_stages.append(nn.Sequential(*MB_stage_blocks1))
        # self.Mb_stages.append(nn.Sequential(*MB_stage_blocks2))
        # self.Mb_stages.append(nn.Sequential(*MB_stage_blocks3))
        # self.Mb_stages.append(nn.Sequential(*MB_stage_blocks4))

        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        # print(x.shape)
        # x = self.Mb_stages[0](x)
        x = self.stages[0](x)
        # print(x.shape)

        hidden_states.append(x)

        # for i in range(1, 4):
        #     x = self.downsample_layers[i](x)
        #     x = self.stages[i-1](x)
        #     if i == 3:  # Reshape the output of the last stage
        #         x = einops.rearrange(x, "b c h w d -> b (h w d) c")
        #     hidden_states.append(x)
        # return x, hidden_states

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            # x = self.Mb_stages[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class  UnetrUpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
            #---------------------#
            # att_type: str = 'global',
            att_head_dim: int = 16,
            att_windowsize: tuple  = (4,4,4),
            att_depth: int = 2,
            # att_type: str  = 'local',
            # att_head_dim:int = None,
            # att_window_size: tuple = None,


    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_decoder = conv_decoder
        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()
        # self.mb = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            # stage_blocks,stage_mb_blocks = [],[]
            # for j in range(att_depth):
            #     stage_blocks.append(Attentionlayer(dim = out_channels ,head = att_head_dim,
            #                                              window_size = att_window_size))
            # self.decoder_block.append(nn.Sequential(*stage_blocks))
            # for i in range(1):
            #     stage_mb_blocks.append(MBConv3D(in_channels = out_channels,out_channels=out_channels,downscale=False))
            # self.decoder_block.append(nn.Sequential(*stage_mb_blocks))
            # self.decoder_block.append(nn.Sequential(*stage_mb_blocks))
            # self.decoder_block.append(nn.Sequential(*stage_blocks))
            # if att_type == 'local':
            #     print('local')
            #     for j in range(att_depth):
            #         stage_blocks.append(Local_Attentionlayer(dim = out_channels ,head = att_head_dim,
            #                                                  window_size = att_window_size))
            #     self.decoder_block.append(nn.Sequential(*stage_blocks))
            # elif att_type == 'global':
            #     print('global')
            #     for j in range(att_depth):
            #         stage_blocks.append(Global_Attentionlayer(dim = out_channels ,head = att_head_dim,
            #                                                  window_size = att_window_size))
            #     self.decoder_block.append(nn.Sequential(*stage_blocks))

            stage_blocks = []
            for j in range(att_depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size=out_channels, proj_size=proj_size,
                                                     num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True,att_head_dim=att_head_dim, windowsize = att_windowsize))
            self.decoder_block.append(nn.Sequential(*stage_blocks))
            # stage_blocks = []
            # for j in range(depth):
            #     stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size=out_channels, proj_size=proj_size,
            #                                          num_heads=num_heads,
            #                                          dropout_rate=0.15, pos_embed=True))
            # self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        # print('transp_conv.shape',out.shape)
        # print('type of self.decoder_block',type(self.decoder_block),len(self.decoder_block))
        out = out + skip
        out = self.decoder_block[0](out)
        # if self.conv_decoder == True:
        #     out = self.decoder_block[0](out)
        #     # out = self.decoder_block[1](out)
        # else:
        #     out = self.decoder_block[0](out)
        #     # out = self.decoder_block[1](out)

        return out


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


# class Attentionlayer(nn.Module):
#     def __init__(self, dim, head, window_size):
#         super(Attentionlayer, self).__init__()
#         self.dim = dim
#         self.head = head
#         self.window_size = window_size
#
#     def forward(self,x):
#         x1 = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1], w3=self.window_size[2])(x)
#         x2 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0, window_size=self.window_size))(x1)
#         x3 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.))(x2)
#         x4 = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)')(x3)
#
#         x5 = Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1], w3=self.window_size[2])(x4)
#         x6 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0, window_size=self.window_size))(x5)
#         x7 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.))(x6)
#         x8 = Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)')(x7)
#
#         return x8

# class Attentionlayer(nn.Module):
#     def __init__(self, dim, head, window_size):
#         super(Attentionlayer, self).__init__()
#         self.dim = dim
#         self.head = head
#         self.window_size = window_size

#     def forward(self,x):
#         x1 = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1], w3=self.window_size[2]).cuda()(x)
#         x2 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0, window_size=self.window_size).cuda()).cuda()(x1)
#         x3 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.)).cuda()(x2)
#         x4 = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)').cuda()(x3)

#         x5 = Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1], w3=self.window_size[2]).cuda()(x4)
#         x6 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0, window_size=self.window_size).cuda()).cuda()(x5)
#         x7 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.).cuda()).cuda()(x6)
#         x8 = Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)').cuda()(x7)

#         return x8

class Attentionlayer(nn.Module):
    def __init__(self, dim, head, window_size):
        super(Attentionlayer, self).__init__()
        self.dim = dim
        self.head = head
        self.window_size = window_size

    def forward(self, x):
        x1 = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
                       w3=self.window_size[2]).cuda().half()(x)
        x2 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0,
                                                 window_size=self.window_size).cuda().half()).cuda().half()(x1)
        x3 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.)).cuda().half()(x2)
        x4 = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)').cuda().half()(x3)

        x5 = Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
                       w3=self.window_size[2]).cuda().half()(x4)
        x6 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0,
                                                 window_size=self.window_size).cuda().half()).cuda().half()(x5)
        x7 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.).cuda().half()).cuda().half()(x6)
        x8 = Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)').cuda().half()(x7)

        return x8


class Global_Attentionlayer(nn.Module):
    def __init__(self, dim, head, window_size):
        super(Global_Attentionlayer, self).__init__()
        self.dim = dim
        self.head = head
        self.window_size = window_size

    def forward(self,x):

        x = Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1], w3=self.window_size[2]).cuda().half()(x)
        x = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0, window_size=self.window_size).cuda().half()).cuda().half()(x)
        x = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.).cuda().half()).cuda().half()(x)
        x = Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)').cuda().half()(x)

        return x

class Local_Attentionlayer(nn.Module):
    def __init__(self, dim, head, window_size):
        super(Local_Attentionlayer, self).__init__()
        self.dim = dim
        self.head = head
        self.window_size = window_size

    def forward(self,x):
        x1 = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1], w3=self.window_size[2]).cuda().half()(x)
        x2 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0, window_size=self.window_size).cuda().half()).cuda().half()(x1)
        x3 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.)).cuda().half()(x2)
        x4 = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)').cuda().half()(x3)

        return x4


# class Global_Attentionlayer(nn.Module):
#     def __init__(self, dim, head, window_size):
#         super(Global_Attentionlayer, self).__init__()
#         self.dim = dim
#         self.head = head
#         self.window_size = window_size
#
#     def forward(self, x):
#         x = Rearrange('b d (w1 x) (w2 y) (w3 z) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
#                       w3=self.window_size[2]).cuda()(x)
#         x = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0,
#                                                 window_size=self.window_size).cuda()).cuda()(x)
#         x = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.).cuda()).cuda()(x)
#         x = Rearrange('b x y z w1 w2 w3 d -> b d (w1 x) (w2 y) (w3 z)').cuda()(x)
#
#         return x


# class Local_Attentionlayer(nn.Module):
#     def __init__(self, dim, head, window_size):
#         super(Local_Attentionlayer, self).__init__()
#         self.dim = dim
#         self.head = head
#         self.window_size = window_size
#
#     def forward(self, x):
#         x1 = Rearrange('b d (x w1) (y w2) (z w3) -> b x y z w1 w2 w3 d', w1=self.window_size[0], w2=self.window_size[1],
#                        w3=self.window_size[2]).cuda()(x)
#         x2 = PreNormResidual(self.dim, Attention(dim=self.dim, dim_head=self.head, dropout=0,
#                                                  window_size=self.window_size).cuda()).cuda()(x1)
#         x3 = PreNormResidual(self.dim, FeedForward(dim=self.dim, dropout=0.)).cuda()(x2)
#         x4 = Rearrange('b x y z w1 w2 w3 d -> b d (x w1) (y w2) (z w3)').cuda()(x3)
#
#         return x4