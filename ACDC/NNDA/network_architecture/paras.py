# from synapse.unetr_pp_synapse import UNETR_PP 
import torch
import os
from ptflops import get_model_complexity_info
from nnformer.network_architecture.nnFormer_acdc import nnFormer
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# model = UNETR_PP(in_channels=1, out_channels=4, img_size=(16, 160, 160), feature_size=16, num_heads=4,
#             norm_name = 'batch', depths = [3,3,3,3], dims = [32, 64, 128, 256], do_ds = False).cuda()

model = nnFormer(crop_size=(14,160,160),
                        embedding_dim=96,
                        input_channels=1,
                        num_classes=4,
                        conv_op=nn.Conv3d,
                        depths=[2, 2, 2, 2],
                        num_heads=[3, 6, 12, 24],
                        patch_size=[1,4,4],
                        window_size=[[3,5,5],[3,5,5],[7,10,10],[3,5,5]],
                        down_stride=[[1,4,4],[1,8,8],[2,16,16],[4,32,32]],
                        deep_supervision=False)

macs, params = get_model_complexity_info(model, (1,14, 160, 160), as_strings=True,
                                        print_per_layer_stat=False, verbose=True)
                                        
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))