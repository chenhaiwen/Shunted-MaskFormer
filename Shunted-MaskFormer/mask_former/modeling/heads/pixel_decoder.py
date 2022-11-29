# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import math
#from mmcv.ops.deform_conv import DeformConv2d as DCN

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.position_encoding import PositionEmbeddingSine
from ..transformer.transformer import TransformerEncoder, TransformerEncoderLayer


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    # print("name: ", name)
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=256, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        self.scale = scale
        weight_init.c2_xavier_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1)

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)

        return x

class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = Conv2d(out_nc * 2, 144, kernel_size=1, stride=1, padding=0, bias=False) # no norm
        self.dcpack_L2 = DCN(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deform_groups=8)
        self.relu = nn.ReLU(inplace=True)
        weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2(feat_up, offset))  # [feat, offset]
        return feat_align + feat_arm


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_channel, out_channel, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_channel, in_channel, kernel_size=1, bias=False, norm=get_norm(norm, in_channel))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_channel, out_channel, kernel_size=1, bias=False, norm=get_norm('', out_channel))
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):

        avg_feature = F.avg_pool2d(x, x.size()[2:])
        atten = self.sigmoid(self.conv_atten(avg_feature))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat

@SEM_SEG_HEADS_REGISTRY.register()
class MypixelDecoder(nn.Module):
    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            conv_dim: int,
            mask_dim: int,
            norm: Optional[Union[str, Callable]] = None,
    ):
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        strides = [v.stride for k, v in input_shape]
        #[4, 8, 16, 32]
        in_channels_per_feature = [v.channels for k, v in input_shape]
        #print("in_channels_per_feature:", in_channels_per_feature)
        #[64, 128, 256, 512]
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"

        align_modules = []
        output_convs = []

        use_bias = norm == ""
        out_channels = 256
        for idx, in_channels in enumerate(in_channels_per_feature[:-1]):
            stage = int(math.log2(strides[idx]))
            lateral_norm = get_norm(norm, 144)
            align_module = FeatureAlign_V2(in_channels, out_channels, norm=lateral_norm)  # proposed fapn
            self.add_module("fan_align{}".format(stage), align_module)
            align_modules.append(align_module)
            output_conv = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                 norm=get_norm(norm, out_channels), )
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)
            output_convs.append(output_conv)
        stage = int(math.log2(strides[len(in_channels_per_feature) - 1]))
        lateral_conv = Conv2d(in_channels_per_feature[-1], out_channels, kernel_size=1, bias=use_bias,
                              norm=get_norm(norm, out_channels))
        align_modules.append(lateral_conv)
        self.add_module("fan_align{}".format(stage), lateral_conv)
        # Place convs into top-down order (from low to high resolution) to make the top-down computation in forward clearer.
        self.align_modules = align_modules[::-1]
        self.output_convs = output_convs[::-1]
        self.dupsample = DUpsampling(320, 2, num_class=256)


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):  # 'cfg' must be the first argument
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features):

        x = [features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.align_modules[0](x[0])
        results.append(prev_features)
        for feature, align_module, output_conv in zip(x[1:], self.align_modules[1:], self.output_convs[0:]):
            prev_features = align_module(feature, prev_features)
            results.insert(0, output_conv(prev_features))
        out = self.output_convs[-1](prev_features)
        # print("last_feature: ", results[0].size())
        # print("out", out.size())

        #将特征对齐上采样后的特征图res3 ([1, 256, 32, 32])与res2([1,64,64,64])的特征图下采样到“res2”([1,64,32,32])合并后采用DUsimple方法上采样到64
        # feature_res2_down = F.interpolate(features['res2'], [features['res3'].size()[2], features['res3'].size()[3]],
        #                                    mode='bilinear', align_corners=True)
        # low_level_feature = torch.cat((feature_res2_down, results[1]), dim=1)  #chennal: 256+64 = 320 [1,320,32,32]
        # out = self.dupsample(low_level_feature)
        # print("out", out.size())



        return out, None



    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of MyPixelDecoder module.")
        return self.forward_features(features)



@SEM_SEG_HEADS_REGISTRY.register()
class DUpsamplepixelDecoder(nn.Module):

    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            conv_dim: int,
            mask_dim: int,
            norm: Optional[Union[str, Callable]] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(448, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()

        self.lateral_1_conv = FeatureSelectionModule(64, 64, norm="")

        self.conv2 = nn.Conv2d(560, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.5)

        self.lateral_2_conv = FeatureSelectionModule(128, 128, norm="")

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.1)

        self.lateral_3_conv = FeatureSelectionModule(256, 256, norm="")

        self.conv4 = nn.Conv2d(256, 256, kernel_size=1)

        self.lateral_4_conv = FeatureSelectionModule(512, 512, norm="")

        self.lateral_5_conv = FeatureSelectionModule(256, 256, norm="")

        self.dupsample = DUpsampling(256, 8, num_class=256)
        self._init_weight()

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):  # 'cfg' must be the first argument
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features):
        #Shunted Backbone  output: {'res2':[N,64,64,64], 'res3':[N, 128, 32, 32], 'res4':[N, 256, 16, 16], 'res5':[N, 512,8, 8]}

        features_res2_down = F.interpolate(features['res2'], [features['res5'].size()[2], features['res5'].size()[3]], mode='bilinear', align_corners=True)
        features_res2_down_fsm = self.lateral_1_conv(features_res2_down)
        # Shunted Backbone torch.Size([1, 64, 8, 8])
        # torch.Size([1, 128, 20, 20])

        features_res3_down = F.interpolate(features['res3'], [features['res5'].size()[2], features['res5'].size()[3]], mode='bilinear', align_corners=True)
        features_res3_down_fsm = self.lateral_2_conv(features_res3_down)
        # Shunted Backbone torch.Size([1, 128, 8, 8])
        # torch.Size([1, 256, 20, 20])
        features_res4_down = F.interpolate(features['res4'], [features['res5'].size()[2], features['res5'].size()[3]], mode='bilinear', align_corners=True)
        features_res4_down_fsm = self.lateral_3_conv(features_res4_down)
        # Shunted Backbone torch.Size([1, 256, 8, 8])
        # torch.Size([1, 512, 20, 20])

        low_level_feature = torch.cat((features_res2_down, features_res3_down), dim=1)
        low_level_feature = torch.cat((low_level_feature, features_res4_down), dim=1)
        # Shunted Backbone torch.Size([1, 448, 8, 8])
        # low_level_feature = torch.cat((features_res2_down, features['res4']), dim=1)

        # torch.Size([1, 896, 20, 20])
        x = features['res5']
        #x = self.lateral_4_conv(x)

        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)

        x_4_cat = torch.cat((x, low_level_feature), dim=1)
        # Shunted Backbone 48+512=560  [1, 560, 20, 20]
        # channel : 48+1024=1072  [1, 1072, 20, 20]

        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        # [1, 256, 20, 20]
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        # [1, 256, 20, 20]
        x_4_cat = self.conv4(x_4_cat)
        #x_4_cat = self.lateral_5_conv(x_4_cat)

        out = self.dupsample(x_4_cat)
        #torch.Size([1, 256, 64, 64])

        # print("out:", out.size())

        return out, None

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of MyPixelDecoder module.")
        return self.forward_features(features)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


@SEM_SEG_HEADS_REGISTRY.register()
class BasePixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        #[('res2', ShapeSpec(channels=128, height=None, width=None, stride=4)),
        # ('res3', ShapeSpec(channels=256, height=None, width=None, stride=8)),
        # ('res4', ShapeSpec(channels=512, height=None, width=None, stride=16)),
        # ('res5', ShapeSpec(channels=1024, height=None, width=None, stride=32))]
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        #['res2', 'res3', 'res4', 'res5']
        feature_channels = [v.channels for k, v in input_shape]
        #[128, 256, 512, 1024]


        lateral_convs = []
        output_convs = []

        use_bias = norm == ""  #不使用归一化就使用偏置
        for idx, in_channels in enumerate(feature_channels):
            #最后一层
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                #调整每个特征层都输出256个通道
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        #这里将所有的层次反转
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        #['res5', 'res4', 'res3', 'res2']
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                #size的参数就是前一层的大小，将y上采样2倍后与前一层相加
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        #print("out: ", self.mask_features(y).size())
        return self.mask_features(y), None

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)


class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


@SEM_SEG_HEADS_REGISTRY.register()
class TransformerEncoderPixelDecoder(BasePixelDecoder):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        transformer_pre_norm: bool,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        return ret

    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                # save intermediate feature as input to Transformer decoder
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y), transformer_encoder_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)
