# -*- coding: UTF-8 -*-
import torch
from torch import nn

from torch.autograd import Variable
from torchvision import models
# import cv2
import sys
import time
import numpy as np


def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def replace_inblock_layers(struct_scale,model, i, indexes, layers):
    l_indexes=[]
    for k,(l,b,bl) in enumerate(indexes):
        l_indexes.append(l)
    if i in l_indexes:
        old_layer = model[i]
        new_part = layers[l_indexes.index(i)]
        # print("new part in list %d:"%l_indexes.index(i))
        l, b, bl = new_pos = indexes[l_indexes.index(i)]
        # print("new part at %d,%d,%d" % (l, b, bl))
        if b == -1 and bl == -1:
            new_layer = new_part
        else:
            new_block = None
            blocks=[]
            for k,(_,block) in enumerate(model._modules.items()[i][1]._modules.items()):
                blocks.append(block)
            new_blocks = None
            while new_part is not None:
                new_blocks = []
                for k, block in enumerate(blocks):
                    if k == b:
                        new_block = block
                        for bi, (_,nm) in enumerate(struct_scale[l][b].items()):
                            if bi == bl:
                                if nm == 'conv1':
                                    new_block.conv1 = new_part
                                    # print("new_part is conv and inchannels: %d"%new_block.conv1.in_channels)
                                elif nm == 'bn1':
                                    new_block.bn1 = new_part
                                    # print("new_part is bn and inchannels: %d"%new_block.bn1.num_features)
                                elif nm == 'conv2':
                                    new_block.conv2 = new_part
                                    # print("new_part is conv and inchannels: %d"%new_block.conv2.in_channels)
                                elif nm == 'bn2':
                                    new_block.bn2 = new_part
                                    # print("new_part is bn and inchannels: %d"%new_block.bn2.num_features)
                                elif nm == 'conv3':
                                    new_block.conv3 = new_part
                                    # print("new_part is conv and inchannels: %d"%new_block.conv3.in_channels)
                                elif nm == 'bn3':
                                    new_block.bn3 = new_part
                                    # print("new_part is bn and inchannels: %d"%new_block.bn3.num_features)
                        new_blocks += [new_block]
                    else:
                        new_blocks += [block]
                blocks = new_blocks
                key = l_indexes.index(i)
                l_indexes.pop(key)
                if i in l_indexes:
                    indexes.pop(key)
                    layers.pop(key)
                    new_part = layers[l_indexes.index(i)]
                    # print("new part in list %d:" % l_indexes.index(i))
                    l, b, bl = new_pos = indexes[l_indexes.index(i)]
                    # print("new part at %d,%d,%d" % (l, b, bl))
                else:
                    new_part = None
            new_layer = torch.nn.Sequential(*blocks)
            # print("new layer is layer %d,and the first block inchannels is %d/%d/%d" % (
            # i, blocks[0].conv1.in_channels, blocks[0]._modules.items()[0][1].in_channels,
            # new_layer._modules.items()[0][1]._modules.items()[0][1].in_channels))
        return new_layer
    return model[i]

def replace_crossblock_layers(struct_scale,model, i, indexes, layers, downsamples,downsamples_pos):
    #在剪枝第一个block前的conv后，第一个block的in_channel没有相应的变
    #在第一次剪枝第一个block最后一个conv后,下一个block的第一个conv的in_channel变化也有问题，从256直接变成了64,很不正常
    #原本4，0，4conv被剪枝18次，这样4，1，0conv的inchannels应该从256变成238

    '''
    cross block:1、一个在block之前，一个在block中，两个layer 2、两个在同一layer的相邻block里，3、两个在相邻layer的相邻block里
    在当前layer需要处理的：
    1、conv+btn：next_conv+first block downsample
    2、block_i conv+btn+curb_downsample block_i+1 conv+nextb_downsample
    3、conv+btn+curb_downsample：conv+nextb_downsample
    :param struct_scale: 当前model.feature的结构
    :param model: model.feature
    :param i: model.feature中待处理的某一layer的index
    :param indexes: 所有在一次剪枝操作(剪掉目标位置的一个conv的一个filter)中受影响的部分的具体位置
    :param layers: 所有在一次剪枝操作(剪掉目标位置的一个conv的一个filter)中受影响的部分的新结构
    :param downsamples: 所有在一次剪枝操作(剪掉目标位置的一个conv的一个filter)中受影响的block的downsamples的新结构
    :param downsamples_pos: 所有在一次剪枝操作(剪掉目标位置的一个conv的一个filter)中受影响的block的downsamples的位置
    :return:
    '''
    dcb,dnb,dfb=downsamples
    dcb_p, dnb_p, dfb_p = downsamples_pos
    l_indexes = []
    # print("replace %d layer"%i)
    for k, (l, b, bl) in enumerate(indexes):
        # print("replace:%d,%d,%d"%(l,b,bl))
        l_indexes.append(l)
        '''
        i=0
        1、0，1，2 next_conv的替换不对，没有替换掉
        2、0，0，0，0
        3、0，0，1，1
        '''
    # print("tar_l: %s"%l_indexes)
    if i in l_indexes:
        old_layer = model[i]
        new_part = layers[l_indexes.index(i)]
        # print("new part in list %d:"%l_indexes.index(i))
        l, b, bl = new_pos = indexes[l_indexes.index(i)]
        # print("new part at %d,%d,%d" % (l, b, bl))
        if b == -1 and bl == -1:
            new_layer = new_part
        else:
            new_block = None
            blocks=[]
            for k,(_,block) in enumerate(model._modules.items()[i][1]._modules.items()):
                blocks.append(block)
            new_blocks = None
            while new_part is not None:
                new_blocks = []
                for k, block in enumerate(blocks):
                    if k == b:
                        new_block = block
                        for bi, (_,nm) in enumerate(struct_scale[l][b].items()):
                            if bi == bl:
                                if nm == 'conv1':
                                    new_block.conv1 = new_part
                                    # print("new_part is conv and inchannels: %d"%new_block.conv1.in_channels)
                                elif nm == 'bn1':
                                    new_block.bn1 = new_part
                                    # print("new_part is bn and inchannels: %d"%new_block.bn1.num_features)
                                elif nm == 'conv2':
                                    new_block.conv2 = new_part
                                    # print("new_part is conv and inchannels: %d"%new_block.conv2.in_channels)
                                elif nm == 'bn2':
                                    new_block.bn2 = new_part
                                    # print("new_part is bn and inchannels: %d"%new_block.bn2.num_features)
                                elif nm == 'conv3':
                                    new_block.conv3 = new_part
                                    # print("new_part is conv and inchannels: %d"%new_block.conv3.in_channels)
                                elif nm == 'bn3':
                                    new_block.bn3 = new_part
                                    # print("new_part is bn and inchannels: %d"%new_block.bn3.num_features)
                        if (i,k) == dcb_p:
                            new_block.downsample = dcb
                            # print("new_downsample curb inchannels: %d" % new_block.downsample._modules.items()[0][1].in_channels)
                        if (i,k) == dnb_p:
                            new_block.downsample = dnb
                            # print("new_downsample nextb inchannels: %d" % new_block.downsample._modules.items()[0][1].in_channels)
                        if (i,k) == dfb_p:
                            new_block.downsample = dfb
                            # print("new_downsample firstb inchannels: %d" % new_block.downsample._modules.items()[0][1].in_channels)
                        new_blocks += [new_block]
                    else:
                        new_blocks += [block]
                blocks = new_blocks
                key = l_indexes.index(i)
                l_indexes.pop(key)
                if i in l_indexes:
                    indexes.pop(key)
                    layers.pop(key)
                    new_part = layers[l_indexes.index(i)]
                    # print("new part in list %d:" % l_indexes.index(i))
                    l, b, bl = new_pos = indexes[l_indexes.index(i)]
                    # print("new part at %d,%d,%d" % (l, b, bl))
                else:
                    new_part = None
            new_layer = torch.nn.Sequential(*blocks)
            # print("new layer is layer %d,and the first block inchannels is %d/%d/%d" % (
            # i, blocks[0].conv1.in_channels, blocks[0]._modules.items()[0][1].in_channels,
            # new_layer._modules.items()[0][1]._modules.items()[0][1].in_channels))
        return new_layer
    return model[i]

def prune_vgg16_conv_layer(model, layer_index, filter_index):
    _, conv = model.features._modules.items()[layer_index]
    next_conv = None
    offset = 1

    while layer_index + offset < len(model.features._modules.items()):
        res = model.features._modules.items()[layer_index + offset]
        if isinstance(res[1], torch.nn.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1

    if conv.bias is not None:
        bool_bias = True
    else:
        bool_bias = False
    new_conv = \
        torch.nn.Conv2d(in_channels=conv.in_channels, \
                        out_channels=conv.out_channels - 1,
                        kernel_size=conv.kernel_size, \
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=bool_bias)
    # bias=conv.bias)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    bias_numpy = conv.bias.data.cpu().numpy()

    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1:]
    new_conv.bias.data = torch.from_numpy(bias).cuda()

    # delete filter in one conv layer will only effect the next conv layer(if exist) and won't effect other layers
    if not next_conv is None:
        if next_conv.bias is not None:
            next_bool_bias = True
        else:
            next_bool_bias = False
        next_new_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - 1, \
                            out_channels=next_conv.out_channels, \
                            kernel_size=next_conv.kernel_size, \
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=next_bool_bias)
        # bias=next_conv.bias)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index, layer_index + offset], \
                             [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features

    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [layer_index], \
                             [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear laye found in classifier")
        params_per_input_channel = old_linear_layer.in_features / conv.out_channels

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                             [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model


def prune_resnet_conv_layer(model, m_scale, layer_index, filter_index):
    isBlockEnd_conv = False
    isBlock_conv = False
    isBeforeFirstBlock_conv = False
    l_index, block_index, bl_index = layer_index
    total_l,model_scale,struct_scale = m_scale

    if l_index > -1 and block_index == -1 and bl_index == -1:
        _, conv = model.features._modules.items()[l_index]
        _, bn = model.features._modules.items()[l_index+1]
        isBlock_conv = False
        isBlockEnd_conv = False
        '''***'''
        # print(struct_scale[l_index])
    else:
        checkname, conv = model.features._modules.items()[l_index][1]._modules.items()[block_index][1]._modules.items()[bl_index]
        _, bn = model.features._modules.items()[l_index][1]._modules.items()[block_index][1]._modules.items()[bl_index+1]
        isBlock_conv = True
        if bl_index == len(struct_scale[l_index][block_index]) - 3:
            isBlockEnd_conv = True
        else:
            isBlockEnd_conv = False
        '''***'''
        # print(struct_scale[l_index][block_index][bl_index])
        # print(checkname)

    next_conv = None
    next_l = l_index
    next_block = block_index
    next_bl = bl_index

    if next_conv is None:
        l_offset = 0
        while next_l + l_offset < len(struct_scale):
            if isinstance(struct_scale[next_l + l_offset], str):
                res = model.features._modules.items()[next_l + l_offset]
                if not (
                                next_l + l_offset == l_index and next_block == block_index and next_bl == bl_index) and isinstance(
                        res[1], torch.nn.Conv2d):
                    next_l = next_l + l_offset
                    next_block = -1
                    next_bl = -1
                    next_name, next_conv = res
                    if isBlock_conv:
                        isBlockEnd_conv = True
                    break
                else:
                    l_offset = l_offset + 1
            else:
                if next_block == -1:
                    next_block = 0
                block_offset = 0
                while next_block + block_offset < len(struct_scale[next_l + l_offset]):
                    if next_bl == -1:
                        next_bl = 0
                    bl_offset = 0
                    while next_bl + bl_offset < len(struct_scale[next_l + l_offset][next_block + block_offset]):
                        res= model.features._modules.items()[next_l + l_offset][1]._modules.items()[next_block + block_offset][1]._modules.items()[next_bl + bl_offset]
                        if isinstance(res[1], torch.nn.Conv2d) and not (
                                        next_l + l_offset == l_index and next_block + block_offset == block_index and next_bl + bl_offset == bl_index):
                            next_l = next_l + l_offset
                            next_block = next_block + block_offset
                            next_bl = next_bl + bl_offset
                            next_name, next_conv = res
                            if isBlock_conv and (l_offset > 0 or block_offset > 0):
                                isBlockEnd_conv = True
                            if not isBlock_conv and l_offset > 0 and next_block == 0:
                                isBeforeFirstBlock_conv = True
                            break
                        bl_offset += 1
                    if next_conv is not None:
                        break
                    else:
                        block_offset += 1
                        next_bl = 0
                if next_conv is not None:
                    break
                else:
                    l_offset += 1
                    next_block = 0
                    next_bl = 0

    if next_conv is None and isBlock_conv:
        isBlockEnd_conv = True

    next_layer_index = (next_l, next_block, next_bl)
    # '''***'''
    # if next_conv is not None:
    #     # print("next conv name:%s"%next_name)
    #     # print("next conv pos:%d,%d,%d"%(next_l,next_block,next_bl))
    #     # cmp_next_conv = model.features._modules.items()[next_l][1]._modules.items()[next_block][1]._modules.items()[next_bl][1]
    #     # if isinstance(cmp_next_conv,torch.nn.Conv2d):
    #     #     print("cmp %d"%cmp_next_conv.in_channels)
    #     #     print("cmp %d"%cmp_next_conv.out_channels)
    #     #     print(next_conv.in_channels)
    #     #     print(next_conv.out_channels)
    #     print("compare:")
    #     f_conv = model.features._modules.items()[4][1]._modules.items()[0][1]._modules.items()[0][1]
    #     m_conv = model._modules.items()[4][1]._modules.items()[0][1]._modules.items()[0][1]
    #     print(f_conv.in_channels)
    #     print(m_conv.in_channels)
    #     f_conv = model.features._modules.items()[4][1]._modules.items()[1][1]._modules.items()[0][1]
    #     m_conv = model._modules.items()[4][1]._modules.items()[1][1]._modules.items()[0][1]
    #     print(f_conv.in_channels)
    #     print(m_conv.in_channels)
    #     f_conv = model.features._modules.items()[4][1]._modules.items()[2][1]._modules.items()[0][1]
    #     m_conv = model._modules.items()[4][1]._modules.items()[2][1]._modules.items()[0][1]
    #     print(f_conv.in_channels)
    #     print(m_conv.in_channels)
    #     print("endddd")
    # else:
    #     print("next conv none.")


    if conv.bias is not None:
        bool_bias = True
    else:
        bool_bias = False
    new_conv = \
        torch.nn.Conv2d(in_channels=conv.in_channels, \
                        out_channels=conv.out_channels - 1,
                        kernel_size=conv.kernel_size, \
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=bool_bias)
    new_bn = torch.nn.BatchNorm2d(conv.out_channels-1)
    # print("in %d"%conv.in_channels)
    # print("out %d"%conv.out_channels)
    # print("index %d"%filter_index)
    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    if torch.cuda.is_available():
        new_conv.weight.data = torch.from_numpy(new_weights).cuda()
    else:
        new_conv.weight.data = torch.from_numpy(new_weights).cpu()

    if conv.bias is not None:
        bias_numpy = conv.bias.data.cpu().numpy()
        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index:] = bias_numpy[filter_index + 1:]
        if torch.cuda.is_available():
            new_conv.bias.data = torch.from_numpy(bias).cuda()
        else:
            new_conv.bias.data = torch.from_numpy(bias)

    old_conv_bn_weights = bn.weight.data.cpu().numpy()
    new_conv_bn_weights = new_bn.weight.data.cpu().numpy()
    new_conv_bn_weights[: filter_index] = old_conv_bn_weights[: filter_index]
    new_conv_bn_weights[filter_index:] = old_conv_bn_weights[filter_index + 1:]
    if torch.cuda.is_available():
        new_bn.weight.data = torch.from_numpy(new_conv_bn_weights).cuda()
    else:
        new_bn.weight.data = torch.from_numpy(new_conv_bn_weights)

    conv_bn_bias_numpy = bn.bias.data.cpu().numpy()
    conv_bn_bias = np.zeros(shape=(conv_bn_bias_numpy.shape[0] - 1), dtype=np.float32)
    conv_bn_bias[:filter_index] = conv_bn_bias_numpy[:filter_index]
    conv_bn_bias[filter_index:] = conv_bn_bias_numpy[filter_index + 1:]
    if torch.cuda.is_available():
        new_bn.bias.data = torch.from_numpy(conv_bn_bias).cuda()
    else:
        new_bn.bias.data = torch.from_numpy(conv_bn_bias)

    # delete filter in one conv layer will only effect the next conv layer(if exist) and won't effect other layers
    if not next_conv is None:
        if next_conv.bias is not None:
            next_bool_bias = True
        else:
            next_bool_bias = False
        next_new_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - 1, \
                            out_channels=next_conv.out_channels, \
                            kernel_size=next_conv.kernel_size, \
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=next_bool_bias)
        # print(next_conv.in_channels)
        # print(next_conv.out_channels)
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()
        # print(filter_index)
        # print(old_weights.size)
        # print(new_weights.size)
        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        if torch.cuda.is_available():
            next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()
        else:
            next_new_conv.weight.data = torch.from_numpy(new_weights)

        if next_conv.bias is not None:
            next_new_conv.bias.data = next_conv.bias.data

    old_downsample_curb = None
    new_downsample_curb = None
    new_downsample_curb_pos=None
    old_downsample_nextb = None
    new_downsample_nextb = None
    new_downsample_nextb_pos=None
    old_first_downsample = None
    new_first_downsample = None
    new_first_downsample_pos=None

    if isBlockEnd_conv:
        old_downsample_curb = model.features._modules.items()[l_index][1]._modules.items()[block_index][1].downsample
        new_downsample_curb_pos = (l_index, block_index)
        if old_downsample_curb is None:
            in_channels = model.features._modules.items()[l_index][1]._modules.items()[block_index][1].conv1.in_channels
            stride = model.features._modules.items()[l_index][1]._modules.items()[block_index][1].stride
            new_downsample_curb = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels - 1,
                                kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(in_channels - 1),
            )
        else:
            new_downsample_curb = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=old_downsample_curb._modules.items()[0][1].in_channels,
                                out_channels=old_downsample_curb._modules.items()[0][1].out_channels - 1,
                                kernel_size=old_downsample_curb._modules.items()[0][1].kernel_size,
                                stride=old_downsample_curb._modules.items()[0][1].stride,
                                bias=False),
                torch.nn.BatchNorm2d(old_downsample_curb._modules.items()[1][1].num_features - 1),
            )
            old_conv_weights = old_downsample_curb._modules.items()[0][1].weight.data.cpu().numpy()
            new_conv_weights = new_downsample_curb._modules.items()[0][1].weight.data.cpu().numpy()

            new_conv_weights[: filter_index, :, :, :] = old_conv_weights[: filter_index, :, :, :]
            new_conv_weights[filter_index:, :, :, :] = old_conv_weights[filter_index + 1:, :, :, :]
            if torch.cuda.is_available():
                new_downsample_curb._modules.items()[0][1].weight.data = torch.from_numpy(new_conv_weights).cuda()
            else:
                new_downsample_curb._modules.items()[0][1].weight.data = torch.from_numpy(new_conv_weights)
            # bias = False
            # bias_numpy = old_downsample_curb._modules.items()[0].bias.data.cpu().numpy()
            #
            # bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
            # bias[:filter_index] = bias_numpy[:filter_index]
            # bias[filter_index:] = bias_numpy[filter_index + 1:]
            # new_downsample_curb._modules.items()[0].bias.data = torch.from_numpy(bias).cuda()

            old_bn_weights = old_downsample_curb._modules.items()[1][1].weight.data.cpu().numpy()
            new_bn_weights = new_downsample_curb._modules.items()[1][1].weight.data.cpu().numpy()

            new_bn_weights[: filter_index] = old_bn_weights[: filter_index]
            new_bn_weights[filter_index:] = old_bn_weights[filter_index + 1:]
            if torch.cuda.is_available():
                new_downsample_curb._modules.items()[1][1].weight.data = torch.from_numpy(new_bn_weights).cuda()
            else:
                new_downsample_curb._modules.items()[1][1].weight.data = torch.from_numpy(new_bn_weights)
            bn_bias_numpy = old_downsample_curb._modules.items()[1][1].bias.data.cpu().numpy()

            bn_bias = np.zeros(shape=(bn_bias_numpy.shape[0] - 1), dtype=np.float32)
            bn_bias[:filter_index] = bn_bias_numpy[:filter_index]
            bn_bias[filter_index:] = bn_bias_numpy[filter_index + 1:]
            if torch.cuda.is_available():
                new_downsample_curb._modules.items()[1][1].bias.data = torch.from_numpy(bn_bias).cuda()
            else:
                new_downsample_curb._modules.items()[1][1].bias.data = torch.from_numpy(bn_bias)

        if next_conv is not None and next_block != -1:
            old_downsample_nextb = model.features._modules.items()[next_l][1]._modules.items()[next_block][1].downsample
            new_downsample_nextb_pos = (next_l,next_block)
            if old_downsample_nextb is None:
                in_channels = model.features._modules.items()[next_l][1]._modules.items()[next_block][1].conv1.in_channels
                stride = model.features._modules.items()[next_l][1]._modules.items()[next_block][1].stride
                new_downsample_nextb = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels - 1, out_channels=in_channels,
                                    kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(in_channels),
                )
            else:
                new_downsample_nextb = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=old_downsample_nextb._modules.items()[0][1].in_channels - 1,
                                    out_channels=old_downsample_nextb._modules.items()[0][1].out_channels,
                                    kernel_size=old_downsample_nextb._modules.items()[0][1].kernel_size,
                                    stride=old_downsample_nextb._modules.items()[0][1].stride,
                                    bias=False),
                    torch.nn.BatchNorm2d(old_downsample_nextb._modules.items()[1][1].num_features),
                )
                old_nx_conv_weights = old_downsample_nextb._modules.items()[0][1].weight.data.cpu().numpy()
                new_nx_conv_weights = new_downsample_nextb._modules.items()[0][1].weight.data.cpu().numpy()

                new_nx_conv_weights[:, : filter_index, :, :] = old_nx_conv_weights[:, : filter_index, :, :]
                new_nx_conv_weights[:, filter_index:, :, :] = old_nx_conv_weights[:, filter_index + 1:, :, :]
                if torch.cuda.is_available():
                    new_downsample_nextb._modules.items()[0][1].weight.data = torch.from_numpy(new_nx_conv_weights).cuda()
                else:
                    new_downsample_nextb._modules.items()[0][1].weight.data = torch.from_numpy(new_nx_conv_weights)

                # bias=False
                # new_downsample_nextb._modules.items()[0].bias.data = old_downsample_nextb._modules.items()[0].bias.data

                new_downsample_nextb._modules.items()[1][1].weight.data = old_downsample_nextb._modules.items()[
                    1][1].weight.data
                new_downsample_nextb._modules.items()[1][1].bias.data = old_downsample_nextb._modules.items()[1][1].bias.data
    if isBeforeFirstBlock_conv:
        old_first_downsample = model.features._modules.items()[next_l][1]._modules.items()[next_block][1].downsample
        new_first_downsample_pos = (next_l,next_block)
        if old_first_downsample is None:
            in_channels = model.features._modules.items()[next_l][1]._modules.items()[next_block][1].conv1.in_channels
            stride = model.features._modules.items()[next_l][1]._modules.items()[next_block][1].stride
            new_first_downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels - 1, out_channels=in_channels,
                                kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(in_channels),
            )
        else:
            new_first_downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=old_first_downsample._modules.items()[0][1].in_channels - 1,
                                out_channels=old_first_downsample._modules.items()[0][1].out_channels,
                                kernel_size=old_first_downsample._modules.items()[0][1].kernel_size,
                                stride=old_first_downsample._modules.items()[0][1].stride,
                                bias=False),
                torch.nn.BatchNorm2d(old_first_downsample._modules.items()[1][1].num_features),
            )
            old_f_conv_weights = old_first_downsample._modules.items()[0][1].weight.data.cpu().numpy()
            new_f_conv_weights = new_first_downsample._modules.items()[0][1].weight.data.cpu().numpy()

            new_f_conv_weights[:, : filter_index, :, :] = old_f_conv_weights[:, : filter_index, :, :]
            new_f_conv_weights[:, filter_index:, :, :] = old_f_conv_weights[:, filter_index + 1:, :, :]
            if torch.cuda.is_available():
                new_first_downsample._modules.items()[0][1].weight.data = torch.from_numpy(new_f_conv_weights).cuda()
            else:
                new_first_downsample._modules.items()[0][1].weight.data = torch.from_numpy(new_f_conv_weights)

            # bias=False
            # new_first_downsample._modules.items()[0][1].bias.data = old_first_downsample._modules.items()[0][1].bias.data

            new_first_downsample._modules.items()[1][1].weight.data = old_first_downsample._modules.items()[1][1].weight.data
            new_first_downsample._modules.items()[1][1].bias.data = old_first_downsample._modules.items()[1][1].bias.data

    # TODO: implement replace_inblock_layers and replace_crossblock_layers
    if block_index==-1 and bl_index ==-1:
        bn_layer_index = (l_index+1,block_index,bl_index)
    else:
        bn_layer_index = (l_index, block_index, bl_index+1)
    if not next_conv is None:
        if not isBeforeFirstBlock_conv and not isBlockEnd_conv:
            features = torch.nn.Sequential(
                *(replace_inblock_layers(struct_scale,model.features, i, [layer_index,bn_layer_index,next_layer_index], \
                                         [new_conv, new_bn, next_new_conv]) for i, _ in enumerate(model.features)))
            del model.features
            del conv
            del next_conv
            model.features = features
        else:
            # print("Exp1:")
            # print(new_conv.in_channels)
            # print(new_conv.out_channels)
            # print(new_bn.num_features)
            # # 当前剪枝的conv和bn的out_channel变了，但只有第一次对当前层剪枝的时候下一个conv新的in_channel是对的
            # print(next_conv.in_channels)
            # print(next_conv.out_channels)
            # print(next_new_conv.in_channels)
            # print(next_new_conv.out_channels)
            # print("ennddd")
            features = torch.nn.Sequential(
                *(
                replace_crossblock_layers(struct_scale,model.features, i, [layer_index, bn_layer_index,next_layer_index], [new_conv, new_bn, next_new_conv], \
                                          (new_downsample_curb, new_downsample_nextb, new_first_downsample),(new_downsample_curb_pos, new_downsample_nextb_pos, new_first_downsample_pos)) for i, _ in
                enumerate(model.features)))
            del model.features
            del conv
            del next_conv
            model.features = features
    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        if not isBeforeFirstBlock_conv and not isBlockEnd_conv:
            features = torch.nn.Sequential(
                *(replace_inblock_layers(struct_scale,model.features, i, [layer_index,bn_layer_index], \
                                         [new_conv,new_bn]) for i, _ in enumerate(model.features)))

        else:
            features = torch.nn.Sequential(
                *(replace_crossblock_layers(struct_scale,model.features, i, [layer_index,bn_layer_index], [new_conv,new_bn], \
                                            (new_downsample_curb, new_downsample_nextb, new_first_downsample),(new_downsample_curb_pos, new_downsample_nextb_pos, new_first_downsample_pos)) for i, _
                  in enumerate(model.features)))
        # TODO: ***************************************************************
        layer_index = 0
        old_linear_layer = None
        # for g,h in enumerate(model.fc.modules()):
        #     print(h)
        if isinstance(model.fc, torch.nn.Linear):
            old_linear_layer = model.fc
        else:
            for _, module in model.fc._modules.items():
                # print(module)
                if isinstance(module, torch.nn.Linear):
                    old_linear_layer = module
                    break
                layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")
        params_per_input_channel = old_linear_layer.in_features / conv.out_channels

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel,
                            old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]

        new_linear_layer.bias.data = old_linear_layer.bias.data
        if torch.cuda.is_available():
            new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()
        else:
            new_linear_layer.weight.data = torch.from_numpy(new_weights)
        if layer_index > 0:
            fc = torch.nn.Sequential(
                *(replace_layers(model.fc, i, [layer_index], \
                                 [new_linear_layer]) for i, _ in enumerate(model.fc)))
        else:
            fc = replace_layers(model.fc, layer_index, [layer_index], [new_linear_layer])

        del model.fc
        del next_conv
        del conv
        del model.features
        model.features = features
        model.fc = fc

    # print("HHHHHHHH222222")
    # print(model.features._modules.items()[0][1].in_channels)
    # print(model.features._modules.items()[0][1].out_channels)
    # print(model.features._modules.items()[4][1]._modules.items()[0][1]._modules.items()[0][1].in_channels)
    # print(model.features._modules.items()[5][1]._modules.items()[0][1]._modules.items()[0][1].in_channels)
    # print(model.features._modules.items()[6][1]._modules.items()[0][1]._modules.items()[0][1].in_channels)
    # print(model.features._modules.items()[7][1]._modules.items()[0][1]._modules.items()[0][1].in_channels)
    return model



if __name__ == '__main__':
    model = models.resnet50()
    model.train()
    prune_targets = [(0, 4), (0, 12), (0, 28), (0, 35), (0, 55), (0, 57), (4, 7), (4, 22), (4, 25), (4, 45), (4, 52), (4, 52), (4, 52), (7, 0), (7, 0), (7, 5), (7, 6), (7, 7), (7, 10), (7, 11), (7, 12), (7, 13), (7, 13), (7, 13), (7, 13), (7, 18), (7, 21), (7, 34), (7, 37), (7, 43), (7, 44), (10, 4), (10, 6), (10, 9), (10, 19), (10, 19), (10, 19), (10, 31), (10, 38), (10, 40), (10, 40), (10, 42), (10, 42), (10, 43), (10, 43), (10, 48), (10, 48), (10, 48), (10, 48), (10, 53), (10, 58), (10, 63), (10, 66), (10, 67), (10, 68), (10, 69), (10, 76), (10, 81), (10, 85), (10, 89), (10, 94), (10, 97), (10, 102), (10, 102), (10, 106), (10, 106), (10, 129), (10, 129), (10, 136), (10, 136), (10, 145), (10, 145), (10, 153), (10, 157), (10, 159), (10, 160), (10, 167), (10, 168), (10, 170), (10, 173), (10, 197), (13, 6), (13, 12), (13, 19), (13, 23), (13, 24), (13, 34), (13, 39), (13, 40), (13, 41), (13, 47), (13, 52), (16, 7), (16, 8), (16, 21), (16, 21), (16, 23), (16, 34), (16, 41), (16, 48), (19, 27), (19, 36), (19, 62), (19, 62), (19, 67), (19, 71), (19, 78), (19, 82), (19, 93), (19, 114), (19, 117), (19, 143), (19, 173), (19, 181), (19, 188), (19, 197), (19, 205), (19, 207), (22, 16), (22, 19), (22, 22), (22, 22), (22, 35), (22, 36), (22, 44), (22, 52), (22, 54), (22, 54), (25, 1), (25, 10), (25, 11), (25, 16), (25, 20), (25, 22), (25, 24), (25, 35), (25, 38), (25, 41), (25, 43), (25, 46), (28, 2), (28, 20), (28, 125), (28, 151), (28, 160), (28, 179), (28, 218), (31, 12), (31, 24), (31, 47), (31, 52), (31, 63), (31, 63), (31, 65), (31, 75), (31, 79), (31, 81), (31, 81), (31, 101), (31, 102), (31, 103), (31, 106), (31, 106), (31, 111), (34, 1), (34, 10), (34, 14), (34, 14), (34, 14), (34, 21), (34, 42), (34, 83), (34, 91), (34, 111), (37, 5), (37, 13), (37, 21), (37, 25), (37, 26), (37, 26), (37, 31), (37, 31), (37, 51), (37, 54), (37, 58), (37, 64), (37, 68), (37, 69), (37, 74), (37, 74), (37, 74), (37, 76), (37, 76), (37, 84), (37, 85), (37, 91), (37, 97), (37, 100), (37, 109), (37, 110), (37, 112), (37, 113), (37, 115), (37, 116), (37, 123), (37, 126), (37, 133), (37, 134), (37, 136), (37, 139), (37, 141), (37, 149), (37, 158), (37, 161), (37, 164), (37, 166), (37, 167), (37, 170), (37, 174), (37, 177), (37, 187), (37, 187), (37, 188), (37, 191), (37, 194), (37, 199), (37, 208), (37, 210), (37, 210), (37, 216), (37, 216), (37, 219), (37, 220), (37, 232), (37, 237), (37, 244), (37, 249), (37, 249), (37, 249), (37, 253), (37, 259), (37, 265), (37, 276), (37, 277), (37, 285), (37, 291), (37, 292), (37, 292), (37, 293), (37, 294), (37, 294), (37, 297), (37, 308), (37, 309), (37, 315), (37, 319), (37, 327), (37, 328), (37, 329), (37, 335), (37, 335), (37, 339), (37, 342), (37, 344), (37, 344), (37, 345), (37, 345), (37, 347), (37, 352), (37, 354), (37, 354), (37, 354), (37, 354), (37, 355), (37, 363), (37, 366), (37, 370), (37, 376), (37, 382), (37, 393), (37, 403), (37, 403), (40, 3), (40, 11), (40, 18), (40, 54), (40, 61), (40, 63), (40, 85), (40, 87), (40, 112), (40, 114), (43, 2), (43, 9), (43, 21), (43, 45), (43, 60), (43, 86), (43, 111), (46, 31), (46, 37), (46, 43), (46, 43), (46, 47), (46, 54), (46, 56), (46, 56), (46, 58), (46, 59), (46, 80), (46, 92), (46, 116), (46, 116), (46, 126), (46, 128), (46, 141), (46, 145), (46, 162), (46, 181), (46, 182), (46, 188), (46, 196), (46, 213), (46, 247), (46, 247), (46, 273), (46, 328), (46, 335), (46, 345), (46, 394), (46, 403), (46, 419), (46, 441), (46, 445), (46, 463), (49, 22), (49, 76), (49, 84), (49, 86), (49, 87), (49, 99), (49, 100), (49, 112), (52, 2), (52, 5), (52, 16), (52, 16), (52, 31), (52, 36), (52, 41), (52, 47), (52, 53), (52, 55), (52, 95), (52, 112), (52, 113), (55, 62), (55, 62), (55, 64), (55, 87), (55, 124), (55, 157), (55, 265), (55, 290), (55, 291), (55, 296), (55, 324), (55, 344), (55, 405), (58, 11), (58, 37), (58, 39), (58, 42), (58, 69), (58, 91), (58, 111), (58, 113), (58, 118), (61, 0), (61, 8), (61, 18), (61, 21), (61, 36), (61, 39), (61, 40), (61, 40), (61, 44), (61, 54), (61, 54), (61, 59), (61, 77), (61, 80), (61, 90), (61, 92), (61, 96), (61, 97), (61, 99), (61, 100), (64, 63), (64, 161), (64, 223), (64, 247), (64, 295), (64, 357), (67, 5), (67, 5), (67, 6), (67, 12), (67, 19), (67, 56), (67, 85), (67, 93), (67, 103), (67, 109), (67, 117), (67, 119), (67, 123), (67, 131), (67, 140), (67, 149), (67, 151), (67, 165), (67, 166), (67, 175), (67, 175), (67, 176), (67, 180), (67, 187), (67, 193), (67, 210), (67, 212), (67, 214), (67, 216), (67, 216), (70, 7), (70, 13), (70, 36), (70, 41), (70, 50), (70, 51), (70, 55), (70, 92), (70, 92), (70, 96), (70, 96), (70, 122), (70, 127), (70, 174), (70, 190), (70, 193), (70, 220), (70, 234), (70, 237), (73, 15), (73, 27), (73, 33), (73, 49), (73, 52), (73, 71), (73, 78), (73, 78), (73, 113), (73, 132), (73, 138), (73, 146), (73, 156), (73, 188), (73, 195), (73, 198), (73, 211), (73, 213), (73, 239), (73, 261), (73, 262), (73, 266), (73, 268), (73, 268), (73, 282), (73, 283), (73, 287), (73, 289), (73, 291), (73, 293), (73, 312), (73, 312), (73, 328), (73, 334), (73, 363), (73, 365), (73, 365), (73, 368), (73, 386), (73, 398), (73, 418), (73, 429), (73, 468), (73, 471), (73, 476), (73, 496), (73, 504), (73, 558), (73, 575), (73, 577), (73, 605), (73, 614), (73, 625), (73, 687), (73, 735), (73, 742), (73, 747), (73, 755), (73, 769)]
    for layer_index, filter_index in prune_targets:
        t0 = time.time()
        before_prune_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model = prune_resnet_conv_layer(model, (None, None, None), (4, 0, 0),
                                        filter_index)
        last_prune_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Before %f,After %f,Parameters Compression ratio is %f." % (before_prune_params,last_prune_params,float(before_prune_params-last_prune_params)/before_prune_params))

        print("The prunning took %f" % (time.time() - t0))
