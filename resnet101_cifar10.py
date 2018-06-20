# -*- coding: UTF-8 -*-
from __future__ import print_function, division
from torchvision import models
import os
import sys
import torch
from torch import nn
from torch import optim
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms, utils
from torch.autograd import Variable
import numpy as np
from torch.optim import lr_scheduler
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn.functional as F
from PIL import Image
import glob
import argparse
from operator import itemgetter
from heapq import nsmallest
from prune import *


class DataSet:
    def __init__(self, torch_v=0.4):
        self.torch_v = torch_v
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_sizes = 0
        self.test_sizes = 0

    def train_loader(self, path, batch_size=32, num_workers=4, pin_memory=True):
        '''normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'''
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if self.torch_v == 0.3:
            resize = transforms.RandomSizedCrop(224)
        else:
            resize = transforms.RandomResizedCrop(224)

        traindata_transforms = transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        train_dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=traindata_transforms)
        self.train_sizes = len(train_dataset)
        return DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory)

    def test_loader(self, path, batch_size=32, num_workers=4, pin_memory=True):
        '''normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'''
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if self.torch_v == 0.3:
            resize = transforms.Scale(256)
        else:
            resize = transforms.Resize(256)
        testdata_transforms = transforms.Compose([
            resize,
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        test_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=testdata_transforms)
        self.test_sizes = len(test_dataset)
        return DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory)

class FilterPrunner:
    '''
    CNN Model Filter Prunner
    Implement the basic functions of prunning operations,include:
    forward: Before computing rank, it need do forward computation through network to get necessary params values
    compute_rank: compute oracle ranking of filters in each layer (all filters in the network)
    normalize_ranks_per_layer: useful normalization operations to the computed rank (mentioned on the paper without clear reasons)
    lowest_ranking_filters: sort the ranking list and find the n-smallest filters
    get_prunning_plan: get target n-smallest filters to be pruned in a list of [(layer-l,filter-i)]
    '''
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        # l = []
        # for i, (nm , md) in enumerate(self.model.features._modules.items()):
        #     for j ,(nm1, md1) in enumerate(md._modules.items()):
        #         l.append((nm+'_'+nm1, md1))
        l=0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            # x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x = module(x)
                # print("module_name: %s, out size: %s"%(name,x.size()))
                x.register_hook(self.compute_rank) # How is compute_rank function used?
                self.activations.append(x)
                # self.activation_to_layer[activation_index] = layer
                self.activation_to_layer[activation_index] = l
                activation_index += 1
                l += 1
            elif isinstance(module, torch.nn.modules.Sequential):
                for ly, (nm, md) in enumerate(module._modules.items()):
                    # each md is a block, which includes several conv2D layers and other layers
                    residual = x
                    block = md._modules.items()
                    length= len(block)
                    lsnm,lsmd = block[length-1]
                    scnm,scmd = block[length-2]
                    if lsnm == 'downsample':
                        for ly1, (nm1, md1) in enumerate(block[0:length-2]):
                            # print("module_name: %s, in size: %s" % (nm1, x.size()))
                            # print("module out channels: %d" % md1.out_channels)
                            x = md1(x)
                            # print("module_name: %s, out size: %s" % (nm1, x.size()))
                            if isinstance(md1, torch.nn.modules.conv.Conv2d):
                                x.register_hook(self.compute_rank)  # How is compute_rank function used?
                                self.activations.append(x)
                                # self.activation_to_layer[activation_index] = layer
                                self.activation_to_layer[activation_index] = l
                                activation_index += 1
                                l += 1
                            elif isinstance(md1, torch.nn.modules.BatchNorm2d) and ly1 != length-3:
                                x = scmd(x)
                                l += 2
                            elif isinstance(md1, torch.nn.modules.BatchNorm2d) and ly1 == length-3:
                                l += 1
                        residual = lsmd(residual)
                        x += residual
                        x = scmd(x)
                        l += 1
                    elif lsnm == 'relu':
                        for ly1, (nm1, md1) in enumerate(block[0:length-1]):
                            x = md1(x)
                            if isinstance(md1, torch.nn.modules.conv.Conv2d):
                                x.register_hook(self.compute_rank)  # How is compute_rank function used?
                                self.activations.append(x)
                                # self.activation_to_layer[activation_index] = layer
                                self.activation_to_layer[activation_index] = l
                                activation_index += 1
                                l += 1
                            elif isinstance(md1, torch.nn.modules.BatchNorm2d) and ly1 != length - 2:
                                x = lsmd(x)
                                l += 2
                            elif isinstance(md1, torch.nn.modules.BatchNorm2d) and ly1 == length - 2:
                                l += 1
                        x += residual
                        x = lsmd(x)
                        l += 1
            else:
                x = module(x)
                # print("module_name: %s, out size: %s"%(name,x.size()))
                l += 1

        return self.model.fc(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        values = \
            torch.sum((activation * grad), dim=0, keepdim=True). \
                sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data

        # Normalize the rank by the filter dimensions
        values = \
            values / (activation.size(0) * activation.size(2) * activation.size(3))

        if activation_index not in self.filter_ranks:
            if torch.cuda.is_available():
                self.filter_ranks[activation_index] = \
                    torch.FloatTensor(activation.size(1)).zero_().cuda()
            else:
                self.filter_ranks[activation_index] = \
                    torch.FloatTensor(activation.size(1)).zero_()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def get_struct_filter_position(self, model_scale, struct_scale, layer_index, filter_index=None):
        l_index =-1
        block_index=-1
        bl_index = -1
        tar = layer_index+1
        for layer in range(len(model_scale)):
            if isinstance(model_scale[layer], str):
                tar -= 1
                if tar == 0:
                    l_index=layer
                    block_index=-1
                    bl_index=-1
                    return l_index,block_index,bl_index
            else:
                for block in range(len(model_scale[layer])):
                    bb_layer = 0
                    for b_layer in range(len(model_scale[layer][block])):
                        mnm = model_scale[layer][block][b_layer]
                        if mnm == 'relu' and b_layer != (len(model_scale[layer][block])-1):
                            bb_layer -= 1
                        tar -= 1
                        if tar == 0:
                            l_index = layer
                            block_index = block
                            bl_index = bb_layer
                            return l_index, block_index, bl_index
                        bb_layer += 1

        if tar > 0:
            raise BaseException("Layer index is out of model layer range.")
        return l_index, block_index, bl_index

    def lowest_ranking_filters(self, num,model_scale=None, struct_scale=None):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                l_index, block_index, bl_index =self.get_struct_filter_position(model_scale,struct_scale,self.activation_to_layer[i])
                if l_index>=4 and bl_index==4:
                    data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]+1000))
                else:
                    data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune,model_scale=None,struct_scale=None):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune,model_scale,struct_scale)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, value) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)
            # print("rank value %s" % value)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune

class PrunningFineTuner_ResNet50:

    def __init__(self, train_path, test_path, model, use_gpu, torch_version=0.4):

        self.torch_version = torch_version
        self.use_gpu = use_gpu
        dataset = DataSet(torch_v=self.torch_version)
        self.train_data_loader = dataset.train_loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)
        self.batch_size = 32
        self.classes = dataset.classes
        self.train_sizes = dataset.train_sizes
        self.test_sizes = dataset.test_sizes

        self.resnet101 = model
        # self.criterion = torch.nn.CrossEntropyLoss()
        # define the loss function
        self.loss_function = nn.CrossEntropyLoss()
        # define the optimizer
        self.optimizer = optim.SGD(self.resnet101.parameters(), lr=0.1, momentum=0.9)
        # define the changing strategy of learning rate
        # learning rate will be changed into gamma*lr after each step_size epochs

        self.prunner = FilterPrunner(self.resnet101)

        self.inputs, self.labels, self.preds = None, None, None
        self.outputs, self.loss = None, None
        self.best_acc = 0.0
        self.best_model_wts = None

        # set model to training mode
        # the default Batch Normalization and Dropout mode is different in train & eval
        self.resnet101.train()

    def model_train(self, optimizer=None, epoches=10,regular_step_size = 20, regular_gamma = 0.1):
        self.resnet101.train()
        use_scheduler = False
        if optimizer is None:
            # allow only fine-tuning fc layer
            optimizer = \
                optim.SGD(self.resnet101.fc.parameters(),
                          lr=0.0001, momentum=0.9)
            use_scheduler = False
            # Make sure all the layers except fc layer are not trainable
            for param in self.resnet101.features.parameters():
                param.requires_grad = False

        else:
            optimizer = optimizer
            use_scheduler = True
            self.scheduler = lr_scheduler.StepLR(optimizer, step_size=regular_step_size, gamma=regular_gamma)

        self.best_model_wts = self.resnet101.state_dict()
        self.best_acc = 0.0

        since = time.time()

        for epoch in range(epoches):
            print('Epoch {}/{}'.format(epoch, epoches - 1))
            print('-' * 10)
            if use_scheduler:
                self.scheduler.step()

            self.train_epoch(epoch=epoch, optimizer=optimizer)

            test_acc = self.model_test_total()
            # deep copy the model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_model_wts = self.resnet101.state_dict()
                torch.save(self.resnet101.state_dict(), './resnet101_params_cifar10.pth')

        time_elapsed = time.time() - since
        print("Finished fine tuning.")
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_acc))
        self.resnet101.load_state_dict(self.best_model_wts)
        return self.best_acc

    def train_epoch(self, epoch=-1, optimizer=None, rank_filters=False):

        epoch_running_loss = 0
        epoch_running_acc = 0
        running_loss = 0
        # generate data pytorch input variables
        total_count = 0
        total_traindata = 0
        for i, data in enumerate(self.train_data_loader, 0):
            self.inputs, self.labels = data

            if self.use_gpu:
                self.inputs = self.inputs.cuda()
                self.labels = self.labels.cuda()
            if self.torch_version == 0.3:
                self.inputs = Variable(self.inputs)
                self.labels = Variable(self.labels)

            if optimizer is not None:
                optimizer.zero_grad()
            self.resnet101.zero_grad()

            if rank_filters:
                output = self.prunner.forward(self.inputs)
                self.loss_function(output, self.labels).backward()
            else:
                # forward
                self.outputs = self.resnet101(self.inputs)
                # select the max probability class: arguments (data, dim, loss)
                _, self.preds = torch.max(self.outputs.data, 1)
                self.loss = self.loss_function(self.outputs, self.labels)

                # backward
                self.loss.backward()
                optimizer.step()

                # statistic
                if self.torch_version == 0.4:
                    epoch_running_loss += self.loss.item()
                    running_loss += self.loss.item()
                elif self.torch_version == 0.3:
                    epoch_running_loss += self.loss.data[0]
                    running_loss += self.loss.data[0]
                epoch_running_acc += torch.sum(self.preds == self.labels.data)

                if i % 250 == 249:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 250))
                    running_loss = 0.0

            total_count += 1
            total_traindata += self.labels.size(0)

        if not rank_filters:
            epoch_loss = epoch_running_loss / total_count
            epoch_acc = epoch_running_acc / total_traindata
            print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def model_test_total(self):
        self.resnet101.eval()
        correct = 0
        total = 0
        if self.torch_version == 0.4:
            with torch.no_grad():
                for data in self.test_data_loader:
                    images, labels = data
                    if self.use_gpu:
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        images = Variable(images)
                        labels = Variable(labels)
                    outputs = self.resnet101(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        elif self.torch_version == 0.3:
            for data in self.test_data_loader:
                images, labels = data
                if self.use_gpu:
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                else:
                    images = Variable(images)
                    labels = Variable(labels)
                outputs = self.resnet101(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.data).sum()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

        '''
        correct = 0
        total = 0
        for data in self.testloader:
            images, labels = data
            if self.use_gpu:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)
            outputs = self.resnet101(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        '''

        self.resnet101.train()

        return float(correct) / total

    def model_test_onestep(self):
        self.resnet101.eval()
        dataiter = iter(self.test_data_loader)
        images, labels = dataiter.next()
        # print images
        # self.imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % self.classes[labels[j]] for j in range(4)))

        test_outputs = self.resnet101(images)

        _, predicted = torch.max(test_outputs, 1)

        print('Predicted: ', ' '.join('%5s' % self.classes[predicted[j]]
                                      for j in range(4)))

    def model_test_diffclasses(self):
        self.resnet101.eval()
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in self.test_data_loader:
                images, labels = data
                outputs = self.resnet101(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))

    def get_candidates_to_prune(self, num_filters_to_prune, model_scale=None,struct_scale=None):
        self.prunner.reset()
        # 虽然调用的是train，但其实在rank_filters设置为True时，train_epoch中做的只是forward与backward(计算grad)计算，没有优化参数(没有更新)
        # 这些forward与backward计算中得到的必要参数用于计算oracle_rank, compute_rank函数被register了，使其可以在forward/backward 的同时被调用计算
        # 具体调用compute_rank的原理要再看一下
        self.train_epoch(rank_filters=True)

        self.prunner.normalize_ranks_per_layer()

        return self.prunner.get_prunning_plan(num_filters_to_prune,model_scale,struct_scale)

    def total_num_filters(self):
        filters = 0
        for name, module in self.resnet101.features._modules.items():
            # print('**** %s ****' % name)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
                # print(filters)
            for nm, md in module._modules.items():
                # print('--- %s ---' % nm)
                if isinstance(md, torch.nn.modules.conv.Conv2d):
                    filters = filters + md.out_channels
                    # print(filters)
                for nm1, md1 in md._modules.items():
                    # print(nm1)
                    if isinstance(md1, torch.nn.modules.conv.Conv2d):
                        filters = filters + md1.out_channels
                        # print(filters)
        return filters

    def get_model_scales(self):
        l = {}
        total_l = 0
        for layer, (name, module) in enumerate(self.resnet101.features._modules.items()):
            if isinstance(module, torch.nn.modules.Sequential):
                l[layer]={}
                for ly, (nm, md) in enumerate(module._modules.items()):
                    # each md is a block, which includes several conv2D layers and other layers
                    l[layer][ly] = {}
                    block = md._modules.items()
                    length = len(block)
                    lsnm, _ = block[length - 1]
                    if lsnm == 'downsample':
                        ll=0
                        for ly1, (nm1, md1) in enumerate(block[0:length - 2]):
                            if isinstance(md1, torch.nn.modules.conv.Conv2d):
                                l[layer][ly][ll]=nm1
                                ll += 1
                                total_l += 1
                            elif isinstance(md1, torch.nn.modules.BatchNorm2d) and ly1 != length - 3:
                                l[layer][ly][ll]=nm1
                                ll += 1
                                l[layer][ly][ll]='relu'
                                ll += 1
                                total_l +=2
                            elif isinstance(md1, torch.nn.modules.BatchNorm2d) and ly1 == length - 3:
                                l[layer][ly][ll]=nm1
                                ll += 1
                                total_l += 1
                        l[layer][ly][ll] = 'relu'
                        ll += 1
                        total_l += 1
                    elif lsnm == 'relu':
                        ll = 0
                        for ly1, (nm1, md1) in enumerate(block[0:length - 1]):
                            if isinstance(md1, torch.nn.modules.conv.Conv2d):
                                l[layer][ly][ll] = nm1
                                ll += 1
                                total_l += 1
                            elif isinstance(md1, torch.nn.modules.BatchNorm2d) and ly1 != length - 2:
                                l[layer][ly][ll] = nm1
                                ll += 1
                                l[layer][ly][ll] = 'relu'
                                ll += 1
                                total_l +=2
                            elif isinstance(md1, torch.nn.modules.BatchNorm2d) and ly1 == length - 2:
                                l[layer][ly][ll] = nm1
                                ll += 1
                                total_l +=1
                        l[layer][ly][ll] = 'relu'
                        ll += 1
                        total_l +=1
            else:
                l[layer]=name
                total_l += 1
        return total_l, l

    def get_structure_scales(self):
        l = {}
        for layer, (name, module) in enumerate(self.resnet101.features._modules.items()):
            if isinstance(module, torch.nn.modules.Sequential):
                l[layer] = {}
                for ly, (nm, md) in enumerate(module._modules.items()):
                    # each md is a block, which includes several conv2D layers and other layers
                    l[layer][ly] = {}
                    block = md._modules.items()
                    length = len(block)
                    lsnm, _ = block[length - 1]
                    if lsnm == 'downsample':
                        ll = 0
                        for ly1, (nm1, md1) in enumerate(block[0:length - 1]):
                            l[layer][ly][ll] = nm1
                            ll += 1
                    elif lsnm == 'relu':
                        ll = 0
                        for ly1, (nm1, md1) in enumerate(block[0:length]):
                            l[layer][ly][ll] = nm1
                            ll += 1
            else:
                l[layer] = name
        return l

    def get_exact_filter_position(self, model_scale, layer_index, filter_index):
        #crosponding to model_scale
        l_index =-1
        block_index=-1
        bl_index = -1
        tar = layer_index+1
        for layer in range(len(model_scale)):
            if isinstance(model_scale[layer], str):
                tar -= 1
                if tar == 0:
                    l_index=layer
                    block_index=-1
                    bl_index=-1
                    return l_index,block_index,bl_index
            else:
                for block in range(len(model_scale[layer])):
                    for b_layer in range(len(model_scale[layer][block])):
                        tar -= 1
                        if tar == 0:
                            l_index = layer
                            block_index = block
                            bl_index = b_layer
                            return l_index, block_index, bl_index
        if tar > 0:
            raise BaseException("Layer index is out of model layer range.")
        return l_index, block_index, bl_index

    def get_struct_filter_position(self, model_scale, struct_scale, layer_index, filter_index=None):
        # 由于layer_index是对应着model_scale的,所以需要model_scale参数，目标是将model_scale中的位置映射到struct_scale中的位置
        #TODO: 改的不要跟model_scale对应了，而是跟原本的实现结构对应，传入prunner的model_scale也可以改成和实现结构对应
        # ，对prunner来说其实不需要准确的model_scale，只要和结构对应的就可以了，可以根据现在的model_scale中的exact位置反推回结构中的位置
        l_index =-1
        block_index=-1
        bl_index = -1
        tar = layer_index+1
        for layer in range(len(model_scale)):
            if isinstance(model_scale[layer], str):
                tar -= 1
                if tar == 0:
                    l_index=layer
                    block_index=-1
                    bl_index=-1
                    return l_index,block_index,bl_index
            else:
                for block in range(len(model_scale[layer])):
                    bb_layer = 0
                    for b_layer in range(len(model_scale[layer][block])):
                        mnm = model_scale[layer][block][b_layer]
                        if mnm == 'relu' and b_layer != (len(model_scale[layer][block])-1):
                            bb_layer -= 1
                        tar -= 1
                        if tar == 0:
                            l_index = layer
                            block_index = block
                            bl_index = bb_layer
                            return l_index, block_index, bl_index
                        bb_layer += 1

        if tar > 0:
            raise BaseException("Layer index is out of model layer range.")
        return l_index, block_index, bl_index

    def prune(self):
        # Get the accuracy before prunning
        before_prune_acc = self.model_test_total()

        self.resnet101.train()

        # Make sure all the layers are trainable
        for param in self.resnet101.parameters():
            param.requires_grad = True

        before_prune_params = sum(p.numel() for p in self.resnet101.parameters() if p.requires_grad)

        total_l, model_scale=self.get_model_scales()
        struct_scale = self.get_structure_scales()

        number_of_filters = self.total_num_filters()
        # print(number_of_filters)
        num_filters_to_prune_per_iteration = 1024
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        # iterations = int(iterations * 2.0 / 3)

        print("Max number of prunning iterations: %d " % iterations)

        # iterations = 1
        for it_k in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration,model_scale,struct_scale)
            # prune_targets =[(0, 4), (0, 12), (0, 28), (0, 35), (0, 55), (0, 57), (4, 7), (4, 22), (4, 25), (4, 45), (4, 52),
            #           (4, 52), (4, 52), (7, 0), (7, 0), (7, 5), (7, 6), (7, 7), (7, 10), (7, 11), (7, 12), (7, 13),
            #           (7, 13), (7, 13), (7, 13), (7, 18), (7, 21), (7, 34), (7, 37), (7, 43), (7, 44), (10, 4), (10, 6),
            #           (10, 9), (10, 19), (10, 19), (10, 19), (10, 31), (10, 38), (10, 40), (10, 40), (10, 42), (10, 42),
            #           (10, 43), (10, 43), (10, 48), (10, 48), (10, 48), (10, 48), (10, 53), (10, 58), (10, 63),
            #           (10, 66), (10, 67), (10, 68), (10, 69), (10, 76), (10, 80), (10, 80), (10, 84), (10, 88),
            #           (10, 93), (10, 96), (10, 101), (10, 101), (10, 105), (10, 105), (10, 128), (10, 128), (10, 135),
            #           (10, 135), (10, 144), (10, 144), (10, 152), (10, 156), (10, 158), (10, 159), (10, 168), (10, 170),
            #           (10, 173), (10, 197), (13, 6), (13, 12), (13, 19), (13, 23), (13, 24), (13, 34), (13, 39),
            #           (13, 40), (13, 41), (13, 47), (13, 52), (16, 7), (16, 8), (16, 21), (16, 21), (16, 23), (16, 34),
            #           (16, 41), (16, 45), (16, 47), (19, 27), (19, 36), (19, 62), (19, 62), (19, 67), (19, 71),
            #           (19, 78), (19, 82), (19, 93), (19, 97), (19, 113), (19, 116), (19, 142), (19, 172), (19, 180),
            #           (19, 187), (19, 196), (19, 204), (19, 206), (22, 16), (22, 19), (22, 22), (22, 22), (22, 35),
            #           (22, 36), (22, 44), (22, 52), (22, 54), (22, 54), (25, 1), (25, 10), (25, 11), (25, 16), (25, 20),
            #           (25, 22), (25, 24), (25, 35), (25, 38), (25, 41), (25, 43), (25, 46), (28, 2), (28, 20),
            #           (28, 125), (28, 151), (28, 160), (28, 179), (28, 218), (31, 12), (31, 24), (31, 35), (31, 46),
            #           (31, 49), (31, 50), (31, 61), (31, 61), (31, 63), (31, 73), (31, 77), (31, 79), (31, 79),
            #           (31, 85), (31, 98), (31, 99), (31, 100), (31, 103), (31, 103), (31, 108), (34, 1), (34, 9),
            #           (34, 9), (34, 13), (34, 13), (34, 13), (34, 20), (34, 41), (34, 77), (34, 81), (34, 89),
            #           (34, 109), (37, 5), (37, 13), (37, 21), (37, 25), (37, 26), (37, 26), (37, 31), (37, 31),
            #           (37, 51), (37, 54), (37, 58), (37, 64), (37, 68), (37, 69), (37, 74), (37, 74), (37, 74),
            #           (37, 76), (37, 76), (37, 84), (37, 85), (37, 91), (37, 97), (37, 100), (37, 109), (37, 110),
            #           (37, 112), (37, 113), (37, 115), (37, 116), (37, 123), (37, 126), (37, 133), (37, 134), (37, 136),
            #           (37, 139), (37, 141), (37, 149), (37, 158), (37, 161), (37, 164), (37, 166), (37, 167), (37, 170),
            #           (37, 174), (37, 177), (37, 187), (37, 187), (37, 188), (37, 191), (37, 194), (37, 199), (37, 208),
            #           (37, 210), (37, 210), (37, 216), (37, 216), (37, 219), (37, 220), (37, 232), (37, 237), (37, 244),
            #           (37, 249), (37, 249), (37, 249), (37, 253), (37, 259), (37, 265), (37, 276), (37, 277), (37, 285),
            #           (37, 291), (37, 292), (37, 292), (37, 293), (37, 294), (37, 294), (37, 297), (37, 308), (37, 309),
            #           (37, 315), (37, 319), (37, 327), (37, 328), (37, 329), (37, 335), (37, 335), (37, 339), (37, 342),
            #           (37, 344), (37, 344), (37, 345), (37, 345), (37, 347), (37, 352), (37, 354), (37, 354), (37, 354),
            #           (37, 354), (37, 355), (37, 363), (37, 366), (37, 370), (37, 376), (37, 382), (37, 393), (37, 403),
            #           (37, 403), (40, 3), (40, 11), (40, 18), (40, 54), (40, 61), (40, 63), (40, 85), (40, 87),
            #           (40, 112), (40, 114), (43, 2), (43, 9), (43, 21), (43, 45), (43, 60), (43, 86), (43, 111),
            #           (46, 31), (46, 37), (46, 43), (46, 43), (46, 47), (46, 54), (46, 56), (46, 56), (46, 60),
            #           (46, 81), (46, 93), (46, 117), (46, 117), (46, 127), (46, 129), (46, 142), (46, 146), (46, 163),
            #           (46, 182), (46, 183), (46, 189), (46, 197), (46, 214), (46, 248), (46, 248), (46, 272), (46, 273),
            #           (46, 328), (46, 335), (46, 345), (46, 387), (46, 393), (46, 402), (46, 418), (46, 440), (46, 444),
            #           (46, 462), (49, 22), (49, 76), (49, 84), (49, 86), (49, 87), (49, 99), (49, 100), (49, 112),
            #           (52, 2), (52, 5), (52, 16), (52, 16), (52, 31), (52, 36), (52, 41), (52, 47), (52, 53), (52, 55),
            #           (52, 95), (52, 112), (52, 113), (55, 31), (55, 61), (55, 61), (55, 63), (55, 86), (55, 123),
            #           (55, 156), (55, 264), (55, 289), (55, 290), (55, 292), (55, 294), (55, 322), (55, 342), (55, 403),
            #           (58, 11), (58, 37), (58, 39), (58, 42), (58, 69), (58, 112), (58, 114), (58, 119), (61, 0),
            #           (61, 8), (61, 18), (61, 21), (61, 36), (61, 39), (61, 40), (61, 40), (61, 44), (61, 54), (61, 54),
            #           (61, 59), (61, 77), (61, 80), (61, 90), (61, 92), (61, 96), (61, 97), (61, 99), (61, 100),
            #           (64, 63), (64, 161), (64, 223), (64, 247), (64, 295), (64, 357), (67, 5), (67, 5), (67, 6),
            #           (67, 12), (67, 19), (67, 56), (67, 64), (67, 84), (67, 92), (67, 102), (67, 108), (67, 116),
            #           (67, 118), (67, 122), (67, 130), (67, 139), (67, 148), (67, 150), (67, 164), (67, 165), (67, 174),
            #           (67, 174), (67, 175), (67, 179), (67, 186), (67, 192), (67, 209), (67, 211), (67, 213), (67, 215),
            #           (67, 215), (70, 7), (70, 13), (70, 36), (70, 41), (70, 50), (70, 51), (70, 55), (70, 92),
            #           (70, 92), (70, 96), (70, 96), (70, 122), (70, 127), (70, 174), (70, 190), (70, 193), (70, 220),
            #           (70, 234), (70, 237), (73, 15), (73, 27), (73, 33), (73, 49), (73, 52), (73, 71), (73, 78),
            #           (73, 78), (73, 93), (73, 112), (73, 131), (73, 137), (73, 145), (73, 155), (73, 187), (73, 194),
            #           (73, 197), (73, 210), (73, 212), (73, 238), (73, 260), (73, 261), (73, 265), (73, 267), (73, 267),
            #           (73, 281), (73, 282), (73, 286), (73, 288), (73, 290), (73, 292), (73, 311), (73, 311), (73, 327),
            #           (73, 333), (73, 362), (73, 364), (73, 364), (73, 367), (73, 385), (73, 397), (73, 417), (73, 428),
            #           (73, 467), (73, 470), (73, 475), (73, 495), (73, 503), (73, 557)]

            print("Layers that will be prunned: %s" % prune_targets)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned: %s" % layers_prunned)

            print("Prunning filters.. ")
            model = self.resnet101.cpu()

            # print("model:%s" % model_scale)
            # print("struct model:%s" % struct_scale)
            for layer_index, filter_index in prune_targets:
                # l_index, block_index, bl_index = self.get_exact_filter_position(model_scale, layer_index, filter_index)
                # print("l=%d,block=%d,bl=%d"%(l_index, block_index, bl_index))
                l_index, block_index, bl_index = self.get_struct_filter_position(model_scale,struct_scale, layer_index, filter_index)

                if l_index>=4 and bl_index==4:
                    print("Wrong ranking.")
                    continue
                # print("l=%d,block=%d,bl=%d" % (l_index, block_index, bl_index))
                # _,temp_model_scale=self.get_model_scales()
                # temp_struct_scale = self.get_structure_scales()
                #print("current model_scale:%s"%temp_model_scale)
                #print("current struct_scale:%s"%temp_struct_scale)
                # model = prune_resnet101_conv_layer(model, (total_l, model_scale), (l_index, block_index, bl_index), filter_index)
                model = prune_resnet_conv_layer(model, (total_l, model_scale, struct_scale), (l_index, block_index, bl_index), filter_index)

            if self.use_gpu:
                self.resnet101 = model.cuda()
            else:
                self.resnet101 = model
            self.prunner.model = self.resnet101
            struct_scale_temp = self.get_structure_scales()

            if struct_scale_temp == struct_scale:
                print("The same struct!")
            else:
                print("Not the same struct!")
                return
            # f = self.resnet101.features
            # print("HHHHHHHH1111111")
            # print(f._modules.items()[0][1].in_channels)
            # print(f._modules.items()[0][1].out_channels)
            # print(f._modules.items()[4][1]._modules.items()[0][1]._modules.items()[0][1].in_channels)
            # print(f._modules.items()[5][1]._modules.items()[0][1]._modules.items()[0][1].in_channels)
            # print(f._modules.items()[6][1]._modules.items()[0][1]._modules.items()[0][1].in_channels)
            # print(f._modules.items()[7][1]._modules.items()[0][1]._modules.items()[0][1].in_channels)

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned: %s" % str(message))
            self.model_test_total()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.resnet101.parameters(), lr=0.1, momentum=0.9)
            last_acc=self.model_train(optimizer, epoches=30, regular_step_size= 10,regular_gamma= 0.1)
            torch.save(self.resnet101, "./pruned/resnet101_cifar10_prunned_%d" % it_k)
            last_prune_params = sum(p.numel() for p in self.resnet101.parameters() if p.requires_grad)
            print("Parameters Compression ratio is %f." % (
            float(before_prune_params - last_prune_params) / before_prune_params))
            if last_acc <= before_prune_acc-0.06:
                break

        print("Finished. Going to fine tune the model a bit more")
        opt = optim.SGD(self.resnet101.parameters(), lr=0.01, momentum=0.9)
        last_acc = self.model_train(opt, epoches=40,regular_step_size= 20, regular_gamma=0.1)
        last_prune_params = sum(p.numel() for p in self.resnet101.parameters() if p.requires_grad)
        print("Max acc loss is %f. Parameters Compression ratio is %f." % ((before_prune_acc - last_acc), float(before_prune_params-last_prune_params)/before_prune_params))
        torch.save(self.resnet101, "resnet101_cifar10_prunned")

    def recover_from_prune(self):
        after_prune_params = sum(p.numel() for p in self.resnet101.parameters() if p.requires_grad)
        compress_ratio = 0.907765
        before_prune_params = after_prune_params / (1 - compress_ratio)
        before_prune_acc = 0.898100

        print("Finished. Going to fine tune the model a bit more")
        opt = optim.SGD(self.resnet101.parameters(), lr=0.01, momentum=0.9)
        last_acc = self.model_train(opt, epoches=40, regular_step_size=20, regular_gamma=0.1)

        last_prune_params = sum(p.numel() for p in self.resnet101.parameters() if p.requires_grad)

        print("Max acc loss is %f. Parameters Compression ratio is %f." % (
            (before_prune_acc - last_acc), float(before_prune_params - last_prune_params) / before_prune_params))
        torch.save(self.resnet101, "resnet50_cifar100_prunned")


class ResNetModel_101(torch.nn.Module):

    def __init__(self):
        super(ResNetModel_101, self).__init__()
        # Import pre-trained model
        resnet101 = models.resnet101(pretrained=True)

        self.inplanes = resnet101.inplanes
        self.conv1 = resnet101.conv1
        self.bn1 = resnet101.bn1
        self.relu = resnet101.relu
        self.maxpool = resnet101.maxpool
        self.layer1 = resnet101.layer1
        self.layer2 = resnet101.layer2
        self.layer3 = resnet101.layer3
        self.layer4 = resnet101.layer4
        self.avgpool = resnet101.avgpool

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool
        )
        # Get input dim of the last fv layer of resnet101 and replace the last layer with another
        # which has the same input dim while has the different output dim
        num_li = resnet101.fc.in_features
        # ***** modify the model here *****
        self.fc = nn.Linear(num_li, 10)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        #
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #
        # x = self.avgpool(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--recover", dest="recover", action="store_true")
    # parser.add_argument("--train_path", type=str, default="train")
    # parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--test_path", type=str, default="./data")
    parser.add_argument("--train_epoch", dest="train_epoch", action="store", type=int, default=20)
    parser.add_argument("--torch_version", dest="torch_version", action="store", type=float, default=0.4)
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    parser.set_defaults(recover=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args = get_args()
    if args.recover:
        savedStdout = sys.stdout
        with open('./recover_res101_cifar10.txt', 'w+', 1) as redirect_out:
            sys.stdout = redirect_out
            if torch.cuda.is_available():
                model = torch.load("resnet101_cifar10_prunned_14").cuda()
                use_gpu = True
            else:
                model = torch.load("resnet101_cifar10_prunned_14").cpu()
                use_gpu = False

            fine_tuner = PrunningFineTuner_ResNet50(args.train_path, args.test_path, model, use_gpu,
                                                    torch_version=args.torch_version)
            if args.recover:
                fine_tuner.recover_from_prune()
        sys.stdout = savedStdout
        print("Done!\n")
    else:
        savedStdout = sys.stdout
        with open('./out_res101_cifar10.txt', 'w+', 1) as redirect_out:
            sys.stdout = redirect_out
            # args = get_args()
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                if args.train:
                    model = ResNetModel_101().cuda()
                    if os.path.exists('./resnet101_params_cifar10.pth'):
                        model.load_state_dict(torch.load('./resnet101_params_cifar10.pth'))
                    # model = model.cuda()
                elif args.prune:
                    model = torch.load("resnet101_model_cifar10").cuda()
            else:
                if args.train:
                    model = ResNetModel_101()
                    if os.path.exists('./resnet101_params_cifar10.pth'):
                        model.load_state_dict(torch.load('./resnet101_params_cifar10.pth'))
                elif args.prune:
                    # python3本地版没有办法load服务器python2保存下来的model(如果只是参数model.pth那种好像可以,但是包括架构的model不行),会有UnicodeDecodeError
                    model = torch.load("resnet101_model_cifar10")

            fine_tuner = PrunningFineTuner_ResNet50(args.train_path, args.test_path, model, use_gpu,
                                                 torch_version=args.torch_version)
            if args.train:
                fine_tuner.model_train(optimizer=fine_tuner.optimizer, epoches=args.train_epoch)
                torch.save(model.state_dict(), './resnet101_params_cifar10.pth')
                torch.save(model, "resnet101_model_cifar10")
            elif args.prune:
                fine_tuner.prune()
        sys.stdout = savedStdout
        print("Done!\n")
