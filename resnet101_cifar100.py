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
        train_dataset = datasets.CIFAR100(root=path, train=True, download=True, transform=traindata_transforms)
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
        test_dataset = datasets.CIFAR100(root=path, train=False, download=True, transform=testdata_transforms)
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

    def lowest_ranking_filters(self, num,model_scale=None,struct_scale=None):
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
                torch.save(self.resnet101.state_dict(), './resnet101_params_cifar100.pth')

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

        total_l, model_scale = self.get_model_scales()
        struct_scale = self.get_structure_scales()

        number_of_filters = self.total_num_filters()
        # print(number_of_filters)

        num_filters_to_prune_per_iteration = 1024

        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        # iterations = int(iterations * 2.0 / 3)

        print("Max number of prunning iterations: %d " % iterations)

        # prune_targets_1 =[(0, 4), (0, 4), (0, 5), (0, 5), (0, 6), (0, 12), (0, 21), (0, 29), (0, 29), (0, 32), (0, 32), (0, 32),
        #           (0, 34), (0, 35), (0, 36), (0, 38), (0, 39), (0, 45), (4, 2), (4, 7), (4, 7), (4, 18), (4, 20),
        #           (4, 24), (4, 25), (4, 31), (4, 31), (4, 35), (4, 37), (4, 37), (4, 42), (4, 43), (4, 44), (4, 48),
        #           (7, 5), (7, 14), (7, 15), (7, 41), (10, 2), (10, 43), (10, 50), (10, 62), (10, 66), (10, 66),
        #           (10, 67), (10, 68), (10, 72), (10, 73), (10, 73), (10, 78), (10, 80), (10, 87), (10, 87), (10, 92),
        #           (10, 108), (10, 120), (10, 130), (10, 134), (10, 137), (10, 138), (10, 148), (10, 152), (10, 152),
        #           (10, 154), (10, 158), (10, 163), (10, 170), (10, 190), (10, 190), (10, 195), (10, 195), (10, 197),
        #           (10, 200), (10, 204), (10, 216), (10, 216), (13, 0), (13, 2), (13, 2), (13, 2), (13, 4), (13, 7),
        #           (13, 7), (13, 7), (13, 7), (13, 7), (13, 10), (13, 11), (13, 11), (13, 11), (13, 11), (13, 11),
        #           (13, 12), (13, 13), (13, 14), (13, 14), (13, 15), (13, 16), (13, 17), (13, 17), (13, 18), (13, 18),
        #           (13, 18), (13, 18), (13, 19), (13, 20), (13, 21), (13, 21), (13, 21), (13, 22), (13, 24), (16, 0),
        #           (16, 3), (16, 5), (16, 6), (16, 9), (16, 12), (16, 13), (16, 14), (16, 14), (16, 20), (16, 20),
        #           (16, 24), (16, 24), (16, 25), (16, 25), (16, 29), (16, 32), (16, 36), (16, 37), (16, 42), (19, 2),
        #           (19, 79), (19, 81), (19, 97), (19, 97), (19, 99), (19, 109), (19, 114), (19, 145), (19, 150),
        #           (19, 165), (19, 179), (19, 207), (19, 214), (19, 216), (19, 219), (19, 237), (22, 11), (22, 18),
        #           (22, 19), (22, 21), (22, 29), (22, 29), (22, 33), (22, 36), (22, 49), (22, 49), (25, 0), (25, 17),
        #           (25, 22), (25, 31), (25, 36), (25, 48), (25, 53), (28, 40), (28, 43), (31, 23), (31, 44), (31, 44),
        #           (31, 50), (31, 67), (31, 70), (31, 87), (31, 90), (31, 106), (34, 11), (34, 19), (34, 34), (34, 55),
        #           (34, 67), (34, 98), (37, 4), (37, 10), (37, 11), (37, 13), (37, 23), (37, 23), (37, 28), (37, 32),
        #           (37, 35), (37, 36), (37, 43), (37, 43), (37, 51), (37, 51), (37, 51), (37, 59), (37, 60), (37, 76),
        #           (37, 81), (37, 100), (37, 104), (37, 114), (37, 130), (37, 136), (37, 144), (37, 153), (37, 156),
        #           (37, 158), (37, 167), (37, 171), (37, 172), (37, 182), (37, 191), (37, 191), (37, 193), (37, 193),
        #           (37, 193), (37, 209), (37, 211), (37, 217), (37, 220), (37, 222), (37, 248), (37, 251), (37, 252),
        #           (37, 256), (37, 260), (37, 279), (37, 280), (37, 284), (37, 286), (37, 286), (37, 286), (37, 294),
        #           (37, 294), (37, 295), (37, 295), (37, 299), (37, 316), (37, 316), (37, 318), (37, 320), (37, 323),
        #           (37, 324), (37, 325), (37, 326), (37, 330), (37, 333), (37, 335), (37, 336), (37, 336), (37, 345),
        #           (37, 345), (37, 354), (37, 355), (37, 357), (37, 361), (37, 391), (37, 392), (37, 394), (37, 394),
        #           (37, 395), (37, 410), (37, 421), (37, 423), (37, 424), (40, 6), (40, 26), (40, 33), (40, 50),
        #           (40, 59), (40, 80), (40, 110), (43, 0), (43, 7), (43, 9), (43, 34), (43, 37), (43, 39), (43, 42),
        #           (43, 60), (43, 61), (43, 84), (43, 89), (43, 103), (43, 114), (46, 4), (46, 42), (46, 62), (46, 66),
        #           (46, 72), (46, 88), (46, 93), (46, 112), (46, 116), (46, 126), (46, 129), (46, 167), (46, 173),
        #           (46, 178), (46, 199), (46, 212), (46, 218), (46, 239), (46, 245), (46, 262), (46, 270), (46, 285),
        #           (46, 311), (46, 314), (46, 314), (46, 322), (46, 348), (46, 354), (46, 357), (46, 371), (46, 375),
        #           (46, 398), (46, 425), (46, 441), (46, 459), (46, 472), (49, 9), (49, 11), (49, 14), (49, 18),
        #           (49, 63), (49, 68), (49, 69), (49, 74), (49, 78), (49, 81), (49, 93), (49, 98), (49, 107), (49, 108),
        #           (49, 111), (52, 3), (52, 13), (52, 38), (52, 44), (52, 53), (52, 64), (52, 67), (52, 67), (52, 71),
        #           (52, 81), (52, 81), (52, 95), (52, 115), (55, 64), (55, 118), (55, 167), (55, 179), (55, 223),
        #           (55, 235), (55, 284), (55, 330), (55, 341), (55, 376), (55, 390), (55, 394), (55, 417), (55, 419),
        #           (55, 443), (55, 459), (55, 491), (58, 4), (58, 14), (58, 17), (58, 29), (58, 30), (58, 33), (58, 62),
        #           (58, 65), (58, 75), (58, 77), (58, 101), (58, 102), (61, 23), (61, 34), (61, 69), (61, 84), (61, 93),
        #           (61, 99), (61, 113), (64, 216), (64, 289), (64, 472), (67, 1), (67, 18), (67, 55), (67, 61), (67, 64),
        #           (67, 68), (67, 68), (67, 85), (67, 92), (67, 101), (67, 106), (67, 116), (67, 151), (67, 167),
        #           (67, 195), (67, 201), (67, 201), (67, 224), (70, 13), (70, 23), (70, 24), (70, 25), (70, 27),
        #           (70, 30), (70, 43), (70, 53), (70, 67), (70, 77), (70, 159), (70, 170), (70, 174), (70, 183),
        #           (70, 191), (70, 193), (70, 212), (73, 0), (73, 12), (73, 14), (73, 27), (73, 30), (73, 47), (73, 57),
        #           (73, 63), (73, 73), (73, 81), (73, 92), (73, 128), (73, 140), (73, 147), (73, 159), (73, 159),
        #           (73, 168), (73, 168), (73, 177), (73, 186), (73, 186), (73, 193), (73, 199), (73, 210), (73, 217),
        #           (73, 222), (73, 228), (73, 231), (73, 264), (73, 291), (73, 306), (73, 329), (73, 350), (73, 369),
        #           (73, 375), (73, 381), (73, 429), (73, 440), (73, 454), (73, 455), (73, 469), (73, 474), (73, 483),
        #           (73, 491), (73, 494), (73, 498), (73, 515), (73, 515), (73, 519), (73, 525), (73, 531), (73, 538),
        #           (73, 557), (73, 570), (73, 574), (73, 588), (73, 597), (73, 612), (73, 617), (73, 631), (73, 652),
        #           (73, 669), (73, 670), (73, 671), (73, 690), (73, 691), (73, 691), (73, 695), (73, 699), (73, 700),
        #           (73, 719), (73, 727), (73, 742), (73, 746), (73, 747), (73, 747), (73, 757), (73, 757), (73, 768),
        #           (73, 785), (73, 805), (73, 837), (73, 845), (73, 850), (73, 853), (73, 854)]
        # prune_targets_2 = [(0, 18), (130, 21), (130, 24), (130, 26), (130, 43), (130, 59), (130, 65), (130, 70),
        #                  (130, 114),
        #                  (130, 127), (130, 179), (130, 186), (130, 192), (130, 194), (130, 211), (130, 211), (130, 217),
        #                  (130, 219), (130, 222), (130, 233), (133, 156), (133, 178), (133, 216), (133, 230), (7, 5),
        #                  (139, 3), (139, 12), (139, 43), (139, 69), (139, 101), (139, 109), (139, 121), (139, 129),
        #                  (139, 134), (139, 134), (139, 138), (139, 141), (139, 164), (139, 179), (139, 200), (139, 201),
        #                  (139, 203), (139, 205), (139, 228), (139, 230), (139, 231), (142, 49), (142, 68), (142, 77),
        #                  (142, 80), (142, 95), (142, 148), (142, 200), (142, 209), (142, 225), (16, 26), (19, 160),
        #                  (19, 187), (19, 207), (148, 40), (148, 49), (148, 136), (148, 185), (148, 228), (22, 16),
        #                  (22, 19), (22, 26), (22, 36), (22, 42), (151, 22), (151, 38), (151, 46), (151, 104),
        #                  (151, 108),
        #                  (151, 125), (151, 192), (151, 224), (25, 1), (25, 22), (25, 28), (25, 48), (25, 49), (25, 49),
        #                  (28, 35), (28, 42), (28, 55), (28, 62), (28, 93), (28, 171), (28, 194), (157, 1), (157, 21),
        #                  (157, 32), (157, 54), (157, 81), (157, 91), (157, 108), (157, 153), (157, 187), (157, 187),
        #                  (31, 23), (31, 69), (31, 84), (160, 18), (160, 25), (160, 76), (160, 157), (160, 205),
        #                  (160, 235),
        #                  (34, 35), (34, 39), (34, 87), (37, 5), (37, 15), (37, 31), (37, 43), (37, 46), (37, 201),
        #                  (37, 213), (37, 216), (37, 231), (37, 231), (37, 251), (37, 263), (37, 281), (37, 302),
        #                  (37, 302),
        #                  (37, 320), (37, 332), (37, 362), (37, 383), (37, 384), (37, 384), (166, 0), (166, 1),
        #                  (166, 19),
        #                  (166, 130), (166, 180), (166, 184), (166, 198), (166, 202), (40, 41), (40, 49), (40, 72),
        #                  (40, 73), (40, 101), (169, 137), (169, 153), (43, 4), (43, 20), (43, 35), (43, 39), (43, 50),
        #                  (43, 88), (43, 104), (43, 105), (46, 217), (46, 227), (175, 60), (175, 63), (175, 69),
        #                  (175, 76),
        #                  (175, 81), (175, 95), (175, 119), (175, 171), (175, 179), (175, 199), (175, 237), (175, 244),
        #                  (49, 9), (49, 14), (49, 54), (49, 54), (49, 84), (178, 21), (178, 30), (178, 45), (178, 81),
        #                  (178, 94), (178, 118), (178, 136), (178, 137), (178, 156), (178, 181), (178, 197), (178, 212),
        #                  (178, 230), (178, 236), (52, 3), (52, 15), (52, 27), (52, 35), (52, 70), (52, 91), (52, 91),
        #                  (52, 103), (52, 104), (52, 106), (55, 255), (55, 259), (184, 24), (184, 38), (184, 40),
        #                  (184, 100), (184, 134), (184, 182), (184, 227), (58, 62), (187, 12), (187, 29), (187, 37),
        #                  (187, 45), (187, 150), (187, 197), (187, 239), (61, 1), (61, 9), (61, 69), (61, 97), (64, 72),
        #                  (64, 249), (64, 281), (193, 215), (67, 8), (67, 23), (67, 43), (67, 55), (67, 99), (67, 124),
        #                  (67, 140), (67, 146), (67, 186), (67, 215), (67, 225), (196, 22), (196, 64), (196, 200),
        #                  (196, 240), (70, 35), (70, 137), (70, 157), (70, 168), (73, 3), (73, 13), (73, 30), (73, 66),
        #                  (73, 122), (73, 156), (73, 196), (73, 206), (73, 231), (73, 331), (73, 343), (73, 352),
        #                  (73, 470),
        #                  (73, 559), (73, 568), (73, 797), (73, 849), (73, 876), (202, 25), (76, 17), (76, 19), (76, 21),
        #                  (76, 33), (76, 34), (76, 36), (76, 37), (76, 45), (76, 47), (76, 54), (76, 65), (76, 77),
        #                  (76, 98), (76, 101), (76, 110), (76, 113), (76, 122), (76, 138), (76, 138), (76, 142),
        #                  (76, 154),
        #                  (76, 158), (76, 164), (76, 172), (76, 173), (76, 173), (76, 178), (76, 183), (76, 185),
        #                  (76, 198),
        #                  (76, 203), (76, 208), (76, 210), (79, 0), (79, 4), (79, 5), (79, 12), (79, 12), (79, 14),
        #                  (79, 17), (79, 38), (79, 38), (79, 45), (79, 47), (79, 49), (79, 54), (79, 54), (79, 66),
        #                  (79, 68), (79, 82), (79, 82), (79, 84), (79, 85), (79, 90), (79, 96), (79, 104), (79, 108),
        #                  (79, 128), (79, 128), (79, 135), (79, 140), (79, 162), (79, 165), (79, 166), (79, 166),
        #                  (79, 167),
        #                  (79, 203), (79, 204), (79, 204), (79, 207), (79, 214), (82, 82), (82, 119), (82, 151),
        #                  (82, 382),
        #                  (82, 789), (82, 834), (85, 14), (85, 31), (85, 56), (85, 80), (85, 95), (85, 106), (85, 108),
        #                  (85, 125), (85, 146), (85, 149), (85, 174), (85, 183), (85, 183), (85, 183), (85, 193),
        #                  (85, 215),
        #                  (85, 238), (85, 238), (88, 3), (88, 8), (88, 44), (88, 71), (88, 84), (88, 85), (88, 89),
        #                  (88, 107), (88, 116), (88, 116), (88, 127), (88, 160), (88, 170), (91, 120), (91, 956),
        #                  (94, 6),
        #                  (94, 11), (94, 12), (94, 28), (94, 36), (94, 54), (94, 99), (94, 100), (94, 129), (94, 143),
        #                  (94, 162), (94, 210), (94, 217), (94, 218), (94, 219), (94, 221), (97, 18), (97, 39), (97, 73),
        #                  (97, 84), (97, 122), (97, 122), (97, 130), (97, 143), (97, 145), (97, 153), (97, 163),
        #                  (97, 171),
        #                  (97, 171), (97, 177), (97, 189), (97, 192), (97, 193), (97, 197), (97, 209), (97, 221),
        #                  (97, 222),
        #                  (100, 387), (100, 956), (103, 2), (103, 28), (103, 48), (103, 56), (103, 57), (103, 76),
        #                  (103, 85), (103, 94), (103, 110), (103, 119), (103, 138), (103, 159), (103, 182), (103, 196),
        #                  (103, 203), (106, 34), (106, 39), (106, 47), (106, 53), (106, 55), (106, 63), (106, 86),
        #                  (106, 86), (106, 90), (106, 94), (106, 96), (106, 100), (106, 100), (106, 110), (106, 112),
        #                  (106, 176), (106, 182), (106, 184), (106, 186), (106, 190), (106, 199), (106, 211), (106, 233),
        #                  (109, 20), (109, 386), (112, 2), (112, 18), (112, 31), (112, 44), (112, 53), (112, 56),
        #                  (112, 59),
        #                  (112, 60), (112, 80), (112, 82), (112, 92), (112, 123), (112, 128), (112, 146), (112, 146),
        #                  (112, 154), (112, 159), (112, 178), (112, 183), (112, 185), (112, 198), (112, 215), (115, 17),
        #                  (115, 19), (115, 20), (115, 25), (115, 34), (115, 47), (115, 71), (115, 72), (115, 75),
        #                  (115, 88),
        #                  (115, 89), (115, 109), (115, 122), (115, 123), (115, 144), (115, 147), (115, 149), (115, 149),
        #                  (115, 152), (115, 155), (115, 175), (115, 212), (118, 387), (121, 17), (121, 54), (121, 70),
        #                  (121, 97), (121, 114), (121, 114), (121, 161), (121, 195), (121, 200), (124, 34), (124, 49),
        #                  (124, 66), (124, 112), (124, 142), (124, 227), (127, 920)]
        # prune_t = [prune_targets_1,prune_targets_2]
        # prune_2048=[(0, 4), (0, 4), (0, 5), (0, 5), (0, 6), (0, 12), (0, 18), (0, 20), (0, 23), (0, 27), (0, 27), (0, 30), (0, 30),
        #  (0, 30), (0, 32), (0, 33), (0, 34), (0, 36), (0, 37), (0, 43), (4, 2), (4, 7), (4, 7), (4, 7), (4, 17),
        #  (4, 19), (4, 23), (4, 24), (4, 29), (4, 29), (4, 29), (4, 33), (4, 35), (4, 35), (4, 40), (4, 41), (4, 42),
        #  (4, 46), (7, 5), (7, 14), (7, 15), (7, 41), (10, 2), (10, 43), (10, 50), (10, 62), (10, 66), (10, 66),
        #  (10, 67), (10, 68), (10, 72), (10, 73), (10, 73), (10, 78), (10, 80), (10, 87), (10, 87), (10, 89), (10, 91),
        #  (10, 98), (10, 103), (10, 105), (10, 117), (10, 127), (10, 131), (10, 134), (10, 135), (10, 145), (10, 149),
        #  (10, 149), (10, 151), (10, 155), (10, 160), (10, 166), (10, 166), (10, 186), (10, 186), (10, 191), (10, 191),
        #  (10, 193), (10, 196), (10, 200), (10, 212), (10, 212), (13, 0), (13, 2), (13, 2), (13, 2), (13, 4), (13, 7),
        #  (13, 7), (13, 7), (13, 7), (13, 7), (13, 10), (13, 11), (13, 11), (13, 11), (13, 11), (13, 11), (13, 12),
        #  (13, 13), (13, 14), (13, 14), (13, 15), (13, 16), (13, 17), (13, 17), (13, 18), (13, 18), (13, 18), (13, 18),
        #  (13, 19), (13, 19), (13, 19), (13, 20), (13, 20), (13, 20), (13, 21), (13, 23), (16, 0), (16, 3), (16, 5),
        #  (16, 6), (16, 9), (16, 12), (16, 13), (16, 14), (16, 14), (16, 20), (16, 20), (16, 24), (16, 24), (16, 25),
        #  (16, 25), (16, 26), (16, 28), (16, 31), (16, 32), (16, 34), (16, 35), (16, 40), (19, 2), (19, 5), (19, 71),
        #  (19, 72), (19, 76), (19, 78), (19, 82), (19, 93), (19, 93), (19, 95), (19, 105), (19, 108), (19, 109),
        #  (19, 124), (19, 139), (19, 144), (19, 159), (19, 173), (19, 196), (19, 199), (19, 199), (19, 201), (19, 205),
        #  (19, 207), (19, 210), (19, 228), (22, 11), (22, 18), (22, 19), (22, 21), (22, 29), (22, 29), (22, 33),
        #  (22, 36), (22, 49), (22, 49), (25, 0), (25, 17), (25, 17), (25, 21), (25, 22), (25, 29), (25, 34), (25, 42),
        #  (25, 45), (28, 0), (28, 0), (28, 38), (28, 88), (28, 217), (31, 23), (31, 39), (31, 43), (31, 43), (31, 49),
        #  (31, 66), (31, 69), (31, 86), (31, 89), (31, 96), (31, 103), (31, 103), (34, 11), (34, 19), (34, 34), (34, 40),
        #  (34, 54), (34, 66), (34, 88), (34, 96), (37, 4), (37, 10), (37, 11), (37, 13), (37, 23), (37, 23), (37, 28),
        #  (37, 32), (37, 35), (37, 36), (37, 43), (37, 43), (37, 51), (37, 51), (37, 51), (37, 59), (37, 60), (37, 65),
        #  (37, 75), (37, 80), (37, 99), (37, 103), (37, 107), (37, 112), (37, 128), (37, 134), (37, 142), (37, 151),
        #  (37, 154), (37, 156), (37, 165), (37, 169), (37, 170), (37, 180), (37, 189), (37, 189), (37, 191), (37, 191),
        #  (37, 191), (37, 201), (37, 206), (37, 208), (37, 214), (37, 217), (37, 219), (37, 236), (37, 244), (37, 247),
        #  (37, 248), (37, 252), (37, 256), (37, 275), (37, 276), (37, 280), (37, 282), (37, 282), (37, 282), (37, 290),
        #  (37, 290), (37, 290), (37, 290), (37, 290), (37, 294), (37, 311), (37, 311), (37, 313), (37, 315), (37, 318),
        #  (37, 319), (37, 320), (37, 321), (37, 325), (37, 328), (37, 330), (37, 331), (37, 331), (37, 340), (37, 340),
        #  (37, 342), (37, 348), (37, 349), (37, 351), (37, 355), (37, 385), (37, 386), (37, 388), (37, 388), (37, 389),
        #  (37, 394), (37, 400), (37, 402), (37, 402), (37, 412), (37, 414), (37, 415), (40, 6), (40, 21), (40, 25),
        #  (40, 32), (40, 49), (40, 58), (40, 79), (40, 91), (40, 108), (40, 111), (43, 0), (43, 7), (43, 9), (43, 9),
        #  (43, 33), (43, 36), (43, 38), (43, 41), (43, 59), (43, 60), (43, 83), (43, 88), (43, 102), (43, 106),
        #  (43, 112), (46, 4), (46, 42), (46, 62), (46, 66), (46, 72), (46, 78), (46, 87), (46, 92), (46, 111), (46, 115),
        #  (46, 125), (46, 128), (46, 157), (46, 165), (46, 171), (46, 176), (46, 197), (46, 209), (46, 209), (46, 215),
        #  (46, 220), (46, 235), (46, 239), (46, 240), (46, 257), (46, 265), (46, 274), (46, 279), (46, 305), (46, 308),
        #  (46, 308), (46, 316), (46, 317), (46, 341), (46, 347), (46, 350), (46, 364), (46, 368), (46, 378), (46, 388),
        #  (46, 389), (46, 414), (46, 415), (46, 431), (46, 447), (46, 448), (46, 461), (49, 9), (49, 11), (49, 14),
        #  (49, 18), (49, 44), (49, 62), (49, 67), (49, 68), (49, 73), (49, 77), (49, 80), (49, 92), (49, 97), (49, 106),
        #  (49, 107), (49, 110), (52, 3), (52, 13), (52, 38), (52, 44), (52, 48), (52, 52), (52, 56), (52, 62), (52, 65),
        #  (52, 65), (52, 69), (52, 79), (52, 79), (52, 93), (52, 113), (55, 10), (55, 43), (55, 62), (55, 64), (55, 106),
        #  (55, 114), (55, 143), (55, 162), (55, 174), (55, 218), (55, 230), (55, 237), (55, 278), (55, 324), (55, 335),
        #  (55, 341), (55, 364), (55, 368), (55, 382), (55, 386), (55, 397), (55, 408), (55, 410), (55, 432), (55, 433),
        #  (55, 449), (55, 450), (55, 466), (55, 479), (58, 4), (58, 14), (58, 17), (58, 29), (58, 30), (58, 33),
        #  (58, 62), (58, 65), (58, 75), (58, 77), (58, 100), (58, 100), (58, 101), (61, 23), (61, 34), (61, 34),
        #  (61, 68), (61, 83), (61, 88), (61, 91), (61, 94), (61, 96), (61, 110), (64, 67), (64, 71), (64, 171),
        #  (64, 213), (64, 286), (64, 368), (64, 398), (64, 445), (64, 450), (64, 465), (67, 1), (67, 9), (67, 17),
        #  (67, 54), (67, 57), (67, 59), (67, 62), (67, 63), (67, 65), (67, 65), (67, 82), (67, 89), (67, 98), (67, 103),
        #  (67, 113), (67, 148), (67, 164), (67, 192), (67, 198), (67, 198), (67, 208), (67, 228), (70, 13), (70, 13),
        #  (70, 22), (70, 23), (70, 24), (70, 26), (70, 29), (70, 42), (70, 52), (70, 66), (70, 68), (70, 71), (70, 74),
        #  (70, 89), (70, 97), (70, 154), (70, 165), (70, 169), (70, 178), (70, 182), (70, 185), (70, 187), (70, 206),
        #  (73, 0), (73, 12), (73, 14), (73, 21), (73, 26), (73, 27), (73, 28), (73, 30), (73, 38), (73, 40), (73, 42),
        #  (73, 51), (73, 51), (73, 57), (73, 58), (73, 62), (73, 65), (73, 73), (73, 77), (73, 83), (73, 100), (73, 105),
        #  (73, 115), (73, 116), (73, 128), (73, 135), (73, 147), (73, 147), (73, 156), (73, 156), (73, 161), (73, 164),
        #  (73, 173), (73, 173), (73, 176), (73, 179), (73, 181), (73, 182), (73, 183), (73, 193), (73, 193), (73, 196),
        #  (73, 199), (73, 201), (73, 203), (73, 204), (73, 208), (73, 211), (73, 219), (73, 224), (73, 236), (73, 237),
        #  (73, 240), (73, 264), (73, 265), (73, 265), (73, 268), (73, 273), (73, 278), (73, 301), (73, 312), (73, 321),
        #  (73, 323), (73, 323), (73, 332), (73, 337), (73, 343), (73, 349), (73, 361), (73, 374), (73, 395), (73, 406),
        #  (73, 420), (73, 421), (73, 435), (73, 440), (73, 448), (73, 448), (73, 456), (73, 459), (73, 460), (73, 462),
        #  (73, 464), (73, 465), (73, 477), (73, 477), (73, 481), (73, 487), (73, 493), (73, 494), (73, 499), (73, 499),
        #  (73, 506), (73, 510), (73, 510), (73, 514), (73, 527), (73, 527), (73, 530), (73, 544), (73, 546), (73, 552),
        #  (73, 567), (73, 568), (73, 571), (73, 572), (73, 581), (73, 583), (73, 589), (73, 591), (73, 595), (73, 601),
        #  (73, 618), (73, 619), (73, 620), (73, 639), (73, 639), (73, 639), (73, 639), (73, 643), (73, 647), (73, 648),
        #  (73, 667), (73, 675), (73, 686), (73, 689), (73, 693), (73, 694), (73, 694), (73, 704), (73, 704), (73, 712),
        #  (73, 713), (73, 713), (73, 715), (73, 729), (73, 742), (73, 748), (73, 751), (73, 753), (73, 759), (73, 777),
        #  (73, 785), (73, 790), (73, 793), (73, 794), (73, 800), (73, 804), (73, 809), (73, 809), (73, 817), (73, 852),
        #  (73, 856), (73, 857), (73, 862), (76, 0), (76, 6), (76, 21), (76, 29), (76, 34), (76, 36), (76, 46), (76, 48),
        #  (76, 55), (76, 62), (76, 65), (76, 77), (76, 98), (76, 111), (76, 122), (76, 123), (76, 127), (76, 138),
        #  (76, 138), (76, 142), (76, 150), (76, 153), (76, 157), (76, 163), (76, 170), (76, 170), (76, 171), (76, 171),
        #  (76, 172), (76, 179), (76, 180), (76, 182), (76, 198), (76, 200), (76, 205), (76, 207), (76, 208), (79, 0),
        #  (79, 4), (79, 5), (79, 12), (79, 16), (79, 18), (79, 27), (79, 38), (79, 52), (79, 52), (79, 56), (79, 69),
        #  (79, 81), (79, 82), (79, 90), (79, 95), (79, 110), (79, 110), (79, 110), (79, 133), (79, 133), (79, 140),
        #  (79, 145), (79, 159), (79, 159), (79, 171), (79, 171), (79, 204), (79, 207), (79, 209), (79, 209), (79, 210),
        #  (79, 211), (79, 218), (79, 221), (82, 8), (82, 17), (82, 28), (82, 29), (82, 30), (82, 36), (82, 40), (82, 45),
        #  (82, 73), (82, 86), (82, 92), (82, 115), (82, 122), (82, 126), (82, 146), (82, 159), (82, 168), (82, 168),
        #  (82, 173), (82, 176), (82, 185), (82, 185), (82, 188), (82, 191), (82, 195), (82, 196), (82, 215), (82, 227),
        #  (82, 260), (82, 263), (82, 280), (82, 280), (82, 288), (82, 303), (82, 326), (82, 331), (82, 346), (82, 349),
        #  (82, 371), (82, 377), (82, 389), (82, 402), (82, 402), (82, 434), (82, 448), (82, 479), (82, 482), (82, 487),
        #  (82, 490), (82, 491), (82, 493), (82, 497), (82, 509), (82, 509), (82, 513), (82, 517), (82, 525), (82, 526),
        #  (82, 532), (82, 544), (82, 564), (82, 567), (82, 589), (82, 613), (82, 614), (82, 625), (82, 631), (82, 633),
        #  (82, 645), (82, 645), (82, 650), (82, 661), (82, 662), (82, 677), (82, 680), (82, 680), (82, 680), (82, 685),
        #  (82, 689), (82, 710), (82, 718), (82, 729), (82, 732), (82, 734), (82, 735), (82, 736), (82, 747), (82, 756),
        #  (82, 776), (82, 779), (82, 799), (82, 808), (82, 847), (82, 853), (82, 864), (82, 900), (82, 912), (82, 913),
        #  (82, 918), (85, 2), (85, 13), (85, 30), (85, 41), (85, 71), (85, 78), (85, 80), (85, 92), (85, 96), (85, 102),
        #  (85, 121), (85, 121), (85, 142), (85, 142), (85, 144), (85, 150), (85, 151), (85, 156), (85, 177), (85, 182),
        #  (85, 182), (85, 186), (85, 208), (85, 213), (85, 224), (85, 229), (85, 229), (88, 3), (88, 8), (88, 9),
        #  (88, 54), (88, 62), (88, 69), (88, 77), (88, 81), (88, 82), (88, 86), (88, 90), (88, 90), (88, 95), (88, 101),
        #  (88, 104), (88, 106), (88, 108), (88, 108), (88, 119), (88, 126), (88, 151), (88, 155), (88, 158), (88, 159),
        #  (88, 175), (88, 214), (91, 30), (91, 30), (91, 44), (91, 49), (91, 85), (91, 90), (91, 96), (91, 98),
        #  (91, 104), (91, 111), (91, 116), (91, 128), (91, 130), (91, 149), (91, 160), (91, 176), (91, 189), (91, 189),
        #  (91, 196), (91, 200), (91, 201), (91, 233), (91, 234), (91, 287), (91, 312), (91, 318), (91, 356), (91, 359),
        #  (91, 381), (91, 414), (91, 447), (91, 461), (91, 477), (91, 491), (91, 496), (91, 499), (91, 502), (91, 503),
        #  (91, 505), (91, 509), (91, 532), (91, 540), (91, 541), (91, 546), (91, 559), (91, 579), (91, 605), (91, 643),
        #  (91, 649), (91, 683), (91, 701), (91, 705), (91, 746), (91, 761), (91, 765), (91, 766), (91, 773), (91, 786),
        #  (91, 888), (91, 898), (91, 898), (91, 944), (91, 948), (91, 953), (94, 12), (94, 13), (94, 29), (94, 37),
        #  (94, 55), (94, 100), (94, 101), (94, 119), (94, 129), (94, 143), (94, 162), (94, 170), (94, 209), (94, 216),
        #  (94, 219), (94, 221), (97, 3), (97, 17), (97, 17), (97, 33), (97, 36), (97, 70), (97, 120), (97, 120),
        #  (97, 128), (97, 141), (97, 143), (97, 151), (97, 161), (97, 169), (97, 176), (97, 177), (97, 179), (97, 186),
        #  (97, 189), (97, 190), (97, 194), (97, 206), (97, 218), (97, 219), (100, 46), (100, 51), (100, 93), (100, 100),
        #  (100, 111), (100, 121), (100, 131), (100, 132), (100, 156), (100, 164), (100, 181), (100, 189), (100, 193),
        #  (100, 193), (100, 200), (100, 204), (100, 238), (100, 264), (100, 305), (100, 359), (100, 362), (100, 365),
        #  (100, 387), (100, 391), (100, 419), (100, 425), (100, 451), (100, 465), (100, 496), (100, 505), (100, 508),
        #  (100, 512), (100, 513), (100, 515), (100, 543), (100, 546), (100, 547), (100, 566), (100, 586), (100, 612),
        #  (100, 621), (100, 649), (100, 663), (100, 689), (100, 707), (100, 718), (100, 720), (100, 751), (100, 763),
        #  (100, 765), (100, 769), (100, 770), (100, 791), (100, 815), (100, 892), (100, 903), (100, 939), (100, 953),
        #  (100, 958), (100, 962), (103, 2), (103, 4), (103, 4), (103, 21), (103, 25), (103, 45), (103, 53), (103, 54),
        #  (103, 67), (103, 72), (103, 74), (103, 80), (103, 89), (103, 89), (103, 94), (103, 103), (103, 112),
        #  (103, 116), (103, 125), (103, 129), (103, 150), (103, 173), (103, 187), (103, 194), (103, 203), (103, 217),
        #  (103, 226), (106, 26), (106, 33), (106, 34), (106, 36), (106, 36), (106, 44), (106, 50), (106, 52), (106, 60),
        #  (106, 83), (106, 83), (106, 87), (106, 91), (106, 93), (106, 97), (106, 97), (106, 107), (106, 109),
        #  (106, 109), (106, 138), (106, 162), (106, 177), (106, 179), (106, 181), (106, 185), (106, 194), (106, 210),
        #  (109, 46), (109, 51), (109, 87), (109, 92), (109, 122), (109, 131), (109, 131), (109, 136), (109, 147),
        #  (109, 148), (109, 181), (109, 194), (109, 195), (109, 206), (109, 267), (109, 337), (109, 370), (109, 392),
        #  (109, 425), (109, 446), (109, 457), (109, 471), (109, 502), (109, 515), (109, 521), (109, 523), (109, 555),
        #  (109, 556), (109, 575), (109, 576), (109, 592), (109, 593), (109, 619), (109, 647), (109, 663), (109, 692),
        #  (109, 700), (109, 714), (109, 718), (109, 739), (109, 759), (109, 769), (109, 772), (109, 776), (109, 777),
        #  (109, 798), (109, 820), (109, 866), (109, 898), (109, 909), (109, 960), (109, 965), (112, 2), (112, 7),
        #  (112, 12), (112, 12), (112, 15), (112, 20), (112, 27), (112, 40), (112, 49), (112, 52), (112, 55), (112, 56),
        #  (112, 79), (112, 80), (112, 88), (112, 116), (112, 118), (112, 123), (112, 141), (112, 141), (112, 149),
        #  (112, 154), (112, 159), (112, 172), (112, 177), (112, 193), (112, 210), (115, 17), (115, 19), (115, 20),
        #  (115, 25), (115, 27), (115, 33), (115, 38), (115, 45), (115, 61), (115, 68), (115, 69), (115, 72), (115, 72),
        #  (115, 86), (115, 98), (115, 99), (115, 104), (115, 117), (115, 118), (115, 136), (115, 138), (115, 141),
        #  (115, 143), (115, 143), (115, 146), (115, 149), (115, 169), (115, 177), (115, 189), (115, 204), (115, 205),
        #  (115, 208), (118, 46), (118, 51), (118, 93), (118, 123), (118, 135), (118, 186), (118, 193), (118, 198),
        #  (118, 211), (118, 212), (118, 244), (118, 252), (118, 268), (118, 268), (118, 280), (118, 371), (118, 393),
        #  (118, 426), (118, 437), (118, 458), (118, 487), (118, 503), (118, 512), (118, 515), (118, 518), (118, 520),
        #  (118, 522), (118, 529), (118, 555), (118, 563), (118, 566), (118, 572), (118, 590), (118, 591), (118, 614),
        #  (118, 616), (118, 715), (118, 719), (118, 776), (118, 780), (118, 781), (118, 785), (118, 801), (118, 903),
        #  (118, 914), (118, 928), (118, 964), (118, 969), (121, 10), (121, 16), (121, 53), (121, 60), (121, 68),
        #  (121, 82), (121, 94), (121, 98), (121, 110), (121, 110), (121, 157), (121, 161), (121, 190), (121, 195),
        #  (124, 33), (124, 33), (124, 48), (124, 65), (124, 111), (124, 141), (124, 226), (127, 4), (127, 45), (127, 65),
        #  (127, 92), (127, 119), (127, 121), (127, 133), (127, 139), (127, 183), (127, 196), (127, 209), (127, 210),
        #  (127, 269), (127, 311), (127, 333), (127, 357), (127, 370), (127, 375), (127, 391), (127, 424), (127, 435),
        #  (127, 456), (127, 502), (127, 511), (127, 514), (127, 520), (127, 522), (127, 529), (127, 530), (127, 554),
        #  (127, 573), (127, 574), (127, 592), (127, 618), (127, 628), (127, 636), (127, 685), (127, 690), (127, 718),
        #  (127, 732), (127, 774), (127, 801), (127, 801), (127, 875), (127, 902), (127, 909), (127, 912), (127, 931),
        #  (127, 962), (127, 967), (130, 14), (130, 20), (130, 23), (130, 25), (130, 27), (130, 41), (130, 49), (130, 56),
        #  (130, 62), (130, 71), (130, 94), (130, 103), (130, 109), (130, 111), (130, 121), (130, 128), (130, 165),
        #  (130, 165), (130, 170), (130, 177), (130, 183), (130, 185), (130, 202), (130, 202), (130, 208), (130, 210),
        #  (130, 213), (130, 224), (130, 225), (133, 36), (133, 70), (133, 86), (133, 107), (133, 152), (133, 174),
        #  (133, 181), (133, 211), (133, 225), (136, 4), (136, 45), (136, 53), (136, 91), (136, 91), (136, 134),
        #  (136, 136), (136, 184), (136, 191), (136, 196), (136, 209), (136, 222), (136, 347), (136, 374), (136, 395),
        #  (136, 395), (136, 427), (136, 507), (136, 520), (136, 526), (136, 528), (136, 538), (136, 561), (136, 580),
        #  (136, 594), (136, 597), (136, 598), (136, 601), (136, 623), (136, 727), (136, 783), (136, 783), (136, 811),
        #  (136, 913), (136, 924), (139, 13), (139, 44), (139, 68), (139, 102), (139, 110), (139, 119), (139, 120),
        #  (139, 120), (139, 128), (139, 133), (139, 138), (139, 141), (139, 164), (139, 179), (139, 200), (139, 201),
        #  (139, 206), (139, 227), (139, 228), (139, 230), (139, 231), (142, 13), (142, 48), (142, 67), (142, 96),
        #  (142, 104), (142, 104), (142, 143), (142, 146), (142, 155), (142, 166), (142, 172), (142, 182), (142, 194),
        #  (142, 200), (142, 202), (142, 215), (142, 217), (142, 222), (142, 224), (145, 46), (145, 55), (145, 55),
        #  (145, 92), (145, 99), (145, 134), (145, 140), (145, 142), (145, 151), (145, 196), (145, 209), (145, 210),
        #  (145, 213), (145, 236), (145, 297), (145, 297), (145, 308), (145, 341), (145, 349), (145, 390), (145, 423),
        #  (145, 456), (145, 459), (145, 482), (145, 485), (145, 499), (145, 512), (145, 518), (145, 520), (145, 527),
        #  (145, 537), (145, 552), (145, 571), (145, 589), (145, 590), (145, 593), (145, 603), (145, 614), (145, 696),
        #  (145, 717), (145, 771), (145, 802), (145, 848), (145, 884), (145, 902), (145, 913), (145, 917), (145, 960),
        #  (148, 18), (148, 26), (148, 38), (148, 47), (148, 134), (148, 173), (148, 182), (148, 208), (148, 224),
        #  (151, 22), (151, 38), (151, 46), (151, 104), (151, 108), (151, 125), (151, 192), (151, 224), (154, 106),
        #  (154, 204), (154, 219), (154, 253), (154, 272), (154, 375), (154, 381), (154, 388), (154, 402), (154, 457),
        #  (154, 480), (154, 498), (154, 506), (154, 525), (154, 600), (154, 655), (154, 682), (154, 709), (154, 709),
        #  (154, 713), (154, 714), (154, 735), (154, 792), (154, 882), (154, 934), (154, 938), (154, 980), (157, 22),
        #  (157, 33), (157, 55), (157, 78), (157, 81), (157, 91), (157, 108), (157, 117), (157, 126), (157, 149),
        #  (157, 150), (157, 171), (157, 183), (157, 183), (157, 195), (157, 197), (160, 18), (160, 25), (160, 76),
        #  (160, 83), (160, 106), (160, 155), (160, 203), (160, 233), (163, 75), (163, 144), (163, 203), (163, 218),
        #  (163, 243), (163, 247), (163, 318), (163, 380), (163, 384), (163, 444), (163, 499), (163, 527), (163, 650),
        #  (163, 719), (163, 720), (163, 734), (163, 761), (163, 888), (163, 940), (163, 981), (166, 0), (166, 1),
        #  (166, 19), (166, 54), (166, 104), (166, 128), (166, 129), (166, 177), (166, 178), (166, 180), (166, 194),
        #  (166, 198), (169, 42), (169, 91), (169, 135), (169, 151), (172, 24), (172, 52), (172, 100), (172, 236),
        #  (172, 266), (172, 340), (172, 351), (172, 380), (172, 384), (172, 490), (172, 498), (172, 498), (172, 502),
        #  (172, 525), (172, 554), (172, 580), (172, 592), (172, 602), (172, 607), (172, 622), (172, 659), (172, 747),
        #  (172, 777), (172, 781), (172, 782), (172, 889), (172, 921), (172, 937), (172, 985), (175, 0), (175, 10),
        #  (175, 10), (175, 17), (175, 18), (175, 21), (175, 22), (175, 26), (175, 28), (175, 32), (175, 32), (175, 37),
        #  (175, 37), (175, 42), (175, 46), (175, 49), (175, 49), (175, 52), (175, 53), (175, 53), (175, 53), (175, 54),
        #  (175, 57), (175, 59), (175, 60), (175, 60), (175, 62), (175, 69), (175, 70), (175, 71), (175, 76), (175, 81),
        #  (175, 83), (175, 84), (175, 89), (175, 90), (175, 90), (175, 97), (175, 98), (175, 103), (175, 130),
        #  (175, 137), (175, 145), (175, 148), (175, 148), (175, 149), (175, 154), (175, 155), (175, 160), (175, 160),
        #  (175, 166), (175, 173), (175, 176), (175, 180), (175, 188), (175, 189), (175, 190), (175, 190), (175, 190),
        #  (175, 192), (175, 195), (178, 0), (178, 14), (178, 19), (178, 23), (178, 24), (178, 26), (178, 41), (178, 65),
        #  (178, 69), (178, 75), (178, 85), (178, 87), (178, 92), (178, 99), (178, 108), (178, 108), (178, 120),
        #  (178, 125), (178, 126), (178, 139), (178, 140), (178, 143), (178, 168), (178, 168), (178, 171), (178, 181),
        #  (178, 181), (178, 189), (178, 195), (178, 208), (178, 208), (178, 211), (178, 212), (178, 216), (178, 219),
        #  (181, 38), (181, 251), (181, 508), (181, 972), (184, 3), (184, 23), (184, 30), (184, 36), (184, 38), (184, 45),
        #  (184, 45), (184, 49), (184, 95), (184, 127), (184, 128), (184, 159), (184, 175), (184, 182), (184, 200),
        #  (184, 210), (184, 217), (184, 218), (187, 12), (187, 29), (187, 37), (187, 45), (187, 104), (187, 149),
        #  (187, 196), (187, 216), (187, 237), (187, 237), (190, 319), (190, 452), (190, 543), (190, 688), (190, 736),
        #  (193, 210), (193, 214), (196, 6), (196, 21), (196, 63), (196, 199), (196, 239), (199, 37), (199, 91),
        #  (199, 237), (199, 257), (199, 406), (199, 569), (199, 575), (202, 25), (202, 29), (202, 29), (202, 72),
        #  (202, 98), (202, 108), (202, 109), (202, 115), (202, 134), (202, 137), (202, 156), (202, 162), (202, 163),
        #  (202, 212), (202, 222), (205, 0), (205, 1), (205, 25), (205, 62), (205, 90), (205, 90), (205, 108), (205, 117),
        #  (205, 167), (205, 179), (205, 182), (205, 183), (208, 56), (208, 409), (208, 484), (208, 775), (208, 803),
        #  (208, 871), (211, 0), (211, 30), (211, 100), (211, 130), (211, 134), (211, 139), (211, 160), (211, 163),
        #  (211, 195), (211, 204), (214, 3), (214, 5), (214, 19), (214, 21), (214, 68), (214, 69), (214, 70), (214, 74),
        #  (214, 99), (214, 101), (214, 153), (214, 154), (214, 155), (214, 185), (217, 396), (217, 423), (217, 450),
        #  (217, 459), (217, 500), (217, 575), (217, 824), (217, 838), (220, 0), (220, 155), (220, 169), (220, 245),
        #  (223, 71), (223, 115), (223, 178), (223, 203), (223, 232), (223, 249), (226, 86), (226, 108), (229, 59),
        #  (229, 105), (229, 107), (229, 141), (229, 154), (229, 177), (232, 0), (232, 38), (232, 43), (232, 85),
        #  (232, 144), (232, 202), (232, 246), (235, 495), (235, 641), (238, 37), (238, 92), (238, 101), (238, 164),
        #  (238, 166), (238, 174), (241, 19), (241, 29), (241, 94), (241, 105), (241, 108), (241, 126), (241, 190),
        #  (241, 208), (241, 232), (244, 344), (244, 567), (244, 795), (244, 862), (247, 2), (247, 6), (247, 11),
        #  (247, 11), (247, 12), (247, 60), (247, 87), (247, 94), (247, 129), (247, 145), (247, 157), (247, 162),
        #  (247, 168), (247, 181), (247, 192), (247, 208), (250, 1), (250, 65), (250, 141), (250, 145), (250, 146),
        #  (250, 167), (250, 204), (250, 213), (250, 221), (250, 240), (253, 180), (253, 773), (256, 34), (256, 58),
        #  (256, 98), (256, 119), (256, 119), (256, 149), (256, 210), (256, 211), (256, 219), (256, 226), (259, 19),
        #  (259, 70), (259, 95), (259, 147), (259, 163), (259, 212), (262, 209), (262, 692), (265, 5), (265, 40),
        #  (265, 62), (265, 102), (265, 154), (265, 183), (265, 197), (265, 208), (265, 222), (265, 231), (268, 65),
        #  (268, 99), (271, 112), (271, 947), (274, 123), (274, 319), (277, 6), (277, 7), (277, 76), (277, 93), (277, 94),
        #  (277, 106), (277, 355), (277, 409), (277, 440), (277, 459), (277, 466), (277, 487), (280, 555), (280, 851),
        #  (280, 870), (280, 1045), (280, 1407), (283, 6), (283, 11), (283, 23), (283, 23), (283, 25), (283, 44),
        #  (283, 90), (283, 102), (283, 130), (283, 132), (283, 135), (283, 138), (283, 158), (283, 160), (283, 172),
        #  (283, 187), (283, 190), (283, 198), (283, 210), (283, 217), (283, 223), (283, 237), (283, 258), (283, 319),
        #  (283, 330), (283, 347), (283, 357), (283, 374), (283, 383), (283, 393), (283, 400), (283, 450), (286, 9),
        #  (286, 28), (286, 42), (286, 42), (286, 73), (286, 111), (286, 152), (286, 173), (286, 203), (286, 213),
        #  (286, 231), (286, 238), (286, 287), (286, 311), (286, 326), (286, 346), (286, 406), (286, 448), (289, 89),
        #  (289, 184), (289, 283), (289, 530), (289, 689), (289, 733), (289, 1120), (289, 1799), (289, 1867), (289, 1875),
        #  (289, 1929), (292, 57), (292, 57), (292, 183), (292, 356), (292, 457), (295, 81), (295, 323), (295, 438),
        #  (298, 94), (298, 703), (298, 726), (298, 1299)]
        # iterations = 2
        for it_k in range(iterations):
            print("Ranking filters.. ")

            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration,model_scale,struct_scale)

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
                l_index, block_index, bl_index = self.get_struct_filter_position(model_scale, struct_scale, layer_index,
                                                                                 filter_index)
                if l_index>=4 and bl_index==4:
                    print("Wrong ranking.")
                    continue
                # print("l=%d,block=%d,bl=%d" % (l_index, block_index, bl_index))
                # _,temp_model_scale=self.get_model_scales()
                # temp_struct_scale = self.get_structure_scales()
                # print("current model_scale:%s"%temp_model_scale)
                # print("current struct_scale:%s"%temp_struct_scale)
                # model = prune_resnet101_conv_layer(model, (total_l, model_scale), (l_index, block_index, bl_index), filter_index)
                model = prune_resnet_conv_layer(model, (total_l, model_scale, struct_scale),
                                                (l_index, block_index, bl_index), filter_index)

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
            # 40,18,0.1
            last_acc=self.model_train(optimizer, epoches=30, regular_step_size= 10,regular_gamma= 0.1)
            torch.save(self.resnet101, "./pruned/resnet101_cifar100_prunned_%d" % it_k)
            #0.03
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
        torch.save(self.resnet101, "resnet101_cifar100_prunned")

    def recover_from_prune(self):
        after_prune_params = sum(p.numel() for p in self.resnet101.parameters() if p.requires_grad)
        compress_ratio = 0.870137
        before_prune_params = after_prune_params / (1 - compress_ratio)
        before_prune_acc = 0.738000

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
        self.fc = nn.Linear(num_li, 100)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    args = get_args()
    if args.recover:
        savedStdout = sys.stdout
        with open('./recover_res101_cifar100.txt', 'w+', 1) as redirect_out:
            sys.stdout = redirect_out
            if torch.cuda.is_available():
                model = torch.load("resnet101_cifar100_prunned_13").cuda()
                use_gpu = True
            else:
                model = torch.load("resnet101_cifar100_prunned_13").cpu()
                use_gpu = False

            fine_tuner = PrunningFineTuner_ResNet50(args.train_path, args.test_path, model, use_gpu,
                                                    torch_version=args.torch_version)
            if args.recover:
                fine_tuner.recover_from_prune()
        sys.stdout = savedStdout
        print("Done!\n")
    else:
        savedStdout = sys.stdout
        with open('./out_res101_cifar100.txt', 'w+', 1) as redirect_out:
            sys.stdout = redirect_out
            # args = get_args()
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                if args.train:
                    model = ResNetModel_101().cuda()
                    if os.path.exists('./resnet101_params_cifar100.pth'):
                        model.load_state_dict(torch.load('./resnet101_params_cifar100.pth'))
                    # model = model.cuda()
                elif args.prune:
                    model = torch.load("resnet101_model_cifar100").cuda()
            else:
                if args.train:
                    model = ResNetModel_101()
                    if os.path.exists('./resnet101_params_cifar100.pth'):
                        model.load_state_dict(torch.load('./resnet101_params_cifar100.pth'))
                elif args.prune:
                    # python3本地版没有办法load服务器python2保存下来的model(如果只是参数model.pth那种好像可以,但是包括架构的model不行),会有UnicodeDecodeError
                    model = torch.load("resnet101_model_cifar100")

            fine_tuner = PrunningFineTuner_ResNet50(args.train_path, args.test_path, model, use_gpu,
                                                 torch_version=args.torch_version)
            if args.train:
                fine_tuner.model_train(optimizer=fine_tuner.optimizer, epoches=args.train_epoch)
                torch.save(model.state_dict(), './resnet101_params_cifar100.pth')
                torch.save(model, "resnet101_model_cifar100")
            elif args.prune:
                fine_tuner.prune()
        sys.stdout = savedStdout
        print("Done!\n")
