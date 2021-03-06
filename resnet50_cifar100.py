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

        self.resnet50 = model
        # self.criterion = torch.nn.CrossEntropyLoss()
        # define the loss function
        self.loss_function = nn.CrossEntropyLoss()
        # define the optimizer
        self.optimizer = optim.SGD(self.resnet50.parameters(), lr=0.1, momentum=0.9)
        # define the changing strategy of learning rate
        # learning rate will be changed into gamma*lr after each step_size epochs

        self.prunner = FilterPrunner(self.resnet50)

        self.inputs, self.labels, self.preds = None, None, None
        self.outputs, self.loss = None, None
        self.best_acc = 0.0
        self.best_model_wts = None

        # set model to training mode
        # the default Batch Normalization and Dropout mode is different in train & eval
        self.resnet50.train()

    def model_train(self, optimizer=None, epoches=10,regular_step_size = 20, regular_gamma = 0.1):
        self.resnet50.train()
        use_scheduler = False
        if optimizer is None:
            # allow only fine-tuning fc layer
            optimizer = \
                optim.SGD(self.resnet50.fc.parameters(),
                          lr=0.0001, momentum=0.9)
            use_scheduler = False
            # Make sure all the layers except fc layer are not trainable
            for param in self.resnet50.features.parameters():
                param.requires_grad = False

        else:
            optimizer = optimizer
            use_scheduler = True
            self.scheduler = lr_scheduler.StepLR(optimizer, step_size=regular_step_size, gamma=regular_gamma)

        self.best_model_wts = self.resnet50.state_dict()
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
                self.best_model_wts = self.resnet50.state_dict()
                torch.save(self.resnet50.state_dict(), './resnet50_params_cifar100.pth')

        time_elapsed = time.time() - since
        print("Finished fine tuning.")
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_acc))
        self.resnet50.load_state_dict(self.best_model_wts)
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
            self.resnet50.zero_grad()

            if rank_filters:
                output = self.prunner.forward(self.inputs)
                self.loss_function(output, self.labels).backward()
            else:
                # forward
                self.outputs = self.resnet50(self.inputs)
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
        self.resnet50.eval()
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
                    outputs = self.resnet50(images)
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
                outputs = self.resnet50(images)
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
            outputs = self.resnet50(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        '''

        self.resnet50.train()

        return float(correct) / total

    def model_test_onestep(self):
        self.resnet50.eval()
        dataiter = iter(self.test_data_loader)
        images, labels = dataiter.next()
        # print images
        # self.imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % self.classes[labels[j]] for j in range(4)))

        test_outputs = self.resnet50(images)

        _, predicted = torch.max(test_outputs, 1)

        print('Predicted: ', ' '.join('%5s' % self.classes[predicted[j]]
                                      for j in range(4)))

    def model_test_diffclasses(self):
        self.resnet50.eval()
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in self.test_data_loader:
                images, labels = data
                outputs = self.resnet50(images)
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
        for name, module in self.resnet50.features._modules.items():
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
        for layer, (name, module) in enumerate(self.resnet50.features._modules.items()):
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
        for layer, (name, module) in enumerate(self.resnet50.features._modules.items()):
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

        self.resnet50.train()

        # Make sure all the layers are trainable
        for param in self.resnet50.parameters():
            param.requires_grad = True

        before_prune_params = sum(p.numel() for p in self.resnet50.parameters() if p.requires_grad)

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

            print("Layers that will be prunned: %s" % prune_targets)

            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned: %s" % layers_prunned)

            print("Prunning filters.. ")
            model = self.resnet50.cpu()

            # print("model:%s" % model_scale)
            # print("struct model:%s" % struct_scale)
            for layer_index, filter_index in prune_targets:
                l_index, block_index, bl_index = self.get_struct_filter_position(model_scale,struct_scale, layer_index, filter_index)

                if l_index >= 4 and bl_index == 4:
                    print("Wrong ranking.")
                    continue
                model = prune_resnet_conv_layer(model, (total_l, model_scale, struct_scale), (l_index, block_index, bl_index), filter_index)

            if self.use_gpu:
                self.resnet50 = model.cuda()
            else:
                self.resnet50 = model
            self.prunner.model = self.resnet50
            struct_scale_temp = self.get_structure_scales()
            if struct_scale_temp == struct_scale:
                print("The same struct!")
            else:
                print("Not the same struct!")
                return
            # f = self.resnet50.features
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
            optimizer = optim.SGD(self.resnet50.parameters(), lr=0.1, momentum=0.9)
            last_acc=self.model_train(optimizer, epoches=30, regular_step_size= 10,regular_gamma= 0.1)
            torch.save(self.resnet50, "./pruned/resnet50_cifar100_prunned_%d" % it_k)
            last_prune_params = sum(p.numel() for p in self.resnet50.parameters() if p.requires_grad)
            print("Parameters Compression ratio is %f." % (float(before_prune_params - last_prune_params) / before_prune_params))
            if last_acc <= before_prune_acc-0.06:
                break

        print("Finished. Going to fine tune the model a bit more")
        opt = optim.SGD(self.resnet50.parameters(), lr=0.01, momentum=0.9)
        last_acc = self.model_train(opt, epoches=40,regular_step_size= 20, regular_gamma=0.1)
        last_prune_params = sum(p.numel() for p in self.resnet50.parameters() if p.requires_grad)
        print("Max acc loss is %f. Parameters Compression ratio is %f." % ((before_prune_acc - last_acc), float(before_prune_params-last_prune_params)/before_prune_params))
        torch.save(self.resnet50, "resnet50_cifar100_prunned")

    def recover_from_prune(self):
        after_prune_params = sum(p.numel() for p in self.resnet50.parameters() if p.requires_grad)
        compress_ratio = 0.852071
        before_prune_params = after_prune_params / (1 - compress_ratio)
        before_prune_acc = 0.7279

        print("Finished. Going to fine tune the model a bit more")
        opt = optim.SGD(self.resnet50.parameters(), lr=0.01, momentum=0.9)
        last_acc = self.model_train(opt, epoches=40, regular_step_size=20, regular_gamma=0.1)

        last_prune_params = sum(p.numel() for p in self.resnet50.parameters() if p.requires_grad)

        print("Max acc loss is %f. Parameters Compression ratio is %f." % (
        (before_prune_acc - last_acc), float(before_prune_params - last_prune_params) / before_prune_params))
        torch.save(self.resnet50, "resnet50_cifar100_prunned")

class ResNetModel_50(torch.nn.Module):

    def __init__(self):
        super(ResNetModel_50, self).__init__()
        # Import pre-trained model
        resnet50 = models.resnet50(pretrained=True)

        self.inplanes = resnet50.inplanes
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = resnet50.avgpool

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
        # Get input dim of the last fv layer of resnet50 and replace the last layer with another
        # which has the same input dim while has the different output dim
        num_li = resnet50.fc.in_features
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args = get_args()
    if args.recover:
        savedStdout = sys.stdout
        with open('./recover_res50_cifar100.txt', 'w+', 1) as redirect_out:
            sys.stdout = redirect_out
            if torch.cuda.is_available():
                model = torch.load("resnet50_cifar100_prunned_6").cuda()
                use_gpu = True
            else:
                model = torch.load("resnet50_cifar100_prunned_6").cpu()
                use_gpu = False

            fine_tuner = PrunningFineTuner_ResNet50(args.train_path, args.test_path, model, use_gpu,
                                                 torch_version=args.torch_version)
            if args.recover:
                fine_tuner.recover_from_prune()
        sys.stdout = savedStdout
        print("Done!\n")
    else:
        savedStdout = sys.stdout
        with open('./out_res50_cifar100.txt', 'w+', 1) as redirect_out:
            sys.stdout = redirect_out
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                if args.train:
                    model = ResNetModel_50().cuda()
                    if os.path.exists('./resnet50_params_cifar100.pth'):
                        model.load_state_dict(torch.load('./resnet50_params_cifar100.pth'))
                    # model = model.cuda()
                elif args.prune:
                    model = torch.load("resnet50_model_cifar100").cuda()
            else:
                if args.train:
                    model = ResNetModel_50()
                    if os.path.exists('./resnet50_params_cifar100.pth'):
                        model.load_state_dict(torch.load('./resnet50_params_cifar100.pth'))
                elif args.prune:
                    # python3本地版没有办法load服务器python2保存下来的model(如果只是参数model.pth那种好像可以,但是包括架构的model不行),会有UnicodeDecodeError
                    model = torch.load("resnet50_model_cifar100")

            fine_tuner = PrunningFineTuner_ResNet50(args.train_path, args.test_path, model, use_gpu,
                                                 torch_version=args.torch_version)
            if args.train:
                fine_tuner.model_train(optimizer=fine_tuner.optimizer, epoches=args.train_epoch)
                torch.save(model.state_dict(), './resnet50_params_cifar100.pth')
                torch.save(model, "resnet50_model_cifar100")
            elif args.prune:
                fine_tuner.prune()
        sys.stdout = savedStdout
        print("Done!\n")
