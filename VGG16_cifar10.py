# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
from torchvision import models
import torchvision.transforms as transforms

from PIL import Image
import glob
import os
import sys
import argparse
from operator import itemgetter
from heapq import nsmallest
import time

# import cv2
# import dataset

from prune import *

'''
Reference:
Blog: https://jacobgil.github.io/deeplearning/pruning-deep-learning
github: https://github.com/jacobgil/pytorch-pruning
Paper:
[1]Molchanov P, Tyree S, Karras T, et al. Pruning convolutional neural networks for resource efficient transfer learning[J]. arXiv preprint arXiv:1611.06440, 2016.
'''

class CatDogDataSet:

    def loader(path, batch_size=32, num_workers=4, pin_memory=True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return data.DataLoader(
            datasets.ImageFolder(path,
                                 transforms.Compose([
                                     transforms.Scale(256),
                                     transforms.RandomSizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory)

    def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return data.DataLoader(
            datasets.ImageFolder(path,
                                 transforms.Compose([
                                     transforms.Scale(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory)


class DataSet:
    def __init__(self, torch_v=0.4):
        self.torch_v = torch_v

    def loader(self, path, batch_size=32, num_workers=4, pin_memory=True):
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
        return data.DataLoader(
            dataset=datasets.CIFAR10(root=path, train=True, download=True, transform=traindata_transforms),
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
        return data.DataLoader(
            dataset=datasets.CIFAR10(root=path, train=False, download=True, transform=testdata_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory)


class ModifiedVGG16Model(torch.nn.Module):

    def __init__(self):
        # Load pre-trained PyTorch VGG16 model
        super(ModifiedVGG16Model, self).__init__()
        model = models.vgg16(pretrained=True)

        # Maintain the features layers of PyTorch pre-trained VGG16
        self.features = model.features

        # Freeze the features layers, which allow only fine-tuning the last 3 layers FC classifier later
        for param in self.features.parameters():
            param.requires_grad = False

        # Replace the last three layers of pre-trained VGG16(on ImageNet)
        # Make the model adaptive to new classification task(Such as DogVCat or Cifar10)

        # original FC setting in PyTorch VGG16 with num_classes=1000
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )
        # FC setting used for Asirra Dogs vs. Cats data set
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(25088, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 2))
        # FC setting used for Cifar19 data set
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank) # How is compute_rank function used?
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

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

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune


class PrunningFineTuner_VGG16:
    '''
    Provide train(include fine-tuning), test, and prune function of the model
    Argments:
    train_path: folder path of train datas
    test_path: folder path of test datas
    model: the model to be trained/continuously trained, tested or pruned
    '''
    def __init__(self, train_path, test_path, model, use_gpu, torch_version = 0.4):
        self.torch_version = torch_version
        self.use_gpu = use_gpu
        dataset=DataSet(torch_v=self.torch_version)
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        # set model to training mode
        # the default Batch Normalization and Dropout mode is different in train & eval
        self.model.train()

    def test(self):
        # set model to evaluating/testing mode
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            if self.use_gpu:
                batch = batch.cuda()
            if self.torch_version == 0.3:
                batch = Variable(batch)
            output = model(batch)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy : %f" % (float(correct) / total))
        # set model return to training mode
        self.model.train()
        return float(correct) / total

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = \
                optim.SGD(model.classifier.parameters(),
                          lr=0.0001, momentum=0.9)
        return_last_test_acc = 0
        for i in range(epoches):
            print("Epoch: %d" % i)
            self.train_epoch(optimizer)
            return_last_test_acc=self.test()
        print("Finished fine tuning.")
        return return_last_test_acc

    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()

        if self.torch_version == 0.3:
            input = Variable(batch)
            label = Variable(label)
        else:
            input = batch

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, label).backward()
        else:
            self.criterion(self.model(input), label).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):
        for batch, label in self.train_data_loader:
            if self.use_gpu:
                batch = batch.cuda()
                label = label.cuda()
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        # 虽然调用的是train，但其实在rank_filters设置为True时，train_epoch中做的只是forward与backward(计算grad)计算，没有优化参数(没有更新)
        # 这些forward与backward计算中得到的必要参数用于计算oracle_rank, compute_rank函数被register了，使其可以在forward/backward 的同时被调用计算
        # 具体调用compute_rank的原理要再看一下
        self.train_epoch(rank_filters=True)

        self.prunner.normalize_ranks_per_layer()

        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        # Get the accuracy before prunning
        before_prune_acc = self.test()
        self.model.train()

        # Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        before_prune_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        # iterations = int(iterations * 2.0 / 3)
        #
        # print("Number of prunning iterations to reduce 2/3 filters: %d " % iterations)
        print("Max number of prunning iterations: %d " % iterations)

        for it_k in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned: %s" % layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index)

            if self.use_gpu:
                self.model = model.cuda()
            else:
                self.model = model
            #TODO: Ensure prunner is useful and correct
            #self.prunner.model = self.model

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned: %s" % str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            last_acc = self.train(optimizer, epoches=10)
            torch.save(model,"./pruned/vgg16_cifar10_prunned_%d" % it_k)
            if last_acc <= before_prune_acc-0.02:
                break

        print("Finished. Going to fine tune the model a bit more")
        last_acc = self.train(optimizer, epoches=15)
        last_prune_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Max acc loss is %f. Parameters Compression ratio is %f." % ((before_prune_acc - last_acc), float(before_prune_params-last_prune_params)/before_prune_params))
        torch.save(model, "vgg16_cifar10_prunned")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    # parser.add_argument("--train_path", type=str, default="train")
    # parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument("--train_path", type=str, default="./data")
    parser.add_argument("--test_path", type=str, default="./data")
    parser.add_argument("--train_epoch", dest="train_epoch", action="store", type=int, default=20)
    parser.add_argument("--torch_version", dest="torch_version", action="store", type=float, default=0.4)
    parser.add_argument("--restore", dest="restore", action="store_true")
    parser.add_argument("--full_train", dest="full_train", action="store_true")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    parser.set_defaults(restore=False)
    parser.set_defaults(full_train=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    第一次运行应该首先运行train,这样ModifiedVGG16会根据新数据进行最后FC层的训练,得到剪枝之前的baseline model，保存为model
    有了model之后，剪枝操作则只需要运行prune,每次运行prune都是起始于model参数，
    整个prune过程包括：
    1.展示原始模型的准确率,
    2.[剪枝一次(512个filters-也可以一次剪枝1个filter但是太慢了)，展示剪枝后模型准确率，fine-tuning(整个模型上，不是仅FC,同时会展示fine-tuning之后的准确率)]-循环n轮，
    3.最后一次fine-tuning
    4.保存最后的模型搭配model_pruned
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    savedStdout = sys.stdout
    with open('./vgg_out.txt', 'w+', 1) as redirect_out:
        sys.stdout = redirect_out
        args = get_args()
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            if args.train:
                model = ModifiedVGG16Model().cuda()
                if args.restore:
                    model.load_state_dict(torch.load("vgg16_model").state_dict())
            elif args.prune:
                model = torch.load("vgg16_model").cuda()
        else:
            if args.train:
                model = ModifiedVGG16Model()
                if args.restore:
                    model.load_state_dict(torch.load("vgg16_model").state_dict())
            elif args.prune:
                # python3本地版没有办法load服务器python2保存下来的model(如果只是参数model.pth那种好像可以,但是包括架构的model不行),会有UnicodeDecodeError
                model = torch.load("vgg16_model")

        fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model, use_gpu, torch_version=args.torch_version)

        if args.train:
            if args.full_train:
                for param in fine_tuner.model.features.parameters():
                    param.requires_grad = True
                optimizer = optim.SGD(fine_tuner.model.parameters(), lr=0.001, momentum=0.9)
                fine_tuner.train(optimizer=optimizer, epoches=args.train_epoch)
            else:
                fine_tuner.train(epoches=args.train_epoch)
            torch.save(model, "vgg16_model")

        elif args.prune:
            fine_tuner.prune()
    sys.stdout = savedStdout
    print("Done!\n")