#coding:utf8
import torch
import torchvision

import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

import numpy as np
import time
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# cifar 100
# cifar 10
# mnist
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

def print_model_parm_flops():

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            # print 'flops:{}'.format(self.__class__.__name__)
            # print 'input:{}'.format(input)
            # print '_dim:{}'.format(input[0].dim())
            # print 'input_shape:{}'.format(np.prod(input[0].shape))
            # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    multiply_adds = False
    list_conv = []

    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            # print 'flops:{}'.format(self.__class__.__name__)
            # print 'input:{}'.format(input)
            # print '_dim:{}'.format(input[0].dim())
            # print 'input_shape:{}'.format(np.prod(input[0].shape))
            # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
        2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)


    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())


    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())


    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)


    def foo(net):
        childrens = list(net.children())
        # print(childrens)
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    def test(net,torch_version,test_data_loader):
        net.eval()
        correct = 0
        total = 0
        if torch_version == 0.4:
            with torch.no_grad():
                since = time.time()
                for data in test_data_loader:
                    images, labels = data
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        images = Variable(images)
                        labels = Variable(labels)
                    outputs = net(images)
                    total += labels.size(0)
                time_elapsed = time.time() - since
                print('Everage forward time is %f s'%(
                    time_elapsed/total))
        elif torch_version == 0.3:
            since = time.time()
            for data in test_data_loader:
                images, labels = data
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                else:
                    images = Variable(images)
                    labels = Variable(labels)
                outputs = net(images)
                total += labels.size(0)
            time_elapsed = time.time() - since
            print('Everage forward time is %f s'%(
                time_elapsed / total))


    inputs = Variable(torch.rand(3, 224, 224).unsqueeze(0), requires_grad=True)
    dataset = DataSet(torch_v=0.3)
    if torch.cuda.is_available():
        pin_memory = True
        inputs=inputs.cuda()
    else:
        pin_memory = False
    test_data_loader = dataset.test_loader("./data", pin_memory=pin_memory)

    # if torch.cuda.is_available():
    #     model = torch.load("vgg16_model").cuda()
    # else:
    #     model=torch.load("vgg16_model").cpu
    # foo(model)
    # out=model(inputs)
    # total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    # print(' VGG16-10 + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    # test(model,0.3,test_data_loader)
    # del model
    #
    # if torch.cuda.is_available():
    #     model = torch.load("vgg16_cifar10_prunned").cuda()
    # else:
    #     model=torch.load("vgg16_cifar10_prunned").cpu
    # foo(model)
    # out=model(inputs)
    # total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    # print(' VGG16-10-pruned + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    # test(model,0.3,test_data_loader)
    # del model

    # if torch.cuda.is_available():
    #     model = torch.load("resnet50_model").cuda()
    # else:
    #     model=torch.load("resnet50_model").cpu
    # foo(model)
    # out = model(inputs)
    # total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    # print(' ResNet50-10 + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    # test(model,0.3,test_data_loader)
    # del model

    # if torch.cuda.is_available():
    #     model = torch.load("resnet50_cifar10_prunned").cuda()
    # else:
    #     model=torch.load("resnet50_cifar10_prunned").cpu
    # foo(model)
    # out = model(inputs)
    # total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    # print(' ResNet50-10-pruned + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    # test(model,0.3,test_data_loader)
    # del model
    #
    # if torch.cuda.is_available():
    #     model = torch.load("resnet101_model_cifar10").cuda()
    # else:
    #     model=torch.load("resnet101_model_cifar10").cpu
    # foo(model)
    # out = model(inputs)
    # total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    # print(' ResNet101-10 + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    # test(model,0.3,test_data_loader)
    # del model

    if torch.cuda.is_available():
        model = torch.load("resnet101_cifar10_prunned_14").cuda()
    else:
        model=torch.load("resnet101_cifar10_prunned_14").cpu
    foo(model)
    out = model(inputs)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    print(' ResNet101-10-pruned + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    test(model,0.3,test_data_loader)
    del model

if __name__ == '__main__':
    print_model_parm_flops()
    '''
    VGG16-10 + Number of FLOPs: 15.50G
    Everage forward time is 0.004333 s
    VGG16-10-pruned + Number of FLOPs: 1.70G
    Everage forward time is 0.001357 s
    ResNet50-10 + Number of FLOPs: 8.22G
    Everage forward time is 0.001562 s
    ResNet50-10-pruned + Number of FLOPs: 0.94G
    Everage forward time is 0.000867 s
    ResNet101-10 + Number of FLOPs: 15.66G
    Everage forward time is 0.002679 s
    ResNet101-10-pruned + Number of FLOPs: 5.44G
    Everage forward time is 0.001637 s

    ResNet101-10-pruned + Number of FLOPs: 1.18G
    Everage forward time is 0.001233 s
    '''