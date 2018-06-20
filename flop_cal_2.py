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
            dataset=datasets.MNIST(root=path, train=True, download=True, transform=traindata_transforms),
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
            dataset=datasets.MNIST(root=path, train=False, download=True, transform=testdata_transforms),
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
        features = model.features
        # features[0]=nn.Conv2d(in_channels=1, out_channels=features[0].out_channels,kernel_size=3, padding=1)
        new_conv = None
        layers = []
        for i, (name,module) in enumerate(features._modules.items()):
            if i == 0:
                new_conv = nn.Conv2d(in_channels=1, out_channels=module.out_channels,kernel_size=3, padding=1)
                layers += [new_conv]
            else:
                layers += [module]
        self.features = nn.modules.Sequential(*layers)
        del features
        del model

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

    if torch.cuda.is_available():
        input = Variable(torch.rand(1, 224, 224).unsqueeze(0), requires_grad=True).cuda()
    else:
        input = Variable(torch.rand(1, 224, 224).unsqueeze(0), requires_grad=True)
    dataset = DataSet(torch_v=0.3)
    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False
    test_data_loader = dataset.test_loader("./mnist_data", pin_memory=pin_memory)

    # if torch.cuda.is_available():
    #     model = torch.load("vgg16_model_mnist").cuda()
    # else:
    #     model=torch.load("vgg16_model_mnist").cpu
    # foo(model)
    # out=model(input)
    # total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    # print(' VGG16-mnist + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    # test(model,0.3,test_data_loader)
    # del model

    if torch.cuda.is_available():
        model = torch.load("vgg16_mnist_prunned").cuda()
    else:
        model=torch.load("vgg16_mnist_prunned").cpu
    foo(model)
    out=model(input)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    print(' VGG16-mnist-pruned + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    test(model,0.3,test_data_loader)
    del model


if __name__ == '__main__':
    print_model_parm_flops()
    '''
    VGG16-mnist + Number of FLOPs: 15.44G
    Everage forward time is 0.004260 s
    VGG16-mnist-pruned + Number of FLOPs: 0.05G
    Everage forward time is 0.000660 s
    '''