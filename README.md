##Running Request:
- python2.7, pytorch 0.3.*, numpy, argparse

##Experienment Environment:
- cuda 8, CuDNN 6.0.20, NVIDIA TITAN XP 12G

##Excute the code:
You could clone the project to your own environment, and run the code with specific command line arguments,
For example:
```
python VGG16_cifar10.py --train --torch_version 0.3 --train_epoch 20
python resnet50_cifar10.py --prune --torch_version 0.3 # Require having trained model in the root folder before pruning
```

##Args explanision:
1. Optional arguments when you run VGG16_*.py:
    * --train, train model from Imagenet12 pre-trained models in pytorch model zoo.
    * --prune, prune the existing model.
    * --train_path, path of the training dataset, default is mnist_data or data.
    * --test_path, path of the testing dataset, default is mnist_data or data.
    * --train_epoch, epoch number of training.
    * --torch_version, set the torch version of code, default is 0.3, partially support 0.4.
    * --restore, restore model from existing one if restore option is set true, only useful when training, and it will be trained from a new model by default.
    * --full_train, fine-tuning the whole pre-trained VGG16 model if full_train is set True, dafault only fine-tuning the fully connected layers of pre-trained VGG16 model.

2. Optional arguments when you run resnet*.py:
    
    Mostly the same as optional arguments in VGG16_*.py, differences are following:
    
    without options:
    * --restore
    * --full_train
    
    add options:
    * --recover, used to continue the last fine-tuning process after the last pruning action when the process is terminated because of trying to remove the whole layer.

##Other Instructions:
- 默认cifar*数据文件放在文件夹./data中，mnist数据文件在文件夹./mnist_data中，运行代码前请确保文件夹存在，代码会自动下载指定数据集。
- flop_cal*.py用于计算剪枝后网络的加速比与浮点运算减少量，因面向三个任务中的网络结构不同，并且是在pytorch默认的模型基础上修改的，因此需要首先声明相应的类，故写在三个文件中。
- VGG16*.py与resnet*.py对应不同的剪枝实验，本次作业中共针对三种数据集，三种网络的7种组合做实验，故共有7个实验实现python文件。
- prune.py集成了针对VGG网络与resnet网络的剪枝方法，作为工具类被不同实验调用。
