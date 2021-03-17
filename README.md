
# 基于Paddle-Lite的柠檬外观分类(4分类任务)

## 背景

项目源自比赛广岛Quest2020：柠檬外观分类使用的图像数据（第1阶段）用广岛县的柠檬形象挑战外观分类

比赛链接：[https://signate.jp/competitions/431](https://signate.jp/competitions/431)

## 任务

根据以下图像数据对柠檬等级进行分类。

有四个等级：0:優良、1:良、2:加工品、3:規格外。

![](https://static.signate.jp/competitions/362/PTRaAuXIelqs1cZsp7EXtoHqFGq4CtGhMo3cgCEr.png)

其中：训练集1102张图像，测试集1651张图像

【注意事项】

在本次比赛中，假设获胜者的模型和专有技术将在Raspberry Pi等小型IoT终端上实施。
禁止将TTA（测试时间增加）用于推理。

## 训练

训练数据使用的深度学习框架为百度的飞桨PaddlePaddle，训练源代码在百度AI Studio实训平台中，链接如下[https://aistudio.baidu.com/aistudio/projectdetail/1592283](https://aistudio.baidu.com/aistudio/projectdetail/1592283)

## 部署

部署采用百度飞桨轻量级推理部署框架Paddle Lite，测试硬件RK3399，OS：Ubuntu18.04.5 LTS，某厂USB2.0 UVC摄像头

* armLinux
    预测库基于Paddle Lite v2.8版本，包含C++(armv8、armv7hf)，Python3.6(armv8)
    
提供了两种预测接口C++与Python3


-----
  
**环境准备**

* C++准备环境：
  
主要安装OpenCV3.2.0(推荐3.2)与CMake3.10

```bash
sudo apt-get update
sudo apt-get install gcc g++ make wget unzip libopencv-dev pkg-config
wget https://www.cmake.org/files/v3.10/cmake-3.10.3.tar.gz
tar -zxvf cmake-3.10.3.tar.gz
cd cmake-3.10.3
./configure
make
sudo make install
```

* Python环境准备：

主要是安装，numpy(1.13.3)，Pillow(8.1.0)，matplotlib(2.1.1)，OpenCV(3.2.0)(推荐3.2)
  
**以上工具版本号仅供参考，非必须对齐。**
<br></br>
优先推荐通过`pip3 install xxx`安装numpy，Pillow，matplotlib，OpenCV。

```bash
pip install numpy==1.13.3 pillow==8.1.0 matplotlib==2.1.1 opencv=3.2.0
```

安装matplotlib，OpenCV可能遇到报错，无需慌张，可`apt install python3-dev`后再次使用pip安装。若依旧不成功可使用`apt install python3-matplotlib` ， `apt install python3-opencv`安装。

配置好环境后稍后克隆一份部署Lemon源码，进入`cd ./lemon/wheels`文件夹后`pip3 install paddlelite-2.8rc0-cp36-cp36m-linux_aarch64.whl`（根据自己的Python版本选择，提供了Python2.7，3.5，3.6，3.7的包）
<br></br>
  
> 此类问题多百度，多参考其他人遇到问题解决的方式。当自己这类问题解决后，也写一篇博客来帮助其他人吧！
  
-----
  
**跑通Demo**

首先克隆一份部署Lemon源码  
  
```bash
git clone https://github.com/hang245141253/lemon.git
```

Lemon部署代码结构如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/77cc1c694255458fb0035b3e8fb463e221b560deff7e44d68070976193f1a610)

部署代码将C++与Python接口代码放入了同一文件中。如果想在Demo的基础上，换新的模型或者改变应用模型的方式，只要替换自己的model.nb或者修改main.cc、lemon.py即可。
<br></br>
如果你已经配置好了对应接口的环境，接下来就可以运行代码了！
<br></br>

* C++运行代码：

`cd ./lemon/code`进入code文件夹里后，执行`sh cmake.sh`会生成build文件夹，目标程序在build文件夹。在code目录下继续执行`sh run.sh`则开始执行部署程序。

![](https://ai-studio-static-online.cdn.bcebos.com/a60d610130424a40b740bd9549580b5521ebff3e69164dd6a7c44f2c622ce316)

* Python运行代码：
  
`cd ./lemon/code`进入code文件夹里后，执行`python3 lemon.py`运行程序。

![](https://ai-studio-static-online.cdn.bcebos.com/b05b6cf1514d4dd9adb94e25d44f30acb26ac0e775f347e0b97fd215b1e64483)
