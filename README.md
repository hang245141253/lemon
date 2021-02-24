
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

训练数据使用的深度学习框架为百度的飞桨PaddlePaddle，训练源代码在百度AI Studio实训平台中，链接如下[https://aistudio.baidu.com/aistudio/projectdetail/1555348](https://aistudio.baidu.com/aistudio/projectdetail/1555348)

## 部署

部署采用百度飞桨轻量级推理部署框架Paddle Lite，测试硬件RK3399，OS：Ubuntu18.04.5 LTS，某厂USB2.0 UVC摄像头

* armLinux
    预测库基于Paddle Lite v2.8版本，包含C++(armv8、armv7hf)，Python3.6(armv8)
    
提供了两种预测接口C++与Python3

具体使用方式如下：
