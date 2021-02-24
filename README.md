
# 基于Paddle-Lite的柠檬外观分类(4分类任务)

## 背景

项目源自比赛广岛Quest2020：柠檬外观分类使用的图像数据（第1阶段）用广岛县的柠檬形象挑战外观分类！ 

比赛链接：![https://signate.jp/competitions/431](https://signate.jp/competitions/431)

## 任务

请根据以下图像数据对柠檬等级进行分类。

有四个等级：0：优秀，1：良好，2：加工产品，3：非标准。

【注意事项】

在本次比赛中，假设获胜者的模型和专有技术将在Raspberry Pi等小型IoT终端上实施。
禁止将TTA（测试时间增加）用于推理。
*有关特殊说明，请参阅“常见问题解答”以了解“卡明”期间出现的用户问题。
*有关其他规则的详细信息，请参阅“规则”页面。

![](https://ai-studio-static-online.cdn.bcebos.com/75c393dba672405781994ec80e053624caae91ed242c423e87f75eb27e7624d2)


## 环境要求

* armLinux
    预测库基于Paddle Lite v2.8版本，包含C++(armv8、armv7hf)，Python3.6(armv8)
