# Task 5: 模型融合


## 5.1 学习目标

- 学习融合策略
- 完成相应学习打卡任务 

## 5.2 内容介绍

https://mlwave.com/kaggle-ensembling-guide/  
https://github.com/MLWave/Kaggle-Ensemble-Guide

模型融合是比赛后期一个重要的环节，大体来说有如下的类型方式。

1. 简单加权融合:
    - 回归（分类概率）：算术平均融合（Arithmetic mean），几何平均融合（Geometric mean）；
    - 分类：投票（Voting)
    - 综合：排序融合(Rank averaging)，log融合

2. stacking/blending:
    - 构建多层模型，并利用预测结果再拟合预测。

3. boosting/bagging（在xgboost，Adaboost,GBDT中已经用到）:
    - 多树的提升方法

## 5.3 相关理论介绍

stacking具体原理详解
1. https://www.cnblogs.com/yumoye/p/11024137.html
2. https://zhuanlan.zhihu.com/p/26890738


## 5.4 本赛题示例

### 5.4.1 准备工作

准备工作进行内容有：
1. 导入数据集并进行简单的预处理
2. 将数据集划分成训练集和验证集
3. 构建单模：Random Forest，LGB，NN
4. 读取并演示如何利用融合模型生成可提交预测数据



## 5.5 经验总结

![](https://img.imgdb.cn/item/6060a05d8322e6675c1d7ef9.jpg)
本部分通过学习示例代码基本实现了赛题所要求内容，了解掌握了基本流程及思想，sklearn的使用使得代码实现起来比较简单，但是对于单个模型原理的理解还需更加深入的学习，在示例代码的三个模型中我仅对随机森林的算法原理了解比较多，对于其他模型还不是很能理解其原理，有待今后进一步学习。


**END.**

【 张晋 ：Datawhale成员，算法竞赛爱好者。CSDN：https://blog.csdn.net/weixin_44585839/】




