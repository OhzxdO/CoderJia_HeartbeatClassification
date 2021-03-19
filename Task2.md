# Task 2 数据分析

此部分主要进行EDA

**赛题：心电图心跳信号多分类预测**

## 2.1 EDA 目标

- EDA的价值主要在于熟悉数据集，了解数据集，对数据集进行验证来确定所获得数据集可以用于接下来的机器学习或者深度学习使用。
- 当了解了数据集之后我们下一步就是要去了解变量间的相互关系以及变量与预测值之间的存在关系。
- 引导数据科学从业者进行数据处理以及特征工程的步骤,使数据集的结构和特征集让接下来的预测问题更加可靠。

## 2.2 内容介绍

1. 载入各种数据科学以及可视化库:
   - 数据科学库 pandas、numpy、scipy；
   - 可视化库 matplotlib、seabon；

2. 载入数据：
   - 载入训练集和测试集；
   - 简略观察数据(head()+shape)；
   ![](https://img.imgdb.cn/item/60549fde524f85ce293ea050.jpg)

3. 数据总览:
   - 通过describe()来熟悉数据的相关统计量
   ![](https://img.imgdb.cn/item/6054a051524f85ce293ef81a.jpg)

   - 通过info()来熟悉数据类型
   ​![](https://img.imgdb.cn/item/6054a01d524f85ce293ed3bc.jpg)
4. 判断数据缺失和异常
   - 查看每列的存在nan情况
   - 异常值检测
   ![](https://img.imgdb.cn/item/6054a0da524f85ce293f63f5.jpg)

   本数据不存在缺失和异常

5. 了解预测值的分布
   - 总体分布概况
   ![](https://img.imgdb.cn/item/6054a3a1524f85ce29418408.jpg)

   - 查看skewness and kurtosis
   ![](https://img.imgdb.cn/item/6054a3ea524f85ce2941c4f6.jpg)

   - 查看预测值的具体频数
   ![](https://img.imgdb.cn/item/6054a416524f85ce2941e92b.jpg)

6. 用pandas_profiling生成数据报告
   ![](https://img.imgdb.cn/item/6054a301524f85ce2940f5c4.jpg)

