# Task3 特征工程


## 3.1 学习目标

* 学习时间序列数据的特征预处理方法
* 学习时间序列特征处理工具 Tsfresh（TimeSeries Fresh）的使用

## 3.2 内容介绍
* 数据预处理
	* 时间序列数据格式处理
	* 加入时间步特征time
* 特征工程
	* 时间序列特征构造
	* 特征筛选
	* 使用 tsfresh 进行时间序列特征处理

## 3.3 对心电特征进行行转列处理，同时为每个心电信号加入时间步特征time
![](https://img.imgdb.cn/item/60586fa98322e6675c8f2162.jpg)

## 3.4 将处理后的心电特征加入到训练数据中，同时将训练数据label列单独存储
![](https://img.imgdb.cn/item/6058707a8322e6675c8f8aaf.jpg)

每个样本的心电特征都由205个时间步的心电信号组成。


## 3.5 使用 tsfresh 进行时间序列特征处理
**Tsfresh（TimeSeries Fresh）**是一个Python第三方工具包。 它可以自动计算大量的时间序列数据的特征。此外，该包还包含了特征重要性评估、特征选择的方法，因此，不管是基于时序数据的分类问题还是回归问题，tsfresh都会是特征提取一个不错的选择。官方文档：[Introduction — tsfresh 0.17.1.dev24+g860c4e1 documentation](https://tsfresh.readthedocs.io/en/latest/text/introduction.html)

**以下部分由于本地内存不足，采用云天池实验室完成。了解了tsfresh相关知识，这是第一次接触到tsfresh，大概了解了它的作用，相关原理还需要进一步消化

1. 特征提取
![](https://img.imgdb.cn/item/60588bfd8322e6675c9e1f8c.jpg)

2. 特征选择 
train_features中包含了heartbeat_signals的779种常见的时间序列特征（所有这些特征的解释可以去看官方文档），这其中有的特征可能为NaN值（产生原因为当前数据不支持此类特征的计算），使用以下方式去除NaN值：
![](https://img.imgdb.cn/item/60588c4e8322e6675c9e481b.jpg)

接下来，按照特征和响应变量之间的相关性进行特征选择，这一过程包含两步：首先单独计算每个特征和响应变量之间的相关性，然后利用Benjamini-Yekutieli procedure [1] 进行特征选择，决定哪些特征可以被保留。
![](https://img.imgdb.cn/item/60588c758322e6675c9e5b28.jpg)

## References

[1] Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. Annals of statistics, 1165–1188

**Task3 特征工程 END.**

--- By: 吉米杜

> 平安NLP算法工程师，Datawhale成员，Coggle开源小组成员
>
> 博客：https://blog.csdn.net/duxiaodong1122?spm=1011.2124.3001.5343&type=blog

关于Datawhale：
Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale 以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时 Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。
本次数据挖掘路径学习，专题知识将在天池分享，详情可关注Datawhale：