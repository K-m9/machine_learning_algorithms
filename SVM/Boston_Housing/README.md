# 波士顿房价模型(Boston_Housing)

## 1 建模目标
分析三个自变量与房价估值之间的关系，建立回归模型。
- How to use NumPy to investigate the latent features of a dataset.
- How to analyze various learning performance plots for variance and bias.
- How to determine the best-guess model for predictions from unseen data.
- How to evaluate a model's performance on unseen data using previous data.

## 2 数据解释
波士顿住房数据收集于1978年，506个条目中的每一个都代表了来自马萨诸塞州波士顿不同郊区的14个住房特征的汇总数据。
- 自变量
 - RM: 每个住宅的平均房间数 (average number of rooms per dwelling)
 - LSTAT: 被认为地位较低的人口的百分比 (percentage of population considered lower status)
 - PTRATIO: 按城镇分配的学生与教师比例 (pupil-teacher ratio by town)

- 因变量
 - MEDV: 该房屋的价值的估值中值 (median value of owner-occupied homes)

## 3 模型的建立与求解
### 3.1 数据预处理
#### 3.1.1 缺失值处理
```python3
df.isnull().any()
```
结果如下：
```python3
RM         False
LSTAT      False
PTRATIO    False
MEDV       False
dtype: bool
```
说明无缺失值。

#### 3.1.2 异常值处理
用马氏距离法计算每个样品到样本均值间的距离。设![](https://latex.codecogs.com/svg.latex?x^{(1)},...,x^{(n)})（其中 ![](https://latex.codecogs.com/svg.latex?x^{(i)}=(x_{i1},...,x_{ip})%27)）
为来自分布为![](https://latex.codecogs.com/svg.latex?N_p(\overline%20x,S))的n个样品，其中![](https://latex.codecogs.com/svg.latex?{\overline%20x}=(\overline%20x_1,\overline%20x_p)).
则样品![](https://latex.codecogs.com/svg.latex?x^{(i)})到均值 ![](https://latex.codecogs.com/svg.latex?\overline%20x)的马氏距离定义为<br>
<div align=center><img width="200" height="40" src="https://latex.codecogs.com/svg.latex?D_i^2(x)=%20(x_i-\overline%20x)S^{-1}(x_i-\overline%20x)^T"/></div>
其中S取样本协方差阵的估计值<sup>[1]-[2]</sup>。<br>

可以得出,![](https://latex.codecogs.com/svg.latex?D_i)服从卡方分布，
![](https://latex.codecogs.com/svg.latex?\chi^2_{0.05}(4)=11.14),可得出所有的![](https://latex.codecogs.com/svg.latex?D_i^2%3E11.14)的个数为0个，说明无异常值。

### 3.2 模型建立
由于模型的特征值较少，本文使用Lasso回归、弹性网回归和SVR进行建模比较，最终得出SVR模型的![](https://latex.codecogs.com/svg.latex?R^2)最大，因此取SVR模型作为最终模型。

#### 3.2.1 SVR基础知识<sup>[3]</sup><br>
给定训练样本![](https://latex.codecogs.com/svg.latex?D=\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\},y_i\in{R}),希望学得一个形如![](https://latex.codecogs.com/svg.latex?f(x)=\omega^Tx+b)的回归模型，使得f(x)与y尽可能接近。假定能容忍f(x)与y之间最多有![](https://latex.codecogs.com/svg.latex?\epsilon)的偏差，
即仅当f(x)与y之间的差别绝对值大于![](https://latex.codecogs.com/svg.latex?\epsilon)时才计算损失。
SVR问题可以形式化为：
<div align=center><img width="220" height="60" src="https://latex.codecogs.com/svg.latex?\min%20\limits_{\omega,b}%20\frac{1}{2}||\omega||^2+C\sum\limits_{i=1}^ml_{\epsilon}(f(x_i)-y_i)"/></div>

其中C为正则化常数，![](https://latex.codecogs.com/svg.latex?l_{\epsilon})是![](https://latex.codecogs.com/svg.latex?\epsilon)-不敏感损失函数,
<div align=center><img width="210" height="60" src="https://latex.codecogs.com/svg.latex?l_{\epsilon}(z)=\left\{\begin{aligned}0&,%20&%20if%20|z|\leq%20\epsilon;\\|z|-\epsilon%20&,&otherwise.%20\\\end{aligned}\right."/></div>

考虑特征映射形式，则SVR可表示为：
<div align=center><img width="210" height="60" src="https://latex.codecogs.com/svg.latex?f(x)=\sum\limits_{i=1}^m\alpha_i\kappa(x,x_i)+b"/></div>

其中![](https://latex.codecogs.com/svg.latex?\kappa(x,x_i))是核函数。

#### 3.2.2 数据集划分<sup>[3]</sup>
将模型划分为训练集和测试集，测试集用于判断最终模型的泛化性能，而训练集再另外划分为新的训练集和验证集，基于验证集上的性能进行模型的选择和调参。最终得出训练集:验证集:测试集=![](https://latex.codecogs.com/svg.latex?\frac{13}{25}:\frac{1}{5}:\frac{7}{25}).

#### 3.2.3 支持向量机回归
调用Python的sklearn包求解出最终的SVR模型为：
<div align=center><img width="210" height="60" src="https://latex.codecogs.com/svg.latex?f(x)=\sum\limits_{i=1}^{342}\alpha_i(x^Tx_i)^3+b"/></div>

其中，![](https://latex.codecogs.com/svg.latex?\alpha_i)和![](https://latex.codecogs.com/svg.latex?b)的值参考表格[系数.xlsx](https://github.com/K-m9/Boston_Housing/blob/master/%E7%B3%BB%E6%95%B0.xlsx)。

#### 3.2.4 性能度量
使用![](https://latex.codecogs.com/svg.latex?R^2)作为性能度量指标，将测试集代入最终的SVR模型，得到![](https://latex.codecogs.com/svg.latex?R^2=0.8688)。画出预测值和真实值的散点图如下图所示。
<div align=center><img width="400" height="250" src="https://github.com/K-m9/Boston_Housing/blob/master/pred_true.png"/></div>

## 4 结论
根据上述分析，波士顿房价模型如下所示：
<div align=center><img width="210" height="60" src="https://latex.codecogs.com/svg.latex?f(x)=\sum\limits_{i=1}^{342}\alpha_i(x^Tx_i)^3+b"/></div>

其中，![](https://latex.codecogs.com/svg.latex?R^2=0.8688)，即模型泛化性能较好，由真实值和预测值的散点图可看出模型预测精度较高。

## 5 参考文献
[1] 赵慧,甘仲惟,肖明.多变量统计数据中异常值检验方法的探讨[J].华中师范大学学报(自然科学版),2003(02):133-137.<br>
[2] 高惠璇.应用多元统计分析[M].北京,北京大学出版社,2005.54:103.<br>
[3] 周志华.机器学习[M].北京,清华大学出版社,2016.23:46,121:139.
