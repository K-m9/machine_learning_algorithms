# [Task01：基于逻辑回归的分类预测](https://developer.aliyun.com/ai/scenario/9ad3416619b1423180f656d1c9ae44f7?spm=a2c6h.14441864.0.0.f362354f7oXmdM)

## Part1 Demo实践

 - Step1:库函数导入
  ```python3
   ##  基础函数库
   import numpy as np 

   ## 导入画图库
   import matplotlib.pyplot as plt
   import seaborn as sns

   ## 导入逻辑回归模型函数
   from sklearn.linear_model import LogisticRegression
  ```
  
 - Step2:模型训练 
  ```python3
  ##Demo演示LogisticRegression分类

  ## 构造数据集
  x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
  y_label = np.array([0, 0, 0, 1, 1, 1])

  ## 调用逻辑回归模型
  lr_clf = LogisticRegression()

  ## 用逻辑回归模型拟合构造的数据集
  lr_clf = lr_clf.fit(x_fearures, y_label) #其拟合方程为 y=w0+w1*x1+w2*x2
  ```
  
 - Step3:模型参数查看 
  ```python3
  ##查看其对应模型的w
  print('the weight of Logistic Regression:',lr_clf.coef_)
  ##查看其对应模型的w0
  print('the intercept(w0) of Logistic Regression:',lr_clf.intercept_)
  ##the weight of Logistic Regression:[[0.73462087 0.6947908]]
  ##the intercept(w0) of Logistic Regression:[-0.03643213]
  ```
 
 - Step4:数据和模型可视化 
 ```python3
  ## 可视化构造的数据样本点
  plt.figure()
  plt.scatter(x_fearures[:,0],x_fearures[:,1], c=y_label, s=50, cmap='viridis')
  plt.title('Dataset')
  plt.show()
 ```
 ```python3
  # 可视化决策边界
  plt.figure()
  plt.scatter(x_fearures[:,0],x_fearures[:,1], c=y_label, s=50, cmap='viridis')
  plt.title('Dataset')

  nx, ny = 200, 100
  x_min, x_max = plt.xlim()
  y_min, y_max = plt.ylim()
  x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))

  z_proba = lr_clf.predict_proba(np.c_[x_grid.ravel(), y_grid.ravel()])
  z_proba = z_proba[:, 1].reshape(x_grid.shape)
  plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')

  plt.show()
 ```
 ```python3
 ### 可视化预测新样本

plt.figure()
## new point 1
x_fearures_new1 = np.array([[0, -1]])
plt.scatter(x_fearures_new1[:,0],x_fearures_new1[:,1], s=50, cmap='viridis')
plt.annotate(s='New point 1',xy=(0,-1),xytext=(-2,0),color='blue',arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))

## new point 2
x_fearures_new2 = np.array([[1, 2]])
plt.scatter(x_fearures_new2[:,0],x_fearures_new2[:,1], s=50, cmap='viridis')
plt.annotate(s='New point 2',xy=(1,2),xytext=(-1.5,2.5),color='red',arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))

## 训练样本
plt.scatter(x_fearures[:,0],x_fearures[:,1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')

# 可视化决策边界
plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')

plt.show()
 ```
 
 - Step5:模型预测
 ```python3
  ##在训练集和测试集上分布利用训练好的模型进行预测
  y_label_new1_predict=lr_clf.predict(x_fearures_new1)
  y_label_new2_predict=lr_clf.predict(x_fearures_new2)
  print('The New point 1 predict class:\n',y_label_new1_predict)
  print('The New point 2 predict class:\n',y_label_new2_predict)
  ##由于逻辑回归模型是概率预测模型（前文介绍的p = p(y=1|x,\theta)）,所有我们可以利用predict_proba函数预测其概率
  y_label_new1_predict_proba=lr_clf.predict_proba(x_fearures_new1)
  y_label_new2_predict_proba=lr_clf.predict_proba(x_fearures_new2)
  print('The New point 1 predict Probability of each class:\n',y_label_new1_predict_proba)
  print('The New point 2 predict Probability of each class:\n',y_label_new2_predict_proba)
  ##TheNewpoint1predictclass:
  ##[0]
  ##TheNewpoint2predictclass:
  ##[1]
  ##TheNewpoint1predictProbabilityofeachclass:
  ##[[0.695677240.30432276]]
  ##TheNewpoint2predictProbabilityofeachclass:
  ##[[0.119839360.88016064]]
 ```
 
 ## Part2 基于鸢尾花（iris）数据集的逻辑回归分类实践

 - Step1:库函数导入 
 - Step2:数据读取/载入 
 ```python3
 ##我们利用sklearn中自带的iris数据作为数据载入，并利用Pandas转化为DataFrame格式
 from sklearn.datasets import load_iris
 data = load_iris() #得到数据特征
 iris_target = data.target #得到数据对应的标签
 iris_features = pd.DataFrame(data=data.data, columns=data.feature_names) #利用Pandas转化为DataFrame格式
 ```
 
 - Step3:数据信息简单查看 
  ```python3
  ##利用.info()查看数据的整体信息
  iris_features.info()

  ##<class'pandas.core.frame.DataFrame'>
  ##RangeIndex:150entries,0to149
  ##Datacolumns(total4columns):
  ###ColumnNon-NullCountDtype
  ##----------------------------
  ##0sepallength(cm)150non-nullfloat64
  ##1sepalwidth(cm)150non-nullfloat64
  ##2petallength(cm)150non-nullfloat64
  ##3petalwidth(cm)150non-nullfloat64
  ##dtypes:float64(4)
  ##memoryusage:4.8KB
  ```
  ```python3
  ##进行简单的数据查看，我们可以利用.head()头部.tail()尾部
  iris_features.head() #iris_features.tail()
  ```
  ```python3
  ##其对应的类别标签为，其中0，1，2分别代表'setosa','versicolor','virginica'三种不同花的类别
  iris_target

  ##array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

  ##0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

  ##0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,

  ##1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,

  ##1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,

  ##2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,

  ##2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
  ```
  
 - Step4:可视化描述 
 
 - Step5:利用 逻辑回归模型 在二分类上 进行训练和预测 
 - Step6:利用 逻辑回归模型 在三分类(多分类)上 进行训练和预测
