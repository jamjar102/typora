# KDD15-AQI inference 

## 一、论文公式python代码实现

$$
a(u,v)=exp(-\sum_{k=1}^{m}{\Pi^2*AF_k(\Delta f_k（u,v）)})              （1）
$$

该公式分为两部分，先求下面这个矩阵
$$
AffinityWeight={AF_k(\Delta f_k（u,v）)}  (2)
$$
然后带入如下
$$
weightMatrix=exp(-\sum_{k=1}^{m}{\Pi^2*(AffinityFunPanel)})   (3)
$$

#### 公式(2)实现方法：

​	论文中表示对于所有的u,v都要求差，所以定义了如下函数

```python
def commonDiffMatrixInit(featureList1, featureList2):

    featureList1 = np.array(featureList1)

    featureList2 = np.array(featureList2).reshape(-1, 1)   #n行1列
    tmp =np.zeros(len(featureList2),len(featureList2))

    it = np.nditer([featureList1, featureList2, tmp],
                   [],
                   [['readonly'], ['readonly'], ['writeonly']])

    subOp = np.subtract   # 减法

    for x, y, z in it:
        subOp(x, y, out = z)

    return abs(it.operands[2])  #todo 平方和再开根号
```

利用其中 np.nditer()函数可以做到两个向量遍历相减得到矩阵。**todo 这个矩阵应该叫啥名呢**

通过对labeld的特征得到这个矩阵，对真实AQI分布进行拟合（线性回归），可以确定亲和函数中的a和b，得到这个a和b后，”AffinityWeight“矩阵就确定了，其中该矩阵即对应论文中的W
$$
W = \left[
\matrix{
  W_{vv} & W_{vu}\\
  W_{uv} & W_{uu}\\
}
\right]
$$
其中的每一个w，都是对应两个点的特征值相减，然后乘以系数a，再加b



#### 公式（3）实现方法：

```python
def weightMatrixUpdate(featureWeightPanel, AffinityFunPanel):

    tempMatrixDict = OrderedDict()

    for feature in featureWeightPanel.items:
        tempMatrixDict[feature] = featureWeightPanel[feature] * \
                                  featureWeightPanel[feature] * \
                                  AffinityFunPanel[feature]

    tempWeightMatrix = - sum(tempMatrixDict.values())
    
    return tempWeightMatrix.applymap(math.exp)

```

pandas Dataframe 的乘法和加法都是对应元素相乘，并**不是矩阵**！



熵函数：
$$
H(P_u)=P(u)log(P(u))+(1-P(u))log(1-P(u))
$$
实现代码：

```python
def entropyFun(entity):
    
    if entity == 0.0 or entity == 1.0:
        return 0.0
    else:
        return entity * math.log(entity, 2) + (1-entity) * math.log(1-entity, 2)

def matrixEntropyFun(dataFrame):
    
    # transform the pandas dataFrame into numpy array
    matrix = dataFrame.T.values

    # the # of nodes = the first dim of matrix
    numOfNodes = matrix.shape[0]
    
    # apply the matrix opeartion function to entropyFun
    entityEntropyFun = np.vectorize(entropyFun, otypes=[np.float])

    return - ( np.sum( entityEntropyFun(matrix) ) ) / numOfNodes
```

其中利用np.vectorize()可以实现对其中每一个元素进行遍历操作一种函数，这种函数有用户自己定义。  np.sum() 返回一个整型数/浮点型数



**更新Π矩阵**只需要按照论文进行简单数学公式编写即可。
$$
Π_k=Π_k-2w_{u,v}·AF_{f_k}(\Delta f_k(u,v))·Π_k
$$

```python
def featureWeightPanelUpdate(featureWeightPanel, weightMatrix, AffinityFunPanel):
    
    #for feature in featureWeightPanel.items:
    for feature in ['rowCol']:
        featureWeightPanel[feature] = (1.0 - 2.0 * weightMatrix * \
                                       AffinityFunPanel[feature]) * \
                                       featureWeightPanel[feature]
```

这块操作的对象还是dataframe，所以是对应元素相乘。



计算P（u）
$$
P_U=(D_{UU}-W_{UU})^{-1}W_{UV}P_V
$$
这是根据那个啥拉普拉斯矩阵来的。

```python
def harmonicFun(weightMatrixRHS, labeledDistriMatrixRHS, unlabeledList):
    
    weightMatrix = np.array( weightMatrixRHS.T.values )
    print("weightMatrix:",weightMatrix)
    labeledDistriMatrix = np.array( labeledDistriMatrixRHS.T.values )

    # l: the # of labeled points
    l = labeledDistriMatrix.shape[0]

    # n: the total # of points
    n = weightMatrix.shape[0]

    # the graph Laplacian L = D - W
    LaplacianMatrix = np.diag( np.sum(weightMatrix, axis = 1) )  #todo 对角线元素
    LaplacianMatrix = np.subtract(LaplacianMatrix, weightMatrix)   #拉普拉斯矩阵 🔺=D-W
    
    # the harmonic function
    unlabeledDistriMatrix = -1 * np.dot( np.dot( inv( LaplacianMatrix[l:n:1, l:n:1] ),   #todo inv 求逆
                                                 LaplacianMatrix[l:n:1, 0:l:1] ),
                                         labeledDistriMatrix )

    return pd.DataFrame(unlabeledDistriMatrix.transpose(),  #todo np转置
                        index = range(unlabeledDistriMatrix.shape[1]),
                        columns = unlabeledList,
                        dtype = float)

```



算出P（U）后计算熵，根据两次迭代熵的差进行判断合适终止迭代，即可得到P（U）