# Python 知识点

## python3:

### 一、读入文件（字符串）

```python
with open(labeledListFileName) as labeledListFile:
    labeledList = labeledListFile.read().splitlines()
```

读入类型为字符串，经过splitlines()后得到 list列表



另外一种玩法：

```python
def loadDataSet(filename):
    dataMat=[]
    label=[]
    f = open(filename)
    for line in f.readlines():
        lineArr= line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        label.append(float(lineArr[2]))
    return dataMat,label
```



### 二、笛卡尔积

```python
labeledList = list(itertools.product(timeStampList, labeledList))
```

如果不加list返回类型是<class 'itertools.product'>，需要转成list



### 三、利用循环列表生成

```
tempLabeledList = [ element for element in labeledList if element[0] == currentTimeStamp ]
```



### 四、随机函数

```python
rand_i=int(random.uniform(0,m))
```



### 五、有序字典

```
labeledAQIDict = OrderedDict()
```

添加新项目

```
labeledAQIDict.update(tempLabeledAQIDict)
```





## Numpy：

#### 这用法牛逼就完事了

```python
it = np.nditer([featureList1, featureList2, tmp],
               [],
               [['readonly'], ['readonly'], ['writeonly']])

subOp = np.subtract  # 减法

for x, y, z in it:
 #z[()] = np.subtract(x,y)
 subOp(x, y, out=z)
```



#### 矩阵拼接

```python
tempMatrix = np.vstack((np.hstack([funDict[funChosen](lList, lList),
                                   funDict[funChosen](uList, lList)]),
                        np.hstack([funDict[funChosen](lList, uList),
                                   funDict[funChosen](uList, uList)])))
```



#### vectorize函数向量化：

```python
def linearizeFun(entity, slope, intercept):
    
    return slope * entity + intercept
    
    
entityLinearizeFun = np.vectorize(linearizeFun, otypes=[np.float])

tempMatrix = entityLinearizeFun(tempMatrix, regressResult[0], regressResult[1])
```



#### 对于mat矩阵：矩阵乘法 np.dot(),如果如果是用* np.multiply() 的话就是对应元素相乘（点乘）：

```
e=np.array([[1,2,3,4,5],[1,2,3,4,5]])
f=np.array([[1,2],[1,2],[1,2],[1,2],[1,2]])
g=np.dot(e,f)
```

详情参见https://blog.csdn.net/zenghaitao0128/article/details/78715140

总结 对于array： multiply 、 * 对应位置相乘， 一维数组 dot是对应位置， 二维是矩阵乘法

​         对于mat ： multiply是对应位置相乘， * 和 dot 都是矩阵乘法

### np.mat和np.array的异同

```python
gamma = np.zeros((3,4))
gamma_mat = np.mat(gamma)
print(gamma);print(gamma[:,2])
```

```
gamma:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
gamma[:,2]
[0. 0. 0.]
gamma_mat
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
gamma_mat[:,2]
[[0.]
 [0.]
 [0.]]
```

**调出列后mat保持列向量不变shape（3，1） array转为（1，3）了**

填充的时候也需要保持array填充为一行，mat填充为一列。

```python
pi1 = np.array([1,2,3])
pi2 = np.array([[1],[2],[3]])

prob[:,2] = pi1
print(prob)

prob_mat[:,2] = pi2
print(prob_mat)

（1）
[[0. 0. 1. 0.]
 [0. 0. 2. 0.]
 [0. 0. 3. 0.]]
（2）
[[0. 0. 1. 0.]
 [0. 0. 2. 0.]
 [0. 0. 3. 0.]]
（3）
```

https://blog.csdn.net/weixin_44898235/article/details/100730721

### np.nonzero(a)

返回数组a中非零元素的索引值数组。a4 (array([1, 2], dtype=int64),) 这块是列表组成的数组，**需要reshape才可以使用，或者取[0]**

```python
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]       

```



### mat[i].A   np的矩阵的第i行转数组array



### 浅拷贝与深拷贝  (修改mat矩阵需要保存旧值的时候需要使用.copy（)

```
a=np.mat([[1,2,3],[5,6,7]])
tt=a.copy()  这样a被修改tt的值不会变
```



## Pandas：

### 一、读入文件（dataframe）

```python
labeledAQITable = pd.read_csv(labeledAQITableFileName)
```

### 二、创建Dataframe

```python
rankTable = pd.DataFrame([ [-1] * unlabeledListLen ] * len(timeStampList),
                         index = timeStampList,
                         columns = unlabeledList)
```

index和columns传入list列表

从dataframe中取出value  加.value() ,然后 .ravel() 得到 numpy数组



### 三、删除列

```
raw_gps_frame.drop(['Unnamed: 4'], axis=1, inplace=True)
```



## Plot





## math&stats

```
regressResult = stats.linregress(labeledFeatureDiffArray, labeledAQIDiffArray)
```

线性拟合函数得到a和b 还有

| Parameters: | **x, y** : array_liketwo sets of measurements. Both arrays should have the same length. If only x is given (and y=None), then it must be a two-dimensional array where one dimension has length 2. The two sets of measurements are then found by splitting the array along the length-2 dimension. |
| :---------- | :----------------------------------------------------------- |
| Returns:    | **slope** : floatslope of the regression line**intercept** : floatintercept of the regression line**r-value** : floatcorrelation coefficient**p-value** : floattwo-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.**stderr** : floatStandard error of the estimate |





### XML