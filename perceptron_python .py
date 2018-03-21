
# coding: utf-8

# In[335]:


import numpy as np
class Perceptron:
    '''
    Parameters
    __________
    eta : learning rate
    n_iter : 迭代次数
    random_state : 初始化权值 w
    __________
    '''
    #initialization 初始化学习率，迭代次数，权值w
    def __init__(self,eta = 0.01,n_iter = 50,random_state = 1 ):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    #fit training data 拟合训练数据
           
    '''
    X : training data [n_samples,n_features]
       输入的训练数据集，eg.[(1,2,3),(4,5,6)],即有两个样本，每个样本有三个特征值
    
    y: target values [n_samples] 每个样本属于的分类 y = {+1 , -1}
    
    size = 1+X.shape[1]  加1原因w[0]代表偏置b,将偏置b并入权值向量

    '''
    def fit(self,X,y):
        
 #RandomState 随机数生成器 RandomState(seed),如果seed相同，产生的伪随机数相同
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0,scale = 0.01,size = 1+X.shape[1]) #loc:均值，scale：方差 size:输出规格
        self.errors_ = []
        
        for _ in range(self.n_iter): #进行n_iter次迭代
            errors = 0
            for xi,target in zip(X,y):
                
                '''
                1. update = eta * ( y - y')   y' 即 predict（xi）
                2. zip(X,y) 输出(1,2,3,1),(4,5,6,-1) 
                3. X.shape[1] 即输出 n_sample 的值
                4. update * xi ->  ▽w(1)=X[1]*update,▽w(2)=X[2]*update,▽w(3)=X[3]*update  即权值w的更新
                5. w_[1:] += update * xi --> 从w[1]开始 进行更新，即每个加上增量
                6. 更新W[0] ，即更新偏置b
                7. 统计错误量，errors_ 
                '''
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0) #如果分类正确，y-y'=0，update=0  
            self.errors_.append(errors)
        return self
    #计算净输入 即x与w的内积 即X[i]乘W[i]，之后相加求和 return x1*w1+x2*w2+...+xn*wn
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    # self.net_input(X)即X与W的内积，如果大于0 ，则为类别 1，否则为 -1 
    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
                
         


# In[333]:


#为了说明 不将权值w全部初始化为0的原因。
#w 不全为0，更新后w值各不相同，点乘X后可以影响向量X的角度和长度
# 如果w全部初始为0，那么相当于向量X点乘一个常数，如下0.5*v2, 只能影响X的长度，不能改变角度
v1 = np.array([1,2,3]) 
v2 = 0.5 * v1
#  v1.dot(v2) / np.linalg.norm(v1)*np.linalg.norm(v2) 即向量的内积/两向量的模长 即两向量的夹角
# arccos 反余弦 即两个向量的夹角为0
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# #  Training a perceptron model on the Iris dataset
# 

# In[334]:


# load iris dataset
import pandas as pd
df = pd.read_csv("G:\Machine Learning\python machine learning\python-machine-learning-book-2nd-edition-master\code\ch02\iris.data",header = None)
df.tail()


# 显示要进行分类的数据集 

# In[336]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# select setosa and versicolor
#iloc()提取某列 ----前100行的第四列，第四例为versicolor和setosa属性
y = df.iloc[0:100,4].values 
#进行分类   1 ( versicolor ) and -1 ( setosa ）
y = np.where(y == 'Iris-setosa',-1, +1)  

# first feature column (sepal length) and the third feature column (petal length)  花瓣长度和花萼长度
X = df.iloc[0:100,[0,2]].values  

#plot data
#前50是setosa，属于类别 -1 
plt.scatter(X[:50,0],X[:50,1],color = 'red',marker = 'o',label = 'setosa') 
#后50行是versilor，属于类别 +1 
plt.scatter(X[50:100,0],X[50:100,1],color = 'blue',marker = 'x',label = 'versilor')


plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')

plt.legend(loc = 'upper left')
plt.show()




# In[344]:


#train perceptron alogrithm
ppn = Perceptron(eta = 0.1 ,n_iter = 10)
#fit函数即继续数据拟合，也就是进行分类的迭代过程
ppn.fit(X,y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker = 'o')

plt.xlabel('Epochs')
plt.ylabel('Number of updates')


# 画超平面

# In[352]:


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    #具体函数 在博文（一）中有记录  博文地址http://blog.csdn.net/Amy_mm/article/details/79625288
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface 画分离超平面
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    #生成间隔为resolution的网格
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    #调用感知机分类函数进行预测
    #np.array([xx1.ravel(),xx2.ravel()]).T 为*行两列的矩阵，对应二维平面上xx1,xx2行程的网格点
    #对这些网格点进行分类为+1，-1,
    #画出等高线，+1类画一种颜色，-1另一种
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    #contour()函数（一）中有记录
    plt.contourf(xx1,xx2,Z,alpha = 0.3,cmap = cmap)
    
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    # plot class samples 画出样本点
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], 
                    y = X[y == cl, 1],
                    alpha = 0.8, 
                    c = colors[idx], 
                    marker = markers[idx],  
                    label = cl,
                    edgecolor = 'black')
    


# In[353]:


#画图
plot_decision_regions(X,y,classifier = ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show


# 可以看出 能够进行正确分类
