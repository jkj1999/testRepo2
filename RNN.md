#### 序列模型

在时序模型中，当前数据跟之前观察到的数据是相关的

时序序列的数学建模

在时间t观察到$x_t$，那么得到T个不独立的随机变量$(x_1,...x_T) \sim p(X)$

由条件概率展开得到所有x的联合概率为$p(X)=p(x_1)*p(x_2|x_1)*p(x_3|x_1,x_2)*...*p(x_T|x_1,...,x_{T-1})$

对条件概率建模

$p(x_t|x_1,...,x_{t-1})=p(x_t|f(x_1,...x_{t-1}))$，右式f为对见过的数据建模，也称**自回归模型**（即数据和标签是同一个）

如何计算f？
		方案A：**马尔可夫假设**

​			假设当前数据只跟过去$\tau$个过去数据点相关，即使用观测序列$x_{t-1},...,x_{t-\tau}$，只需建模$f(x_{t-\tau},...,x_{t-1})$

​		方案B：**潜变量模型**

​			引入潜变量$h_t$来表示对过去观测的总结，即$h_t=f(x_1,...x_{t-1})$，则$x_t=p(x_t|h_t)$

​			<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230830102154119.png" alt="image-20230830102154119" style="zoom: 67%;" />

​		**RNN就是一种潜变量模型**

隐变量真实存在的变量；潜变量包括隐变量，也可以是不存在的变量



#### 语言模型

给定文本序列$x_1,\dots,x_T$，语言模型的目标是估计联合概率$p(x_1,\dots,x_T)$

应用包括：

​		做预训练模型（BERT，GPT）

​		生成文本，给定前面的词，不断使用$x_t \sim p(x_t|x_1,\dots,x_{t-1})$生成后续文本

​		判断多个序列哪个更常见

##### 统计语言建模

###### 使用计数建模语言模型

假设序列长度为2，预测

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230831113011168.png" alt="image-20230831113011168" style="zoom: 33%;" />

​	n为总词数，$$n(x),n(x,x')$$是单个单词和连续单词对出现的次数，该模型很容易拓展到更多长度的情况

当序列很长时，因为文本量不够大，序列的计数可能小于等于1，使用*马尔可夫假设*可以缓解这个问题

###### 马尔可夫模型与n元语法

如果$P(x_{t+1}|x_t,\dots,x_1)=P(x_{t+1}|x_t)$，则序列上的分布满足一阶马尔可夫性质。阶数越高，对应的依赖关系就越长。

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230831115030951.png" alt="image-20230831115030951" style="zoom:67%;" />

通常，涉及一个、两个和三个变量的概率公式分别被称为一元语法、二元语法和三元语法（对应$\tau=0，1，2$）。

n越大，模型精度越大，导致算法复杂度和空间存储也变大

使用时光机器数据集构建词表，对词频进行双对数统计，发现词频以一种明确的方式迅速衰减。 将前几个单词作为例外消除后，剩余的所有单词大致遵循双对数坐标图上的一条直线。

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230831145930581.png" alt="image-20230831145930581" style="zoom:80%;" />

单词的频率满足*齐普夫定律*(Zipf's law)，即第i个最常用单词的频率$n_i$为$n_i \propto \frac 1 {i^\alpha}$ ，等价于$log n_i =-\alpha logi+c$

因此通过计数统计和平滑来建模单词不可行，因为这样会大大高估尾部单词的频率，也就是不常用单词

使用二元语法、三元语法进行统计得到：

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230831152419017.png" alt="image-20230831152419017" style="zoom:80%;" />

图示可以得出：除一元语法外，单词序列似乎也遵循齐普夫定律，以及很多n元组很少出现，这使得拉普拉斯平滑非常不适合语言建模，因此作为替代，使用基于深度学习的模型。



#### 循环神经网络

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230831212647976.png" alt="image-20230831212647976" style="zoom:33%;" />

t时刻的输出$o(t)$是基于$h_t$的，$h_t$又是基于$x_{t-1}$和$h_{t-1}$的，计算损失时才可以看到$x_t$，比较$o(t)$和$x_t$的差异

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230831213041936.png" alt="image-20230831213041936" style="zoom:33%;" />

###### 困惑度perplexity

衡量一个语言模型的好坏可以用平均交叉熵：

$\pi =\frac 1n \sum_{i=1}^n-log\ p(x_t|x_{t-1}\dots)$

假设语料库大小为M，一次预测可以视为一次分类问题，即给语料库中的全部词一个概率。预测一个长度为n的语言序列，就是做n次分类，所以可以使用n次分类的平均交叉熵来衡量语言模型。

NLP使用困惑度$exp(\pi)$来衡量（largen scale），困惑度为1表示模型完美，无穷大时模型最差

###### 梯度裁剪

迭代计算T个时间步上的梯度，在反向传播过程中会产生长度为$O(T)$的矩阵乘法链，会导致数值不稳定，因此使用梯度裁剪预防梯度爆炸

​		如果梯度长度超过θ，那么将梯度长度降回θ：$g \leftarrow min(1,\frac{\theta}{||g||})g$

###### RNN应用

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230901163119817.png" alt="image-20230901163119817" style="zoom: 50%;" />





















