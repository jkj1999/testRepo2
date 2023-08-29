### Attention Mechanism

###### 注意力提示

卷积、全连接、池化都只考虑不随意线索，只使用非自主性的提示，选择偏向于感官输入。

注意力机制则显式的考虑随意线索，包含**自主性的提示**，自主性提示被称为**查询**(query)，给定任何查询，注意力机制通过*注意力汇聚*（attention pooling） 将选择引导至*感官输入*（sensory inputs，例如中间特征表示）。 在注意力机制中，这些感官输入被称为*值*（value）。 更通俗的解释，每个值都与一个*键*（key）配对， 这可以想象为感官输入的非自主提示。

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230824150141780.png" alt="image-20230824150141780" style="zoom:80%;" />

注意力机制通过注意力汇聚(Attention Pooling)将query和key结合，实现对value的选择倾向。

在实际的模型中，key和value一般都是中间的特征表示：

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230825103238686.png" alt="image-20230825103238686" style="zoom: 67%;" />

###### 非参的注意力池化层

最简单的方案：平均池化![image-20230824152132315](C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230824152132315.png)，忽略了输入$x$

更好的方案：Nadaraya-Watson核回归![image-20230824151911348](C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230824151911348.png)

​		其中$x$为query，$x_i$为key，$y_i$为value，$K$是核函数，衡量$x$和$x_i$之间的距离。

​		思想是根据输入的位置对输出$y_i$进行加权。

​		如果使用高斯核K<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230824155520609.png" alt="image-20230824155520609" style="zoom: 67%;" />，代入$f(x)$发现是对距离执行softmax后对$y_i$加权

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230824155553005.png" alt="image-20230824155553005" style="zoom:67%;" />

###### 参数化的注意力机制

在之前基础上引入模型可以学习的w，将$query:x$与$key:x_i$之间的距离乘以可学习参数w

注：在下式中可学习参数w是**一个标量值**

<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230824155453997.png" alt="image-20230824155453997" style="zoom: 67%;" />

###### 注意力权重与注意力分数

注意力汇聚函数f可以表示为$f(q,(k_1,v_1),(k_2,v_2),...(k_m,v_m)=\sum \alpha(q,k_i)v_i=\sum softmax(a(q,k_i))v_i$

$a(q,k_i)$为注意力分数，也称为评分函数，在上述式子里为$-\frac{1}{2}(x-x_i)^2$，是query和key的相似度；

$\alpha(q,k_i)$为注意力权重，是注意力分数的softmax结果（对注意力分数进行了normalization）

###### 评分函数的设计

- Additive Attention 加性注意力

  当查询和键是**不同长度的矢量时**

   给定查询q和键k， 加性注意力（additive attention）的评分函数为<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230829103535968.png" alt="image-20230829103535968" style="zoom:67%;" />

  其中可学习的参数是<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230829103720998.png" alt="image-20230829103720998" style="zoom:67%;" />

  本质是将查询和键连结起来后输入到一个多层感知机（MLP）中， 感知机包含一个隐藏层，其隐藏单元数是一个超参数ℎ。 通过使用tanh作为激活函数，并且禁用偏置项。（**将key和value变换到相同维度空间再相加**）

- Scaled Dot-Product Attention 缩放点积注意力

  当查询和键**具有相同的长度d**

  假设查询和键的所有元素都是满足零均值和单位方差的独立随机变量，那么两个向量的点积的均值为0，方差为d，为保证点积方差为1，评分函数为查询和键的点积除以$\sqrt d$，即<img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230829104638333.png" alt="image-20230829104638333" style="zoom:67%;" />

  向量化版本，从小批量角度提高效率， 例如基于n个查询和m个键-值对计算注意力

  <img src="C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230829105239401.png" alt="image-20230829105239401" style="zoom:67%;" />

###### masked softmax 掩蔽softmax

softmax操作用于输出一个概率分布作为注意力权重。 在某些情况下，并非所有的值都应该被纳入到注意力汇聚中。 为了仅将有意义的作为值来获取注意力汇聚， 可以指定一个有效序列长度， 以便在计算softmax时过滤掉超出指定范围的位置，设计了masked_softmax函数，其中任何超出有效长度的位置都被掩蔽并置为0。

```python
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```python
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

```python
tensor([[[0.5980, 0.4020, 0.0000, 0.0000],
         [0.5548, 0.4452, 0.0000, 0.0000]],
         
        [[0.3716, 0.3926, 0.2358, 0.0000],
         [0.3455, 0.3337, 0.3208, 0.0000]]])
```



### Encoder-Decoder

编解码器架构

![image-20230829162611192](C:\Users\Junlin Li\AppData\Roaming\Typora\typora-user-images\image-20230829162611192.png)

编码器：将输入编码为中间表达形式（特征）

解码器：将中间表示解码为输出

CNN就可以看成是一个E-D架构，前面的layer负责特征抽取作为编码器，最后一层的softmax回归输出作为解码



### Self-Attention

