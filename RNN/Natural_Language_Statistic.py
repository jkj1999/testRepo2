import math
import random
import torch
from d2l import torch as d2l


# 马尔可夫模型

# 一元语法
# token将原始文本分割成基本单位(有重复单位)
tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此把所有文本行拼接到一起形成corpus
corpus = [token for line in tokens for token in line]
# vocab中的token才是唯一的,corpus中不唯一
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])
print(list(vocab.token_to_idx.items())[:10])
# the、that、and这些最常用的词被称为停用词(stop words)

freqs = [freq for token, freq in vocab.token_freqs]
# freqs = list of freq
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
d2l.plt.show()

# 二元语法
# 将原始文本中的相邻词汇两两组合  (len(bigram_tokens) = len(corpus) - 1
# zip将列表组合，创建一个迭代器，其中每个元素是每个输入列表中的相邻元素组成的元组
# [:-1]从第一个元素开始，到倒数第二个元素结束，不包括倒数第一个元素
# [1:]从第二个元素开始，到倒数第一个元素结束，不包括第一个元素
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

# 三元语法
# 将原始文本中的相邻三个词汇组合  (len(bigram_tokens) = len(corpus) - 2
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
d2l.plt.show()


# 读取长序列数据
# 与先前的序列模型(使用tau)不一样，此次读取，每个词元仅出现一次

# 随机采样分区
# 在原始的长序列上任意捕获长度为num_steps的子序列，在迭代过程中，来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
# 对于语言建模，目标是基于到目前为止我们看到的词元来预测下一个词元， 因此标签是移位了一个词元的原始序列。
# 假设时间步长为4,特征:token1,2,3,4 标签:token2,3,4,5(特征和标签的长度相同,标签是特征向后移位一个词元)
def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 使用随机偏移量使得起始位置是随机的,这样既能做到覆盖性,又能做到随机性
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # “特征-标签”子序列对数量，减去1是因为需要考虑标签 //是整数除法操作符，将除法结果向下取整
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的每个子序列的起始下标
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # yield是用于创建生成器的关键字,将一个普通函数转变为生成器函数,在每次请求时生成一个值并保持状态
        yield torch.tensor(X), torch.tensor(Y)

my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


# 顺序采样分区
# 两个相邻的小批量中的子序列在原始序列上也是相邻的,即batch1和batch2中的每对子序列都在原始序列上相邻
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab