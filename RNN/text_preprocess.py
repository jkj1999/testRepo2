# 文本可以看作一串单词序列或字符序列

import collections
import re
from d2l import torch as d2l

# 1.读取数据集
# 加载文本，并将数据集读取到由多条文本行组成的列表中
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 把非英文字符的字符全部变成空格,并将字符全部变成小写  strip()默认去除字符串两侧的空白字符,留下中间的文本
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])


# 2.词元化
# 将每一行即每个文本序列拆分成一个词元列表，词元(token)是文本的基本单位
# 定义tokenize函数,默认将每一行由单词作为基本单位拆分,也可以选择字符作为基本单位拆分
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


# tokens是保存每行文本的token的list
tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


# 3.词表
# 建立词表,将拆分的词元映射到数字索引
# 词元是字符串类型,模型输入需要的是数字,因此需要构建一个字典,通常也叫做词表(vocabulary),将字符串类型的词元映射到从0开始的数字索引中
# 先将训练集中的所有文档合并,对它们的唯一词元进行统计,得到的统计结果称之为语料(corpus)
# 根据每个唯一词元的出现频率,为其分配一个数字索引
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        # key指定排序按照键值对的第二个元素即值进行排序
        # reverse=True表示降序排序
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)

        # 先将unk和保留token添加到映射的前面
        # 索引到词汇的映射(只有token的list)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 词汇到索引的映射(token:idx键值对的dict)
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        # 两个映射是为了方便将文本转换为索引以供模型使用,以及生成输出时将模型的输出索引转换回文本

        for token, freq in self._token_freqs:
            # 如果token的出现频率小于min_freq,移除该词元,降低复杂性
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    # 返回值是字典类型,键是token,值是token频率
    return collections.Counter(tokens)


# 使用加载的数据集作为语料库构建词表
reserved_tokens = ['bos', 'eos']
vocab = Vocab(tokens,reserved_tokens=reserved_tokens)
print(list(vocab.token_to_idx.items())[:10])

# 查看每一条文本行转换成的数字索引列表
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])


# 4.整合功能
# 函数返回corpus(词元索引列表)和vocab(语料库的词表)
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    # 使用字符实现文本词元化
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# corpus为将原始数据中的字符全部转换为对应的token的总和
corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)