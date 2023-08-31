from torch import nn

# 编解码器的基本接口
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    # 指定长度可变的序列作为编码器的输入X
    def forward(self, X):
        raise NotImplementedError


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

    # 将编码器的输出转换为编码后的状态,用于解码器对其进行处理
    def init_state(self, enc_outputs):
        raise NotImplementedError

    # 解码器可以有自己额外的输入X
    def forward(self, X, state):
        raise NotImplementedError

# 合并编解码器
class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        dec_outputs = self.decoder(dec_X, dec_state)
        return dec_outputs

