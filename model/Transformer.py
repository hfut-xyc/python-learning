# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size, 1, L, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size, len_q, len_k

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.W_O = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, mask):
        # Q, K, V: [batch_size, L, d_model]
        residual = Q
        batch_size = Q.size(0)
        # (B, L, d_model) -linear-> (B, L, d_k * n_heads) -split-> (B, L, n_heads, d_k) -trans-> (B, n_heads, L, d_k)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(d_k)

        # mask: [batch_size, 1, L, L] -> [batch_size, 8, L, L]
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, n_heads, 1, 1) 
        # Fills elements of self tensor with value where mask is one.
        scores.masked_fill_(mask, -1e9) 

        attn = nn.Softmax(dim=-1)(scores)   
        output = torch.matmul(attn, v_s)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) 
        output = self.W_O(output)
        return self.layer_norm(output + residual), attn 

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):   # x : [batch_size, L, d_model]
        residual = x 
        output = nn.ReLU()(self.conv1(x.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ffn = FeedForward()

    def forward(self, x, mask):
        # x to same Q,K,V
        output, attn = self.attn(x, x, x, mask)
        # output: [batch_size, L, d_model] 
        output = self.ffn(output) 
        return output, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = MultiHeadAttention()
        self.attn2 = MultiHeadAttention()
        self.ffn = FeedForward()

    def forward(self, dec_inputs, enc_outputs, mask1, mask2):
        dec_outputs, dec_self_attn = self.attn1(dec_inputs, dec_inputs, dec_inputs, mask1)
        dec_outputs, dec_enc_attn = self.attn2(dec_outputs, enc_outputs, enc_outputs, mask2)
        dec_outputs = self.ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): 
        # enc_inputs : [batch_size, source_len]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size, target_len]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        
        pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        self_attn_mask = torch.gt((pad_mask + subsequent_mask), 0)
        
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=False)
    
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.linear(dec_outputs)   # [B, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()

def make_batch(sentences):
    enc_input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    dec_input_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(enc_input_batch), torch.LongTensor(dec_input_batch), torch.LongTensor(target_batch)

if __name__ == '__main__':
    # Transformer Parameters
    # Padding Should be Zero
    
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    src_vocab_size = len(src_vocab)    
    tgt_vocab_size = len(tgt_vocab)
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}

    src_len = 5 # length of source
    tgt_len = 5 # length of target

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)