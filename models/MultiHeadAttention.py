import torch
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn


class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, training=True):
        super(MyMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.kdim = self.head_dim
        self.vdim = self.head_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert self.head_dim * num_heads == self.embed_dim
        self.q_proj_weight = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj_weight = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj_weight = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.training = training

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return self.multi_head_attention_forward(query, key, value, self.num_heads,
                                                 self.dropout, self.out_proj.weight,
                                                 self.out_proj.bias, training=self.training,
                                                 key_padding_mask=key_padding_mask,
                                                 attn_mask=attn_mask)

    def multi_head_attention_forward(self, query, key, value, num_heads, dropout, out_proj_weight, out_proj_bias,
                                     training=True, key_padding_mask=None, attn_mask=None):
        # 第一部分
        q = self.q_proj_weight(query)
        k = self.k_proj_weight(key)
        v = self.v_proj_weight(value)
        # q = F.linear(query, q_proj_weight)
        # k = F.linear(key, k_proj_weight)
        # v = F.linear(value, v_proj_weight)
        # 第二部分
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('2维attn_mask的size错误')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('3维attn_mask的size错误')

        # 第三部分
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        # 第四部分
        if attn_mask is not None:
            attn_output_weights += attn_mask
        if key_padding_mask is not None:
            # 多头在这里开始分开了
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2),
                                                                  float('-inf'))
            # 多头合并
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=dropout, training=training)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # 多头在这里开始分开了
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

        Z = F.linear(attn_output, out_proj_weight, out_proj_bias)
        return Z, attn_output_weights.sum(dim=1) / num_heads


if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    dmodel = 32
    num_head = 4
    src = torch.randn((src_len, batch_size, dmodel))
    src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                         [True, True, True, True, False]])
    my_mh = MyMultiheadAttention(embed_dim=dmodel, num_heads=num_head)
    print(src.shape)
    r = my_mh(src, src, src, key_padding_mask=src_key_padding_mask)
    print(r[0].shape, r[1].shape)
