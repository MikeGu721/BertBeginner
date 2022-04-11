import torch
import os
import logging
import math
from copy import deepcopy
import torch.nn as nn
from models.MultiHeadAttention import MyMultiheadAttention


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id=0, initializer_range=0.02):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        # self._reset_parameters(initializer_range)

    def forward(self, input_ids):
        return self.embedding(input_ids)

    # def _reset_parameters(self, initializer_range):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             no


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings=512, initializer_range=0.02):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, position_ids):
        return self.embedding(position_ids).transpose(0, 1)


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size, initializer_range=0.02):
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)

    def forward(self, token_type_ids):
        return self.embedding(token_type_ids)


class BertEmbedding(nn.Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        self.word_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            initializer_range=config.initializer_range
        )
        # return shape [src_Len, batch_size, hidden_size]

        self.position_embedding = PositionalEmbedding(
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range
        )
        # return shape [src_len, 1, hidden_size]

        self.token_type_embedding = SegmentEmbedding(
            type_vocab_size=config.type_vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range
        )
        # return shape [src_len, batch_size, hidden_size]

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids',
                             torch.arange(config.max_position_embeddings).expand((1, -1)))
        # return [1, max_position_embedding]

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None):
        src_len = input_ids.size(0)
        token_embedding = self.word_embedding(input_ids)
        # shape: [src_len, batch_size, hidden_size]

        if position_ids is None:
            position_ids = self.position_ids[:, :src_len]
        positional_embedding = self.position_embedding(position_ids)
        # [src_len, 1, hidden_size]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, device=self.position_ids.device)
        segment_embedding = self.token_type_embedding(token_type_ids)
        # [src_len, batch_size, hidden_size]

        embedidngs = token_embedding + positional_embedding + segment_embedding
        embeddings = self.LayerNorm(embedidngs)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    '''
    Multi_Head_Attention
    '''

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.multi_head_attention = MyMultiheadAttention(embed_dim=config.hidden_size,
                                                         num_heads=config.num_attention_heads,
                                                         dropout=config.attention_probs_dropout_prob)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return self.multi_head_attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class BertSelfOutput(nn.Module):
    '''
    dropout + resnet + layer_normalization
    '''

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    '''
    BertSelfAttention + BertSelfOutput
    '''

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, hidden_states, hidden_states, attn_mask=None,
                                 key_padding_mask=attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class BertIntermediate(nn.Module):
    '''
    dense layer
    '''

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = GELU()
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = get_activation(config.hidden_act)
        # else:
        #     self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        if self.intermediate_act_fn is None:
            hidden_states = hidden_states
        else:
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''

        :param hidden_states: BertIntermediate的输出
        :param input_tensor: BertAttention的输出
        :return:
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    '''
    BertAttention + BertIntermediate + BertOutput
    '''

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.bert_attention = BertAttention(config)
        self.bert_intermediate = BertIntermediate(config)
        self.bert_output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.bert_attention(hidden_states, attention_mask)
        intermediate_output = self.bert_intermediate(attention_output)
        layer_output = self.bert_output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        all_encoder_layers = []
        layer_output = hidden_states
        for i, layer_module in enumerate(self.bert_layers):
            layer_output = layer_module(layer_output, attention_mask)
            all_encoder_layers.append(layer_output)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        if self.config.pooler_type == 'first_token_transform':
            token_tensor = hidden_states[0, :].reshape(-1, self.config.hidden_size)
        elif self.config.pooler_type == 'all_token_average':
            token_tensor = torch.mean(hidden_states, dim=0)
        pooled_output = self.dense(token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.bert_embedding = BertEmbedding(config)
        self.bert_encoder = BertEncoder(config)
        self.bert_pooler = BertPooler(config)
        self.config = config

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        embedding_output = self.bert_embedding(input_ids=input_ids, position_ids=position_ids,
                                               token_type_ids=token_type_ids)
        all_encoder_outputs = self.bert_encoder(embedding_output, attention_mask=attention_mask)
        sequence_output = all_encoder_outputs[-1]
        pooled_output = self.bert_pooler(sequence_output)
        return pooled_output, all_encoder_outputs

    @classmethod
    def from_pretrained(cls, config, pretrained_model_dir=None):
        model = cls(config)
        pretrained_model_path = os.path.join(pretrained_model_dir, 'pytorch_model.bin')
        loaded_paras = torch.load(pretrained_model_path)
        state_dict = deepcopy(model.state_dict())
        '''
        deepcopy: 复制为一个新的个体
        copy: 将新变量的内存指向老变量
        '''
        loaded_paras_names = list(loaded_paras.keys())[:-8]  # 最后8个为cls的输出
        model_paras_names = list(state_dict.keys())[1:]  # 第一个
        for i in range(len(loaded_paras_names)):
            state_dict[model_paras_names[i]] = loaded_paras[loaded_paras_names[i]]
            logging.info(f'成功将参数{loaded_paras_names[i]}赋值给{model_paras_names[i]}')
        model.load_state_dict(state_dict)
        return model


if __name__ == '__main__':
    import Config
    from transformers import BertTokenizer

    model_config = Config.ModelConfig(bert_name='bert-base-chinese', dataset_name='SQuAD-Chinese', task='MRC')
    tokenizer = BertTokenizer.from_pretrained(model_config.pretrained_model_dir)
    model = BertModel.from_pretrained(model_config, model_config.pretrained_model_dir)
    text = '无可奈何花落去，似曾相识燕归来'
    tokens = tokenizer(text)['input_ids']
    tokens = torch.tensor(tokens, dtype=torch.long)
    print(tokens)
    embedding, two = model(tokens)
    print(embedding.shape)
    print(two.shape)
