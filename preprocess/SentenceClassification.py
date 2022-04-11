import tqdm
import torch
from torch.utils.data import DataLoader
from pytorch_transformers import BertTokenizer
import gzh
from utils import build_vocab, pad_sequence



class LoadSingleSentenceClassificationDataset:
    '''
    根据训练语料完成字典的构建，并将数据预处理后存入DataLoader
    '''

    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\n',
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True):
        self.vocab = build_vocab(vocab_path=vocab_path)
        self.tokenizer = tokenizer
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embedding = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle

    def data_process(self, filepath):
        '''
        将训练集、测试集、验证集转换成Token序列
        :param filepath:
        :return:
        '''
        raw_iter = open(filepath, encoding='utf-8').readlines()
        data = []
        max_len = 0
        label2num = {}
        index = 0
        for raw in tqdm.tqdm(raw_iter, ncols=80):
            line = raw.rstrip('\n').split(self.split_sep)
            l, s = line[0], '\t'.join(line[1:])
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer.tokenize(s)]
            if len(tmp) > self.max_position_embedding - 1:
                tmp = tmp[:self.max_position_embedding - 1]
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            if l not in label2num:
                label2num[l] = len(label2num)
            l_index = label2num.get(l)
            l = torch.tensor(int(l_index), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        print('总共有%d个label' % len(label2num))
        return data, max_len, len(label2num)

    def generate_batch(self, data_batch):
        '''
        对每个batch的token进行padding
        :param data_batch:
        :return:
        '''
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label

    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None):
        '''
        预处理数据并得到DataLoader
        :param train_file_path:
        :param val_file_path:
        :param test_file_path:
        :param only_test:
        :return:
        '''
        if test_file_path:
            test_data, max_sen_len, label_size = self.data_process(test_file_path)
            self.max_sen_len = max_sen_len
            test_iter = DataLoader(test_data, batch_size=self.batch_size,
                                   shuffle=False, collate_fn=self.generate_batch)
        else:
            test_iter = None
        if val_file_path:
            val_data, max_sen_len, label_size = self.data_process(val_file_path)
            self.max_sen_len = max_sen_len
            val_iter = DataLoader(val_data, batch_size=self.batch_size,
                                  shuffle=False, collate_fn=self.generate_batch)
        else:
            val_iter = None
        if train_file_path:
            train_data, max_sen_len, label_size = self.data_process(train_file_path)
            self.max_sen_len = max_sen_len
            train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                    shuffle=True, collate_fn=self.generate_batch)
        else:
            train_iter = None

        return train_iter, val_iter, test_iter, label_size








if __name__ == '__main__':
    import Config

    model_config = Config.ModelConfig(bert_name='bert-base-chinese', dataset_name='SogouNews-Chinese')
    load_dataset = LoadSingleSentenceClassificationDataset(
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(gzh.bert_base_chinese_dir),
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        split_sep=model_config.split_sep,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_index=model_config.pad_token_id,
        is_sample_shuffle=model_config.is_sample_shuffle
    )

    train_iter, val_iter, test_iter, label_size = load_dataset.load_train_val_test_data(model_config.train_file_path,
                                                                            model_config.val_file_path,
                                                                            model_config.test_file_path)

    for sample, label in test_iter:
        print(sample.shape)
        print(sample.transpose(0, 1))
        padding_mask = (sample == load_dataset.PAD_IDX).transpose(0, 1)
        print(padding_mask)
        print(label)
        break
