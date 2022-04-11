from datetime import datetime
import logging
import sys
import os
import torch


def logger_init(log_file_name='monitor',
                log_level=logging.DEBUG,
                log_dir='./logs',
                only_file=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                            format=formatter,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),
                                      logging.StreamHandler(sys.stdout)])


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    '''
    用padding补齐sequence
    :param sequences:
    :param batch_first:
    :param max_len:
    :param padding_value:
    :return:
    '''
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            # 补齐
            tensor = torch.cat([tensor, torch.tensor(
                [padding_value] * (max_len - tensor.size(0))
            )], dim=0)
        else:
            # 截断
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


class Vocab:
    '''
    建立词表
    '''
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}  # str --> id
        self.itos = []  # id --> str
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, item):
        return self.stoi.get(item, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    return Vocab(vocab_path)

if __name__ == '__main__':
    vocab = build_vocab('./bert-base-uncased/vocab.txt')
    print(vocab.stoi['[SEP]'])
    print(vocab)