import json
import logging
import six
import torch
import os
from utils import logger_init


class ModelConfig:
    def __init__(self, bert_name, dataset_name=None, task='Classification'):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.pretrained_model_dir = os.path.join(self.project_dir, bert_name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.project_dir, 'DataSet', task, dataset_name, 'train.txt')
        self.val_file_path = os.path.join(self.project_dir, 'DataSet', task, dataset_name, 'val.txt')
        self.test_file_path = os.path.join(self.project_dir, 'DataSet', task, dataset_name, 'val.txt')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = '\t'
        self.is_sample_shuffle = True
        self.batch_size = 16
        self.max_sen_len = 512
        self.learning_rate = 1e-5
        self.num_labels = 10  # 需要改变
        self.epochs = 10
        self.model_save_per_epoch = 1
        self.max_position_embeddings = 512
        self.pad_token_id = 0
        self.dataset_name = dataset_name

        self.reset_logger('same')

        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.bert_config_path = os.path.join(self.pretrained_model_dir, 'config.json')
        self.model_save_dir = self.pretrained_model_dir
        bert_config = BertConfig.from_json_file(self.bert_config_path)
        bert_config.pretrained_model_path = self.pretrained_model_dir
        bert_config.vocab_path = self.vocab_path
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        logging.info(' ### 将当前配置打印至日志文件 ###')
        for key, value in self.__dict__.items():
            logging.info(f'### {key} = {value}')

    def reset_logger(self, log_file_name):
        logger_init(log_file_name=log_file_name, log_level=logging.INFO,
                    log_dir=os.path.join(self.project_dir, 'logs', self.dataset_name))


class BertConfig:
    def __init__(self,
                 vocab_size=21128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 intermediate_size=3072,
                 pad_token_id=0,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pretrained_model_path='./bert-base-chinese/pytorch_model.bin',
                 vocab_path='./bert-base-chinese/vocab.txt'):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pretrained_model_path = pretrained_model_path
        self.vocab_path = vocab_path

    @classmethod
    def from_dict(cls, json_object):
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, 'r') as reader:
            text = reader.read()
        logging.info(f'成功导入BERT配置文件 {json_file}')
        return cls.from_dict(json.loads(text))
