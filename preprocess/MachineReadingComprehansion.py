from preprocess.SentenceClassification import LoadSingleSentenceClassificationDataset
import json
import torch
import tqdm
import logging
import os
from torch.utils.data import DataLoader
from utils import pad_sequence


def cache(func):
    '''
    把 data_process 写入文件放入缓存
    :param func:
    :return:
    '''

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f'缓存文件{data_path} 不存在，重新处理并缓存！')
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f'缓存文件 {data_path} 存在，直接载入缓存文件！')
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


def get_format_text_and_word_offset(text):
    '''
    拆解每个字符
    :param text:
    :return:
    '''

    def is_whitespace(c):
        if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F:
            return True
        return False

    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset


def improve_answer_span(context_tokens, answer_tokens, start_position, end_position):
    '''
    修正起止位置
    :param context_tokens: tokenize之后的context
    :param answer_tokens: tokenize之后的answer
    :param start_position:
    :param end_position:
    :return:
    '''
    new_end = None
    for i in range(start_position, len(context_tokens)):
        if context_tokens[i] != answer_tokens[0]:
            continue
        for j in range(len(answer_tokens)):
            if answer_tokens[j] != context_tokens[i + j]:
                break
            new_end = i + j
        if new_end - i + 1 == len(answer_tokens):
            return i, new_end
    return start_position, end_position


class LoadSQuADQuestionAnsweringDataset(LoadSingleSentenceClassificationDataset):
    '''
    对原始数据尽心读取得到每个样本原始的字符串形式
    '''

    def __init__(self, doc_stride=64, max_query_length=64, **kwargs):
        super(LoadSQuADQuestionAnsweringDataset, self).__init__(**kwargs)
        print(self.max_sen_len)
        self.doc_stride = doc_stride  # 滑动窗口长度
        self.max_query_length = max_query_length

    def preprocessing(self, filepath, is_training=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.loads(f.read())
            data = raw_data['data']
        examples = []
        for i in tqdm.tqdm(range(len(raw_data)), ncols=80):
            paragraphs = data[i]['paragraphs']
            for j in range(len(paragraphs)):
                context = paragraphs[j]['context']
                context_tokens, word_offset = get_format_text_and_word_offset(context)
                qas = paragraphs[j]['qas']
                for k in range(len(qas)):
                    question_text = qas[k]['question']
                    qas_id = qas[k]['id']
                    if is_training:
                        if not qas[k]['answers']:
                            continue
                        answer_offset = qas[k]['answers'][0]['answer_start']
                        orig_answer_text = qas[k]['answers'][0]['text']
                        answer_length = len(orig_answer_text)
                        start_position = word_offset[answer_offset]
                        end_position = word_offset[answer_offset + answer_length - 1]
                        actual_text = ' '.join(context_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = ' '.join(orig_answer_text.strip().split())
                        if actual_text.find(cleaned_answer_text) == -1:
                            logging.warning('Could not find answer: "%s" vs "%s"' % (actual_text, cleaned_answer_text))
                            continue
                    else:
                        start_position = None
                        end_position = None
                        orig_answer_text = None
                    examples.append([qas_id, question_text, orig_answer_text,
                                     ' '.join(context_tokens), start_position, end_position])
        # question_id，question，answer，context，start，end
        return examples

    @staticmethod
    def get_token_to_orig_map(input_tokens, origin_context, tokenizer):
        '''
        根据input_tokens和原始上下文，返回input_tokens中每个单词在原始单词中所对应的位置索引
        input_tokens是已经tokenize之后的内容，一个词可能被拆分成多个词缀，所以需要重新定位索引
        :param input_tokens:
        :param origin_context:
        :param tokenizer:
        :return:
        '''
        origin_context_tokens = origin_context.split()
        token_id = []
        str_origin_context = ''
        for i in range(len(origin_context_tokens)):
            tokens = tokenizer.tokenize(origin_context_tokens[i])
            str_token = ''.join(tokens)
            str_origin_context += '' + str_token
            for _ in str_token:
                token_id.append(i)
        key_start = input_tokens.index('[SEP]') + 1
        tokenized_tokens = input_tokens[key_start: -1]
        str_tokenized_tokens = ''.join(tokenized_tokens)
        index = str_origin_context.index(str_tokenized_tokens)
        value_start = token_id[index]
        token_to_orig_map = {}
        # 把context中每个词单独tokenize一遍
        token = tokenizer.tokenize(origin_context_tokens[value_start])
        for i in range(len(token), -1, -1):
            s1 = ''.join(token[-i:])
            s2 = ''.join(tokenized_tokens[:i])
            if s1 == s2:
                token = token[-i:]
                break
        while True:
            for j in range(len(token)):
                token_to_orig_map[key_start] = value_start
                key_start += 1
                if len(token_to_orig_map) >= len(tokenized_tokens):  # 全部处理完了
                    return token_to_orig_map
            value_start += 1
            token = tokenizer.tokenize(origin_context_tokens[value_start])

    # @cache
    def data_process(self, filepath, is_training=False, postfix='cache'):
        '''
        这部分比较重要
        :param filepath:
        :param is_training:
        :param postfix:
        :return:
        '''
        logging.info(f'## 使用窗口滑动， doc_stride = {self.doc_stride}')
        # examples: [question_id，question，answer，context，start_id，end_id], 并且还未tokenize过
        examples = self.preprocessing(filepath, is_training)
        all_data = []
        example_id, feature_id = 0, 1e+7
        for example in tqdm.tqdm(examples, ncols=80, desc='正在遍历每个样本'):
            question_tokens = self.tokenizer.tokenize(example[1])
            if len(question_tokens) > self.max_query_length:
                question_tokens = question_tokens[:self.max_query_length]
            question_ids = [self.vocab[token] for token in question_tokens]
            question_ids = [self.CLS_IDX] + question_ids + [self.SEP_IDX]
            context_tokens = self.tokenizer.tokenize(example[3])
            context_ids = [self.vocab[token] for token in context_tokens]
            logging.debug(f'<<<<进入新的example>>>>')
            start_position, end_position, answer_text = -1, -1, None
            if is_training:
                start_position, end_position = example[4], example[5]
                answer_text = example[2]
                answer_tokens = self.tokenizer.tokenize(answer_text)
                start_position, end_position = improve_answer_span(context_tokens,
                                                                   answer_tokens, start_position, end_position)
            rest_len = self.max_sen_len - len(question_ids) - 1
            context_ids_len = len(context_ids)
            logging.debug(f'## 上下文长度为：{context_ids_len}, 剩余长度 rest_len 为：{rest_len}')

            # 处理滑动窗口部分
            # 大致想法就是正负采样，如果有答案，则正采样，如果没答案，就负采样——负采样数量太多？
            if context_ids_len > rest_len:
                logging.debug(f'## 进入滑动窗口 ……')
                s_idx, e_idx = 0, rest_len
                while True:
                    tmp_context_ids = context_ids[s_idx: e_idx]
                    tmp_context_tokens = [self.vocab.itos[item] for item in tmp_context_ids]
                    input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + tmp_context_tokens + ['[SEP]']
                    input_ids = torch.tensor([self.vocab[token] for token in input_tokens])
                    seg = [0] * len(question_ids) + [1] * (len(input_ids) - len(question_ids))
                    seg = torch.tensor(seg)
                    if is_training:
                        new_start_position, new_end_position = 0, 0
                        if start_position >= s_idx and end_position <= e_idx:
                            logging.debug(f'## 滑动窗口中存在答案 -----> ')
                            new_start_position = start_position - s_idx
                            new_end_position = new_start_position + (end_position - start_position)
                            new_start_position += len(question_ids)
                            new_end_position += len(question_ids)
                        all_data.append(
                            [example_id, feature_id, input_ids, seg, new_start_position, new_end_position, answer_text,
                             example[0], input_tokens])
                    else:
                        all_data.append(
                            [example_id, feature_id, input_ids, seg, start_position, end_position, answer_text,
                             example[0], input_tokens])
                    token_to_orig_map = self.get_token_to_orig_map(input_tokens, example[3], self.tokenizer)
                    all_data[-1].append(token_to_orig_map)
                    feature_id += 1
                    if e_idx >= context_ids_len:
                        break
                    # 这一步有待商榷
                    s_idx += self.doc_stride
                    e_idx += self.doc_stride
            else:  # 非滑动窗口部分
                input_ids = torch.tensor(question_ids + context_ids + [self.SEP_IDX])
                input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
                seg = [0] * len(question_ids) + [1] * (len(input_ids) - len(question_ids))
                seg = torch.tensor(seg)
                if is_training:
                    start_position += len(question_ids)
                    end_position += len(question_ids)
                token_to_orig_map = self.get_token_to_orig_map(input_tokens, example[3], self.tokenizer)
                all_data.append(
                    [example_id, feature_id, input_ids, seg, start_position, end_position, answer_text, example[0],
                     input_tokens, token_to_orig_map])
                feature_id += 1
            example_id += 1
        data = {'all_data': all_data, 'max_len': self.max_sen_len, 'examples': examples}
        return data

    def generate_batch(self, data_batch):
        batch_input, batch_seg, batch_label, batch_qid = [], [], [], []
        batch_example_id, batch_feature_id, batch_map = [], [], []
        for item in data_batch:
            batch_example_id.append(item[0])
            batch_feature_id.append(item[1])
            batch_input.append(item[2])
            batch_seg.append(item[3])
            batch_label.append((item[4], item[5]))
            batch_qid.append(item[7])
            batch_map.append(item[9])

        batch_input = pad_sequence(batch_input, padding_value=self.PAD_IDX, batch_first=False, max_len=self.max_sen_len)
        batch_seg = pad_sequence(batch_seg, padding_value=self.PAD_IDX, batch_first=False, max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_input, batch_seg, batch_label, batch_qid, batch_example_id, batch_feature_id, batch_map

    def get_data_iter(self, path, postfix):
        if not path:
            return None, None
        else:
            data = self.data_process(filepath=path,
                                     is_training=True,
                                     postfix=postfix)
            data, examples = data['all_data'], data['examples']
            iter = DataLoader(data, shuffle=True, batch_size=self.batch_size, collate_fn=self.generate_batch)
            return iter, examples

    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None):
        doc_stride = str(self.doc_stride)
        max_sen_len = str(self.max_sen_len)
        max_query_length = str(self.max_query_length)
        postfix = doc_stride + '_' + max_sen_len + '_' + max_query_length
        test_iter, test_examples = self.get_data_iter(test_file_path, postfix)
        train_iter, _ = self.get_data_iter(train_file_path, postfix)
        val_iter, _ = self.get_data_iter(val_file_path, postfix)
        return train_iter, val_iter, test_iter, test_examples


if __name__ == '__main__':
    from Config import ModelConfig
    from pytorch_transformers import BertTokenizer

    model_config = ModelConfig(bert_name='bert-base-uncased', dataset_name='SQuAD-English', task='MRC')
    model_config.test_file_path = model_config.test_file_path.split('.txt')[0] + '.json'
    model_config.train_file_path = model_config.train_file_path.split('.txt')[0] + '.json'
    model_config.val_file_path = model_config.val_file_path.split('.txt')[0] + '.json'
    print(model_config.max_sen_len)
    load_dataset = LoadSQuADQuestionAnsweringDataset(
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(model_config.pretrained_model_dir),
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        split_sep=model_config.split_sep,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_index=model_config.pad_token_id,
        is_sample_shuffle=model_config.is_sample_shuffle
    )
    data = load_dataset.data_process(filepath=model_config.test_file_path, is_training=True,
                                     postfix='')
    for example_id, feature_id, input_ids, seg, start_position, end_position, answer_text, example, input_tokens, token_to_orig_map in \
            data['all_data']:
        print(input_tokens)
        print(torch.tensor([load_dataset.vocab[token] for token in input_tokens], dtype=torch.long))
        print(input_ids)
        break
    batch_input, batch_seg, batch_label, batch_qid, batch_example_id, batch_feature_id, batch_map = load_dataset.generate_batch(
        data['all_data'])

    batch_input = batch_input.transpose(0, 1)
    for i, s, l, q, e, f, m in zip(batch_input, batch_seg, batch_label, batch_qid, batch_example_id, batch_feature_id,
                                   batch_map):
        print(i)
        print(s.shape)
        print(l.shape)
        print(q)
        print(e)
        print(f)
        print(m)
        break
