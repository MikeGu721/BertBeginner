from preprocess.SentenceClassification import LoadSingleSentenceClassificationDataset
import pandas as pd
from utils import pad_sequence
import tqdm
import torch


class LoadMultipleChoiceDataset(LoadSingleSentenceClassificationDataset):
    def __init__(self, num_choice=4, **kwargs):
        super(LoadMultipleChoiceDataset, self).__init__(**kwargs)
        self.num_choice = num_choice

    def data_process(self, filepath):
        print(filepath)
        data = pd.read_csv(filepath)
        questions = data['startphrase']
        answers = []
        for i in range(self.num_choice):
            answers.append(data['ending%d' % i])
        labels = [-1] * len(questions)
        if 'label' in data:
            labels = data['label']
        del (data)
        all_data = []
        max_len = 0

        for i in tqdm.tqdm(range(len(questions)), ncols=80):
            # 转换问题
            t_q = [self.vocab[token] for token in self.tokenizer.tokenize(questions[i])]
            t_q = [self.CLS_IDX] + t_q + [self.SEP_IDX]
            # 转换答案
            t_as = []
            max_ans_len = 0
            for j in range(self.num_choice):
                t_as.append([self.vocab[token] for token in self.tokenizer.tokenize(answers[j][i])])
                for jj in t_as:
                    max_ans_len = max(max_ans_len, len(jj))
            # 最长序列长度
            max_len = max(max_len, len(t_q) + max_ans_len)
            seg_q = [0] * len(t_q)
            # 尾处理
            seg_as = []
            for t_a in t_as:
                seg_as.append([1] * (len(t_a) + 1))
            data = [t_q]
            data.extend(t_as)
            data.append(seg_q)
            data.extend(seg_as)
            data.append(labels[i])
            all_data.append(data)
        return all_data, max_len, self.num_choice

    def generate_batch(self, data_batch):
        batch_qa, batch_seg, batch_label = [], [], []

        def get_seq(q, a):
            seq = q + a
            if len(seq) > self.max_position_embedding - 1:
                seq = seq[:self.max_position_embedding - 1]
            return torch.tensor(seq + [self.SEP_IDX], dtype=torch.long)

        for item in data_batch:
            tmp_qa = []
            for i in range(self.num_choice):
                tmp_qa.append(get_seq(item[0], item[i + 1]))
            tmp_seg = []
            for i in range(self.num_choice):
                tmp_seg.append(
                    torch.tensor(item[self.num_choice + 1] + item[self.num_choice + 1 + i], dtype=torch.long))
            batch_qa.extend(tmp_qa)
            batch_seg.extend(tmp_seg)
            batch_label.append(item[-1])

        batch_qa = pad_sequence(batch_qa, padding_value=self.PAD_IDX, batch_first=True, max_len=self.max_sen_len)
        batch_mask = (batch_qa == self.PAD_IDX).view([-1, self.num_choice, batch_qa.size(-1)])
        batch_qa = batch_qa.view([-1, self.num_choice, batch_qa.size(-1)])
        batch_seg = pad_sequence(batch_seg, padding_value=self.PAD_IDX, batch_first=True, max_len=self.max_sen_len)
        batch_seg = batch_seg.view([-1, self.num_choice, batch_qa.size(-1)])
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_qa, batch_mask, batch_seg, batch_label


if __name__ == '__main__':
    import Config
    from pytorch_transformers import BertTokenizer

    model_config = Config.ModelConfig(bert_name='bert-base-uncased', dataset_name='SWAG', task='ChoiceQA')
    model_config.test_file_path = model_config.test_file_path.split('.txt')[0] + '.csv'
    model_config.train_file_path = model_config.train_file_path.split('.txt')[0] + '.csv'
    model_config.val_file_path = model_config.val_file_path.split('.txt')[0] + '.csv'
    print(model_config.test_file_path)
    load_dataset = LoadMultipleChoiceDataset(
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(model_config.pretrained_model_dir),
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        split_sep=model_config.split_sep,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_index=model_config.pad_token_id,
        is_sample_shuffle=model_config.is_sample_shuffle
    )
    a = pd.read_csv(model_config.test_file_path)
    train_iter, val_iter, test_iter, label_size = load_dataset.load_train_val_test_data(None, None,
                                                                                        model_config.test_file_path)
    test_iter, _, _ = load_dataset.data_process(model_config.test_file_path)
    batch_qa, batch_mask, batch_seg, batch_label = load_dataset.generate_batch(test_iter)
    for qa, mask, seg, label in zip(batch_qa, batch_mask, batch_seg, batch_label):
        print(seg)
        break
