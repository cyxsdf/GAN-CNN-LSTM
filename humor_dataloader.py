#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# In[2]:


'''
you can assign the maximum number number of sentences in context and what will be the maximum number of words of any sentence.

It will do left padding . It will concatenate the word embedding + covarep features + openface features

example:

if max_sen_len = 20 then the punchline sentence dimension = 20 * 752. 
    where 752 = word embedding (300) + covarep (81) + openface(371)  

if max_sen_len = 20 and max_context_len = 5 that means context can have maximum 5 sentences 
and each sentence will have maximum 20 words. The context dimension will be 5 * 20 * 752 

We will do left padding with zeros to maintaing the same dimension.

In our experiments we set max_sen_len = 20 & max_context_len = 5 
'''


class HumorDataset(Dataset):

    def __init__(self, mode, path, max_context_len=5, max_sen_len=20, online=False):
        # Set online to True if you need to make the dataset at the max_context_len and max_sen_len you specify.
        # Note that this will process the data in the __getitem__ method, which will cause a dramatic drop in speed.
        self.online = online
        if online:
            data_folds_file = path + "data_folds.pkl"
            openface_file = path + "openface_features_sdk.pkl"
            covarep_file = path + "covarep_features_sdk.pkl"
            language_file = path + "language_sdk.pkl"  # word_embedding_indexes_sdk
            word_embedding_list_file = path + "word_embedding_list.pkl"
            humor_label_file = path + "humor_label_sdk.pkl"

            data_folds = load_pickle(data_folds_file)
            self.word_aligned_openface_sdk = load_pickle(openface_file)
            self.word_aligned_covarep_sdk = load_pickle(covarep_file)
            self.language_sdk = load_pickle(language_file)
            self.word_embedding_list_sdk = load_pickle(word_embedding_list_file)
            self.humor_label_sdk = load_pickle(humor_label_file)
            self.of_d = 371
            self.cvp_d = 81
            self.glove_d = 300
            self.total_dim = self.glove_d + self.of_d + self.cvp_d
            self.max_context_len = max_context_len
            self.max_sen_len = max_sen_len
            self.id_list = data_folds[mode]
        else:
            # If you already have a dataset with extracted features
            dataset_path = path + '[onlypunch][glove]urfunny-40.pkl'
            dataset = pickle.load(open(dataset_path, 'rb'))
            self.x_p = torch.tensor(dataset[mode]['punchline'].astype(np.float32)).cpu().detach()
            self.y = torch.tensor(dataset[mode]['label'].astype(np.float32)).cpu().detach()

    # left padding with zero  vector upto maximum number of words in a sentence * glove embedding dimension
    def paded_word_idx(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        pad_w = np.concatenate((np.zeros(max_sen_len - len(seq)), seq), axis=0)
        pad_w = np.array([self.word_embedding_list_sdk[int(w_id)] for w_id in pad_w])
        return pad_w

    # left padding with zero  vector upto maximum number of words in a sentence * covarep dimension
    def padded_covarep_features(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        return np.concatenate((np.zeros((max_sen_len - len(seq), self.cvp_d)), seq), axis=0)

    # left padding with zero  vector upto maximum number of words in a sentence * openface dimension
    def padded_openface_features(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        return np.concatenate((np.zeros(((max_sen_len - len(seq)), self.of_d)), seq), axis=0)

    # left padding with zero vectors upto maximum number of sentences in context * maximum num of words in a sentence * 456
    def padded_context_features(self, context_w, context_of, context_cvp, max_context_len=5, max_sen_len=20):
        context_w = context_w[-max_context_len:]
        context_of = context_of[-max_context_len:]
        context_cvp = context_cvp[-max_context_len:]

        padded_context = []
        for i in range(len(context_w)):
            p_seq_w = self.paded_word_idx(context_w[i], max_sen_len)
            p_seq_cvp = self.padded_covarep_features(context_cvp[i], max_sen_len)
            p_seq_of = self.padded_openface_features(context_of[i], max_sen_len)
            padded_context.append(np.concatenate((p_seq_w, p_seq_cvp, p_seq_of), axis=1))

        pad_c_len = max_context_len - len(padded_context)
        padded_context = np.array(padded_context)

        # if there is no context
        if not padded_context.any():
            return np.zeros((max_context_len, max_sen_len, self.total_dim))

        return np.concatenate((np.zeros((pad_c_len, max_sen_len, self.total_dim)), padded_context), axis=0)

    def padded_punchline_features(self, punchline_w, punchline_of, punchline_cvp, max_sen_len=20, left_pad=1):

        p_seq_w = self.paded_word_idx(punchline_w, max_sen_len)
        p_seq_cvp = self.padded_covarep_features(punchline_cvp, max_sen_len)
        p_seq_of = self.padded_openface_features(punchline_of, max_sen_len)
        return np.concatenate((p_seq_w, p_seq_cvp, p_seq_of), axis=1)

    def __len__(self):
        if self.online:
            return len(self.id_list)
        else:
            return len(self.y)

    def __getitem__(self, index):
        if self.online:
            hid = self.id_list[index]
            # 1. 处理 punchline 特征（原有逻辑正确，无需修改）
            punchline_w = np.array(self.language_sdk[hid]['punchline_embedding_indexes'])
            punchline_of = np.array(self.word_aligned_openface_sdk[hid]['punchline_features'])
            punchline_cvp = np.array(self.word_aligned_covarep_sdk[hid]['punchline_features'])

            # 2. 处理 context_w（原有逻辑正确，无需修改）
            context_w = self.language_sdk[hid]['context_embedding_indexes']

            # 3. 正确处理 context_of：复用 padded_openface_features，统一维度为 [句子数, self.max_sen_len, 371]
            context_of_raw = self.word_aligned_openface_sdk[hid]['context_features']  # 原始格式：[句子数, 单词数, 371]
            context_of = []
            for single_sentence_of in context_of_raw:
                # 调用类方法，自动完成：截断/填充到 self.max_sen_len 个单词 + 确保特征维度为 371
                padded_sent_of = self.padded_openface_features(
                    seq=single_sentence_of,
                    max_sen_len=self.max_sen_len  # 用全局统一参数，而非手动max_length
                )
                context_of.append(padded_sent_of)
            context_of = np.array(context_of)  # 此时形状：[句子数, 40, 371]

            # 4. 同步处理 context_cvp：复用 padded_covarep_features，统一维度为 [句子数, self.max_sen_len, 81]
            context_cvp_raw = self.word_aligned_covarep_sdk[hid]['context_features']  # 原始格式：[句子数, 单词数, 81]
            context_cvp = []
            for single_sentence_cvp in context_cvp_raw:
                # 调用类方法，自动完成：截断/填充到 self.max_sen_len 个单词 + 确保特征维度为 81
                padded_sent_cvp = self.padded_covarep_features(
                    seq=single_sentence_cvp,
                    max_sen_len=self.max_sen_len
                )
                context_cvp.append(padded_sent_cvp)
            context_cvp = np.array(context_cvp)  # 此时形状：[句子数, 40, 81]

            # 5. 生成最终特征（原有逻辑正确，无需修改）
            x_p = torch.FloatTensor(
                self.padded_punchline_features(punchline_w, punchline_of, punchline_cvp, self.max_sen_len))
            x_c = torch.FloatTensor(
                self.padded_context_features(context_w, context_of, context_cvp, self.max_context_len,
                                             self.max_sen_len))
            y = torch.FloatTensor([self.humor_label_sdk[hid]])
        else:
            x_p = self.x_p[index]
            x_c = 0
            y = self.y[index]
        return x_c, x_p, y


if __name__ == '__main__':
    path = './data/UR-FUNNY/sdk_features/'
    max_context_len = 8
    max_sen_len = 40
    train_data = HumorDataset('train', path, max_context_len, max_sen_len, online=True)
    print(train_data.__getitem__(0)[0].shape)
    print(train_data.__getitem__(0)[1].shape)
    print(train_data.__getitem__(0)[2].shape)