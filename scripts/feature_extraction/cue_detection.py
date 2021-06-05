"""
Classifier to predict whether the head of each verb group is a verb-cue.

We need to check whether the cues are usually verbs,
and if they are verbs often enough.

Features:
> lexical: token, lemma, adjacent tokens
> VerbNet class membership (or generalize to WordNet class membership?)
> syntactic: node-depth in sentence, parent and sibling nodes
> sentence features: distance from sentence start/end; within quotes?
"""
import glob
import pickle

from tqdm import tqdm

import pandas as pd

from nltk.corpus import verbnet

import scripts.constants as constants

quotes = {'"', '\'', '``', '`', '\'\''}


def feature_label_extraction(filepath):
    """
    This function extracts a number of features
    and cue labels from an input file.

    :param filepath: Input filepath
    :return: a list of feature dicts and a list of labels
    """
    features = []
    labels = []

    active_quote = False

    df = pd.read_csv(
        filepath,
        **constants.conll_kwargs  # TODO: fix problem with quotechar
    )

    sentences = df.groupby(
        df.sent_num
    )

    for sentence in sentences:

        max_index = sentence[1].token_num_sent.max()

        for idx, word in sentence[1].iterrows():

            # Head token
            parent_idx = int(word.dep_head) - 1
            parent_token = sentence[1].token.iloc[parent_idx]

            # Whether there is an active quotation
            if word.token in quotes:
                if active_quote is True:
                    active_quote = False
                else:
                    active_quote = True

            # Adjacent tokens
            if word.token_num_sent == max_index:
                prev_token = sentence[1].token.iloc[word.token_num_sent - 2]
                next_token = '<END>'

            elif word.token_num_sent == 1:
                prev_token = '<START>'
                next_token = sentence[1].token.iloc[word.token_num_sent]

            else:
                prev_token = sentence[1].token.iloc[word.token_num_sent - 2]
                next_token = sentence[1].token.iloc[word.token_num_sent]

            # VerbNet id
            verbnet_id = verbnet.classids(lemma=word.lemma)

            # Create target
            label = 'non-cue'
            if word.attr_labels:
                if 'CUE' in word.attr_labels:
                    label = 'cue'

            # Create feature dict
            feature_dict = {
                'token': word.token,
                'lemma': word.lemma,
                'pos': word.pos,
                'prev_token': prev_token,
                'next_token': next_token,
                'parent_token': parent_token,
                # 'verbnet_id': verbnet_id,  # TODO: which verbnet id to keep?
                'sent_start_distance': word.token_num_sent,
                'sent_end_distance': max_index - word.token_num_sent,
                'active_quote': active_quote
            }

            features.append(feature_dict)
            labels.append(label)

    return features, labels


# path1 = '../../data/parc30-conll/train-conll-foreval/wsj_0003.xml.conll.features.foreval'
# path2 = '../../data/parc30-conll/train-conll-foreval/wsj_0004.xml.conll.features.foreval'
#
# paths = [path1, path2]

# all_files = glob.glob('../../data/**/**/**')
#
# train_files = [
#     filename for filename in all_files
#     if 'train' in filename
# ]
#
# dev_files = [
#     filename for filename in all_files
#     if 'dev' in filename
# ]

# print(train_files)

# train_feature_list = []
# train_label_list = []
# dev_feature_list = []
# dev_label_list = []
#
# with tqdm(total=len(train_files), desc='Training files: ') as pbar:
#
#     for path in train_files:
#         # print(path)
#         features, labels = feature_label_extraction(path)
#
#         for feature, label in zip(features, labels):
#             train_feature_list.append(feature)
#             train_label_list.append(label)
#
#         pbar.update(1)
#
# print()
#
# with tqdm(total=len(dev_files), desc='Development files: ') as pbar:
#
#     for path in dev_files:
#         print(path)
#         features, labels = feature_label_extraction(path)
#
#         for feature, label in zip(features, labels):
#             dev_feature_list.append(feature)
#             dev_label_list.append(label)
#
#         pbar.update(1)
#
# with open('../../data/cue_detection/train_features.pkl', 'wb') as outfile:
#     pickle.dump(train_feature_list, outfile)
#
# with open('../../data/cue_detection/train_labels.pkl', 'wb') as outfile:
#     pickle.dump(train_label_list, outfile)
#
# with open('../../data/cue_detection/dev_features.pkl', 'wb') as outfile:
#     pickle.dump(dev_feature_list, outfile)
#
# with open('../../data/cue_detection/dev_labels.pkl', 'wb') as outfile:
#     pickle.dump(dev_label_list, outfile)

# path = r'D:\VU Amsterdam\NLPT_6\data\parc30-conll\dev-conll-foreval\wsj_2400.xml.conll.features.foreval'
# # path = r'D:\VU Amsterdam\NLPT_6\data\parc30-conll\train-conll-foreval\wsj_0004.xml.conll.features.foreval'
#
# df = pd.read_csv(
#     path,
#     **constants.conll_kwargs
# )
#
# for word, item in zip(df.token, df.dep_head):
#     print(word, item)
