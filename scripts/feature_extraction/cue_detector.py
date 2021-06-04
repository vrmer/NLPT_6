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

# import csv
import scripts.constants as constants
import pandas as pd
from nltk.corpus import verbnet

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
        **constants.conll_kwargs
    )

    sentences = df.groupby(
        df.sent_num
    )

    for sentence in sentences:

        max_index = sentence[1].token_num_sent.max()

        for idx, word in sentence[1].iterrows():

            # Head token
            parent_idx = word.dep_head - 1
            parent_token = sentence[1].token.iloc[parent_idx]

            # Whether there is an active quotation
            if word.token in quotes:
                if active_quote is True:
                    active_quote = False
                else:
                    active_quote = True

            # Adjacent tokens
            if word.token_num_sent == 1:
                prev_token = '<START>'
                next_token = sentence[1].token.iloc[word.token_num_sent]

            elif word.token_num_sent == max_index:
                prev_token = sentence[1].token.iloc[word.token_num_sent - 2]
                next_token = '<END>'

            else:
                prev_token = sentence[1].token.iloc[word.token_num_sent - 2]
                next_token = sentence[1].token.iloc[word.token_num_sent]

            # VerbNet id
            verbnet_id = verbnet.classids(lemma=word.lemma)

            # Create target
            if 'CUE' in word.attr_labels:
                label = 'cue'
            else:
                label = 'non-cue'

            # Create feature dict
            feature_dict = {
                'token': word.token,
                'lemma': word.lemma,
                'pos': word.pos,
                'prev_token': prev_token,
                'next_token': next_token,
                'parent_token': parent_token,
                'verbnet_id': verbnet_id,
                'sent_start_distance': word.token_num_sent,
                'sent_end_distance': max_index - word.token_num_sent,
                'active_quote': active_quote
            }

            features.append(feature_dict)
            labels.append(label)

    return features, labels


# path = '../../data/parc30-conll/train-conll-foreval/wsj_0003.xml.conll.features.foreval'
#
# features, labels = feature_label_extraction(path)
#
# print(features[0])
# print(labels[0])
