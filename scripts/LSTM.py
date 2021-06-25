import pandas as pd
import more_itertools
import joblib
import os
import re
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.layers import LSTM
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
from sklearn.utils import shuffle


# Reading the data
gold = '../data/corpora/polnear-conll/dev-conll-foreval/breitbart_2016-09-12_stealth-over-health-hillary-clin.xml.conll.features.foreval'
column_names = ['article',
                'sent_n',
                'doc_idx',
                'sent_idx',
                'offsets',
                'token',
                'lemma',
                'pos',
                'dep_label',
                'dep_head',
                'att']

# Load file with gold labels
gold_df = pd.read_csv(gold, sep='\t', names=column_names)

# Select gold labels
gold_labels = gold_df['att']

# Function from Dri for extracting gold label
def extract_gold_label(cell):
    '''
    Strip underscores and info attached to gold label (e.g. -PD-0).
    :return: gold labels
    '''
    match = re.findall(r'[BI]-[A-Z]*', str(cell))
    if match:
        for tag in match:
            if "-NE" not in tag: # exclude nested attributions
                cell = tag
    else: # if cell only contains underscores, token does not belong to a source, a cue or a content
        cell = '_'
    return cell

# Extracting the gold labels from input file
gold_labels_cleaned = []
for labels in gold_labels:
    new_label = extract_gold_label(labels)
    gold_labels_cleaned.append(new_label)

# TODO: change this to loop over all documents in dev-conll-foreval folder
# Extracting the features
features_folder = "../data/encodings/polnear-conll/train-conll-foreval/breitbart_2016-09-04_obama-i-m-optimistic-americans-w.xml.conll.features.foreval"

    #'../data/encodings/polnear-conll/dev-conll-foreval/breitbart_2016-09-12_stealth-over-health-hillary-clin.xml.conll.features.foreval'

# Saving all paths in directory to list
file_list = []
for path in os.listdir(features_folder):
    full_path = os.path.join(features_folder, path)
    file_list.append(full_path)

# Link to all article paths
# article_folder = '../data/encodings/polnear-conll/dev-conll-foreval/'

#

def getting_encodings(folder_path):
    """
    :param folder_path: path to folder with subdirs for article encodings
    :return: list with
    """
# all_articles = []
# for path in os.listdir(article_folder):
#     full_path = os.path.join(article_folder, path)
#     all_articles.append(full_path)
# print(all_articles)


# Opening files in file_list
opened_files = []
for files in file_list:
    file = joblib.load(files)
    opened_files.append(file)

# Creating sliding window for 2 sentences, output == list with tuples
sentence_combinations = list(more_itertools.windowed(opened_files,n=2, step=1))

print(sentence_combinations)

# def opening_sent_combinations(sentence_combinations):
#     """
#     :param sentence_combinations: list with pairs of paths to files with sentence representations
#     :return:
#     """
#     for combinations in sentence_combinations:
#
# # TODO: make pairs of filepaths instead of opened files

# Design network
# TODO: classifier settings instead of regression model DONE, TAKEN FROM https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
model = Sequential()
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def extracting_features(sentence_combinations):
    """
    Function to extract the features from the sentence combinations (BERT representations)
    first CLS token, second CLS, all token reprs from 1st sentence and 2nd sentence
    """

    token_representations = []
    #
    for sentences in sentence_combinations:

        current_tup = sentences

        # Indexing sentences
        sent1 = current_tup[0]
        sent2 = current_tup[1]

        # Indexing CLS tokens
        CLS1 = sent1[0]
        CLS2 = sent2[0]

        # Indexing tokens
        tokens1 = sent1[1:]
        print('TOKENS 1', len(tokens1))
        tokens2 = sent2[1:]
        print('TOKENS 2', len(tokens2))

        for tokens in tokens1:
            token_rep = np.concatenate(((CLS1), (CLS2), (tokens)), axis=None)
            token_representations.append(token_rep)

        for tokens in tokens2:
            token_rep = np.concatenate(((CLS1), (CLS2), (tokens)), axis=None)
            token_representations.append(token_rep)

    print('LEN FEATS', len(token_representations))
    return token_representations

def extracting_gold(path):
    """
    Function to extract gold labels from processed documents
    :param path:
    :return:
    """

    gold_labels = []

    file_list = []
    for paths in os.listdir(path):
        full_path = os.path.join(path, paths)
        if full_path.endswith('.conll'):
            file_list.append(full_path)

    for files in file_list:
        df = pd.read_csv(files, delimiter='\t')

        gold = df['10'] #column with gold labels in conll files
        print('GOLD PER FILE', len(gold))
        for labels in gold:

            gold_labels.append(labels)
    print('LEN GOLD LABELS', len(gold_labels))
    # Vectorizing the gold labels
    vec = CountVectorizer()
    gold_vectorized = vec.fit_transform(gold_labels)

    return gold_vectorized

# Extracting sentence features and gold labels
sent_feats = extracting_features(sentence_combinations)
gold_vectorized = extracting_gold("../output_old/output_old/breitbart_2016-09-04_obama-i-m-optimistic-americans-w.xml.conll.features.foreval")


def train_model(train_features, train_labels, model, epochs=20, verbose=2):
    """
    (taken from DMT script)
    :param train_features:
    :param train_labels:
    :param model:
    :return:
    """
    articles = []

    train_features, train_labels = shuffle(train_features, train_labels)

    for i in range(epochs):
        for train_feature, train_label in zip(train_features, train_labels):
                article = model.train_on_batch(train_feature, train_label, reset_metrics=False)

        articles.append(article)

    return articles

# Training the model
trained_model = train_model(sent_feats, gold_vectorized, model, epochs=20)
#
# # Loading file for testing
# gold_test = '../data/corpora/polnear-conll/dev-conll-foreval/breitbart_2016-09-15_pat-caddell-democrat-voters-worr.xml.conll.features.foreval'
#
