import pandas as pd
import more_itertools
import joblib
import os

from tensorflow.keras.layers import LSTM
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
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
features_folder = '../data/encodings/polnear-conll/dev-conll-foreval/breitbart_2016-09-12_stealth-over-health-hillary-clin.xml.conll.features.foreval'

# Saving all paths in directory to list
file_list = []
for path in os.listdir(features_folder):
    full_path = os.path.join(features_folder, path)
    file_list.append(full_path)

# Opening files in file_list
opened_files = []
for files in file_list:
    file = joblib.load(files)
    opened_files.append(file)

# Creating sliding window for 2 sentences, output == list with tuples
sentence_combinations = list(more_itertools.windowed(opened_files,n=2, step=1))

# Design network
model = Sequential()
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam')


def extracting_features(sentence_combinations):
    """
    Function to extract the features from the sentence combinations (BERT representations)
    first CLS token, second CLS, all token reprs from 1st sentence and 2nd sentence
    """
    features = []
    gold_labels = []

    CLS1 = []
    CLS2 = []
    tokens1 = []
    tokens2 = []

    for tups in sentence_combinations:
        for sents in tups:

            sent1 = sents[0]
            sent2 = sents[1]

            CLS1.append(sent1[0])
            CLS2.append(sent2[0])
            tokens1.append(sent1[1:])
            tokens2.append(sent2[1:])

    return CLS1, CLS2, tokens1, tokens2 #this needs to be a dict? or a list?

feats = extracting_features(sentence_combinations)
print(feats)

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
trained_model = train_model(train_features, train_labels, model, epochs=20)

# Loading file for testing
gold_test = '../data/corpora/polnear-conll/dev-conll-foreval/breitbart_2016-09-15_pat-caddell-democrat-voters-worr.xml.conll.features.foreval'

# Load file with gold labels
test_df = pd.read_csv(gold, sep='\t', names=column_names)

# Select gold labels
test_labels = gold_df['att']

