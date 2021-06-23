import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.utils import shuffle
from os import listdir
from os.path import isfile, join


# Reading the data
gold = '../data/corpora/polnear-conll/dev-conll/foreval/breitbart_2016-09-12_stealth-over-health-hillary-clin.txt.xml'
gold_df = pd.read_csv()
print(gold_df)

# Extracting the features
features_folder = '../data/encodings/polnear-conll/dev-conll/foreval/breitbart_2016-09-12_stealth-over-health-hillary-clin.xml.conll.features.foreval'
all_files = [f for f in listdir(features_folder) if isfile(join(features_folder, f))]
print(all_files)
# Extracting the gold labels (different files)

# Creating sliding window for 2 sentences
#
#
# # Design network
# model = Sequential()
# model.add(LSTM(32, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='relu'))
# model.compile(loss='mse', optimizer='adam')
#
# def train_model(train_features, train_labels, model, epochs=20, verbose=2):
#     """
#     (taken from DMT script)
#     :param train_features:
#     :param train_labels:
#     :param model:
#     :return:
#     """
#     articles = []
#
#     train_features, train_labels = shuffle(train_features, train_labels)
#
#     for i in range(epochs):
#         for train_feature, train_label in zip(train_features, train_labels):
#                 article = model.train_on_batch(train_feature, train_label, reset_metrics=False)
#
#         articles.append(article)
#
#     return articles
#
# # Training the model
# trained_model = train_model(train_features, train_labels, model, epochs=20)