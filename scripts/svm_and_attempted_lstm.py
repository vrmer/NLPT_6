import os
import numpy as np
import pandas as pd
import joblib
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

general_path = '../data/instances/breitbart_2015-11-11_the-ninth-circle-the-hellish-vie.xml.conll.features.foreval/'
encoding_path = '../data/encodings/polnear-conll/train-conll-foreval/breitbart_2015-11-11_the-ninth-circle-the-hellish-vie.xml.conll.features.foreval/'

df = pd.read_csv(general_path + '0.pickle.conll', sep='\t', header=0)

# print(df['11'].unique().tolist())

for idx in df['11'].unique().tolist():
    encoding = joblib.load(f'{encoding_path}{idx}.sav')
    # print(len(encoding))


def extract_instance_encodings_labels(input_filepath, corpus='train-conll-foreval'):
    """
    This function returns the correct sentence and token encodings
    for each training instance. The input should be the path to the
    instance files.
    """
    instance_encodings = []

    encoding_path = f'../data/encodings/polnear-conll/{corpus}/'

    filename = os.path.basename(
        os.path.dirname(
            input_filepath
        )
    )

    instance = pd.read_csv(input_filepath, sep='\t')
    sentence_indices = instance['11'].unique().tolist()
    labels = instance['10'].tolist()

    for sentence_index in sentence_indices:
        e_path = f'{encoding_path}/{filename}/{sentence_index}.sav'
        encoding = joblib.load(e_path)
        instance_encodings.append(encoding)

    return instance_encodings, labels


def create_classifier_features(instance_encodings):
    """
    This function creates [[CLS1] [CLS2] [token_n]]
    features that can serve as input to a classifier.
    """
    classifier_features = []

    sentence1 = instance_encodings[0]
    sentence2 = instance_encodings[1]

    # CLS tokens, tokens
    cls1, tokens1 = sentence1[0], sentence1[1:]
    cls2, tokens2 = sentence2[0], sentence2[1:]

    for token in (tokens1 + tokens2):
        token_rep = np.concatenate(
            (cls1, cls2, token), axis=None
        )
        # print(token_rep.shape)
        # token_rep = token_rep.reshape((token_rep.shape[0], 1, 1))
        classifier_features.append(token_rep)

    return classifier_features


enc, labels = extract_instance_encodings_labels(general_path + '0.pickle.conll')

# print(len(create_classifier_features(enc)))
# print(len(labels))

train_features = create_classifier_features(enc)

svm = SVC()

# # LSTM
# model = Sequential()
# model.add(LSTM(100, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(labels)

train_labels = [
    (item, ) for item in train_labels
]

train_features, train_labels = shuffle(train_features, train_labels)

svm.fit(train_features, np.array(train_labels).ravel())

# print(train_labels)

# print(train_features)
#
# for i in range(20):
#     for X, y in zip(train_features, train_labels):
#         model.train_on_batch(X, y, reset_metrics=False)
