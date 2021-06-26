import glob
import os
import numpy as np
# import pandas as pd
import joblib
# from keras import Sequential
# from keras.layers import LSTM, Dropout, Dense
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


instance_paths = glob.glob('../data/instances/**/**/**')

train_paths = [
    path for path in instance_paths
    if 'train-conll-foreval' in path
]


def extract_instance_encodings_labels(input_filepath, corpus='train-conll-foreval'):
    """
    This function returns the correct sentence and token encodings
    for each training instance. The input should be the path to the
    instance files.
    """
    instance_encodings = []

    encoding_path = f'../data/encodings/polnear-conll/{corpus}'

    filename = os.path.basename(
        os.path.dirname(
            input_filepath
        )
    )

    instance = joblib.load(input_filepath)
    sentence_indices = instance[11].unique().tolist()
    labels = instance[10].tolist()

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
        classifier_features.append(token_rep)

    return classifier_features


classes = ['SOURCE', 'CUE', 'CONTENT', 'O']
label_encoder = LabelEncoder()
label_encoder.fit(classes)
all_label_encoder_classes = label_encoder.transform(label_encoder.classes_)  # total classes
sgd = SGDClassifier()


def train_svm(input_filepath, corpus='train-conll-foreval'):
    """
    This function loads an article, and trains on the basis of it.
    """
    encodings, labels = extract_instance_encodings_labels(input_filepath, corpus=corpus)

    train_features = create_classifier_features(encodings)
    train_labels = label_encoder.transform(labels)

    train_labels = np.array(train_labels).ravel()

    train_features, train_labels = shuffle(train_features, train_labels)

    sgd.partial_fit(train_features, train_labels, classes=all_label_encoder_classes)


# def create_dummy_instance(filepath):
#     """
#     This function takes the last sentence in an article
#     and pairs it with a dummy sentence representation
#     to create a two-sentence instance with the last sentence
#     as the first sentence of the instance.
#     """
#     dummy_instance = np.zeros(2)
#     dummy_label = ['O']
#
#     instance = joblib.load(filepath)
#     sentence_indices = instance[11].unique().tolist()
#     sorted_sentence_indices = sorted(sentence_indices)
#
#     last_index = sorted_sentence_indices[1]


with tqdm(total=len(train_paths), desc='Training...') as pbar:

    for idx, path in enumerate(train_paths):

        # if train_paths[idx+1].endswith('0.pickle'):
        #     print('Next one is 0')

        train_svm(path)

        pbar.update(1)


joblib.dump(sgd, '../../data/models/sgd_classifier_two_sentence_instances.sav')
