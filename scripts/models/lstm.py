import glob
# import pickle
import os
import numpy as np
import joblib
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
# from keras.preprocessing import sequence
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt


def extract_instance_encodings_labels(input_filepath, corpus='train-conll-foreval'):
    """
    This function returns the correct sentence and token encodings
    for each training instance. The input should be the path to the
    instance files.
    """
    instance_encodings = []

    encoding_path = f'../../data/encodings/polnear-conll/{corpus}'

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

    tokens = tokens1 + tokens2

    for token in tokens:
        token_rep = np.concatenate(
            (cls1, cls2, token), axis=None
        )
        token_rep = token_rep.reshape((1, token_rep.shape[0]))
        classifier_features.append(token_rep)

    return classifier_features


instance_paths = glob.glob('../../data/instances/**/**/**')

train_paths = [
    path for path in instance_paths
    if 'train-conll-foreval' in path
]

dev_paths = [
    path for path in instance_paths
    if 'dev-conll-foreval' in path
]

classes = ['SOURCE', 'CUE', 'CONTENT', 'O']
label_encoder = LabelEncoder()
label_encoder.fit(classes)
all_label_encoder_classes = label_encoder.transform(label_encoder.classes_)  # total classes

# LSTM for classification
model = Sequential()
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def train_lstm(input_filepath, corpus='train-conll-foreval', epochs=10):
    """
    This function loads an article, and trains on the basis of it.
    """
    history = []

    encodings, labels = extract_instance_encodings_labels(input_filepath, corpus=corpus)

    train_features = create_classifier_features(encodings)
    train_labels = label_encoder.transform(labels)

    train_labels = np.array(train_labels).ravel()

    train_features, train_labels = shuffle(train_features, train_labels)

    for i in range(epochs):

        # for train_feature, train_label in zip(train_features, train_labels):
        history = model.train_on_batch(
            np.asarray(train_features),
            train_labels,
            reset_metrics=False
        )

    return history


def predict_on_data(input_filepath, corpus='dev-conll-foreval'):
    """
    This function carries out predictions.
    """
    encodings, labels = extract_instance_encodings_labels(input_filepath, corpus=corpus)

    dev_features = create_classifier_features(encodings)
    dev_labels = label_encoder.transform(labels)

    dev_labels = np.array(dev_labels).ravel()

    dev_features, dev_labels = shuffle(dev_features, dev_labels)

    predictions = model.predict(np.asarray(dev_features))

    return predictions, dev_labels


if __name__ == '__main__':

    with tqdm(total=len(train_paths), desc='Training...') as pbar:

        training_losses = []

        for idx, path in enumerate(train_paths):

            # if train_paths[idx+1].endswith('0.pickle'):
            #     print('Next one is 0')

            history = train_lstm(path)

            training_losses.append(history)

            pbar.update(1)

    try:
        model.save('../../data/models/lstm_classifier_two_sentence_instances.sav')
    except:
        print('Saving model failed.')

    # model = joblib.load('../../data/models/lstm_classifier_two_sentence_instances.sav')

    predictions = []
    true_labels = []

    with tqdm(total=len(dev_paths), desc='Evaluation on devset... ') as pbar:

        for idx, path in enumerate(dev_paths):

            preds, labels = predict_on_data(path)

            for pred, lab in zip(preds, labels):
                predictions.append(pred)
                true_labels.append(lab)

            pbar.update(1)

    true_labels = label_encoder.inverse_transform(true_labels)
    predictions = label_encoder.inverse_transform(predictions)
    # print(true_labels)

    report = classification_report(true_labels, predictions)

    print(report)
