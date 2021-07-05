import glob
import os
import numpy as np
import joblib
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm


def extract_instance_encodings_labels(input_filepath, corpus='train-conll-foreval'):
    """
    This function returns the correct sentence and token encodings
    for each training instance. The input should be the path to the
    instance files.
    """
    instance_encodings = []

    # Path to saved BERT encodings
    encoding_path =f'../../data/encodings/polnear-conll/{corpus}'


    filename = os.path.basename(
        os.path.dirname(
            input_filepath
        )
    )

    # Loading instances
    instance = joblib.load(input_filepath)
    sentence_indices = instance[11].unique().tolist()

    # Saving labels
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

    # Indexing sentence
    sentence1 = instance_encodings[0]

    # CLS tokens, tokens
    cls1, tokens1 = sentence1[0], sentence1[1:]
    tokens = tokens1

    # Concatenating in one feature
    for token in tokens:
        token_rep = np.concatenate(
            (cls1, token), axis=None
        )
        token_rep = token_rep.reshape((1, token_rep.shape[0]))
        classifier_features.append(token_rep)

    return classifier_features

# Path to instances
instance_paths = glob.glob(f'../../data/instances/**/**/**')

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
model.add(Dense(4, activation='sigmoid'))  # four classes, four outputs
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def train_lstm(input_filepath, corpus='train-conll-foreval', epochs=4):
    """
    This function loads an article, and trains on the basis of it.
    """
    history = []
    arrayed_targets = []

    # Extracting encodings and labels
    encodings, labels = extract_instance_encodings_labels(input_filepath, corpus=corpus)

    # Defining the features and labels for training set
    train_features = create_classifier_features(encodings)
    train_labels = label_encoder.transform(labels)

    for lab in train_labels:
        label_array = np.zeros(4)
        label_array[lab] = 1
        arrayed_targets.append([label_array])

    train_features, train_labels = shuffle(train_features, np.asarray(arrayed_targets))

    for i in range(epochs):

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
    converted_predictions = []

    # Extracting encodings and labels for development set
    encodings, labels = extract_instance_encodings_labels(input_filepath, corpus=corpus)

    # Setting features for development set
    dev_features = create_classifier_features(encodings)
    dev_labels = label_encoder.transform(labels)

    dev_labels = np.array(dev_labels).ravel()

    dev_features, dev_labels = shuffle(dev_features, dev_labels)

    predictions = model.predict(np.asarray(dev_features))

    for pred in predictions:
        label = np.argmax(pred)
        converted_predictions.append(label)

    return converted_predictions, dev_labels


if __name__ == '__main__':

    with tqdm(total=len(train_paths), desc='Training...') as pbar:

        training_losses = []

        for idx, path in enumerate(train_paths):

            history = train_lstm(path)

            training_losses.append(history)

            pbar.update(1)

    # Saving the trained model to a json file
    model_json = model.to_json()

    with open('../../data/models/lstm_classifier_one_sentence_instances.json', 'w') as outfile:
        outfile.write(model_json)


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

     # Generating a classification report
    report = classification_report(true_labels, predictions)

    print(report)
