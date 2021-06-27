from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
import glob
import os
import tensorflow as tf
from tqdm import tqdm
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report



with open('../../data/models/lstm_classifier_two_sentence_instances.json', 'r') as infile:
    loaded_model = infile.read()

model = tf.keras.models.model_from_json(loaded_model)

def extract_instance_encodings_labels(input_filepath, corpus='train-conll-foreval'):
    """
    This function returns the correct sentence and token encodings
    for each training instance. The input should be the path to the
    instance files.
    """
    instance_encodings = []

    encoding_path =f'C:/Users/Myrthe/OneDrive/Documenten/VU/NLPT/NLPT_oud/data/encodings/polnear-conll/{corpus}'

       # f'../../../NLPT_oud/data/encodings/polnear-conll/{corpus}'

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


instance_paths = glob.glob(f'C:/Users/Myrthe/OneDrive/Documenten/VU/NLPT/NLPT_oud/data/instances/**/**/**')

test_paths = [
    path for path in instance_paths
    if 'test-conll-foreval' in path
]

classes = ['SOURCE', 'CUE', 'CONTENT', 'O']
label_encoder = LabelEncoder()
label_encoder.fit(classes)
all_label_encoder_classes = label_encoder.transform(label_encoder.classes_)  # total classes

def predict_on_data(input_filepath, model, corpus='test-conll-foreval'):
    """
    This function carries out predictions.
    """
    converted_predictions = []

    encodings, labels = extract_instance_encodings_labels(input_filepath, corpus=corpus)

    test_features = create_classifier_features(encodings)
    test_labels = label_encoder.transform(labels)

    test_labels = np.array(test_labels).ravel()

    test_features, test_labels = shuffle(test_features, test_labels)

    predictions = model.predict(np.asarray(test_features))

    for pred in predictions:
        label = np.argmax(pred)
        converted_predictions.append(label)

    return converted_predictions, test_labels


if __name__ == '__main__':

    predictions = []
    true_labels = []

    with tqdm(total=len(test_paths), desc='Evaluation on test... ') as pbar:

        for idx, path in enumerate(test_paths):

            preds, labels = predict_on_data(path, model)

            for pred, lab in zip(preds, labels):
                predictions.append(pred)  # TODO: predictions have too many dimensions, we need to reshape
                true_labels.append(lab)

            pbar.update(1)

    true_labels = label_encoder.inverse_transform(true_labels)
    predictions = label_encoder.inverse_transform(predictions)

    report = classification_report(true_labels, predictions)
    print(report)