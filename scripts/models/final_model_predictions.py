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
from sklearn.metrics import confusion_matrix
import pandas as pd


with open('../../data/models/lstm_classifier_one_sentence_instances.json', 'r') as infile:
    loaded_model = infile.read()

model = tf.keras.models.model_from_json(loaded_model)


def extract_instance_encodings_labels(input_filepath, corpus='train-conll-foreval', return_instance=False):
    """
    This function returns the correct sentence and token encodings
    for each training instance. The input should be the path to the
    instance files.
    """
    instance_encodings = []

    encoding_path =f'C:/Users/Myrthe/OneDrive/Documenten/VU/NLPT/NLPT_oud/data/encodings/polnear-conll/{corpus}'
    #encoding_path = f'../../data/encodings/polnear-conll/{corpus}'

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

    if return_instance is True:
        return instance_encodings, labels, instance
    else:
        return instance_encodings, labels


def create_classifier_features(instance_encodings):
    """
    This function creates [[CLS1] [CLS2] [token_n]]
    features that can serve as input to a classifier.
    """
    classifier_features = []

    sentence1 = instance_encodings[0]
    #sentence2 = instance_encodings[1]

    # CLS tokens, tokens
    cls1, tokens1 = sentence1[0], sentence1[1:]
    #cls2, tokens2 = sentence2[0], sentence2[1:]

    tokens = tokens1

    for token in tokens:
        token_rep = np.concatenate(
            (cls1, token), axis=None
        )
        token_rep = token_rep.reshape((1, token_rep.shape[0]))
        classifier_features.append(token_rep)

    return classifier_features

#instance_paths = glob.glob(f'C:/Users/Myrthe/OneDrive/Documenten/VU/NLPT/NLPT_6/data/instances/**/**/**')
instance_paths = glob.glob('../../data/instances/**/**/**')

test_paths = [
    path for path in instance_paths
    if 'test-conll-foreval' in path
]

classes = ['SOURCE', 'CUE', 'CONTENT', 'O']
label_encoder = LabelEncoder()
label_encoder.fit(classes)
all_label_encoder_classes = label_encoder.transform(label_encoder.classes_)  # total classes


def predict_on_data(input_filepath, model, corpus='test-conll-foreval', output_for_instance=None):
    """
    This function carries out predictions.
    """
    converted_predictions = []

    if output_for_instance is None:
        encodings, labels = extract_instance_encodings_labels(input_filepath, corpus=corpus)
    else:
        encodings, labels, instance = extract_instance_encodings_labels(
            input_filepath, corpus=corpus, return_instance=True)

    test_features = create_classifier_features(encodings)
    test_labels = label_encoder.transform(labels)

    test_labels = np.array(test_labels).ravel()

    test_features, test_labels = shuffle(test_features, test_labels)

    predictions = model.predict(np.asarray(test_features))

    for pred in predictions:
        label = np.argmax(pred)
        converted_predictions.append(label)

    converted_predictions = label_encoder.inverse_transform(converted_predictions)

    if output_for_instance is not None:

        article_name = os.path.basename(
            os.path.dirname(
                input_filepath
            )
        )

        filename = os.path.basename(input_filepath)

        directory_to_create = f'../../data/final_output/{corpus}/{article_name}'

        try:
            os.mkdir(directory_to_create)
        except FileExistsError:
            pass

        output_path = directory_to_create + '/' + filename

        instance[12] = converted_predictions
        joblib.dump(instance, output_path)

    return converted_predictions, test_labels


if __name__ == '__main__':

    predictions = []
    true_labels = []

    with tqdm(total=len(test_paths), desc='Evaluation on test... ') as pbar:

        for idx, path in enumerate(test_paths):

            preds, labels = predict_on_data(path, model, output_for_instance=True)

            for pred, lab in zip(preds, labels):
                predictions.append(pred)
                true_labels.append(lab)

            pbar.update(1)

    true_labels = label_encoder.inverse_transform(true_labels)
    # predictions = label_encoder.inverse_transform(predictions)

    report = classification_report(true_labels, predictions)

    data = {'Gold': true_labels, 'Predicted': predictions}
    df = pd.DataFrame(data, columns=['Gold', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print(confusion_matrix)

    print(report)

