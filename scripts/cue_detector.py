import pickle

from tqdm import tqdm

from sklearn.metrics import classification_report

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB


with open('../data/cue_detection/train_features.pkl', 'rb') as infile:
    train_features = pickle.load(infile)

with open('../data/cue_detection/train_labels.pkl', 'rb') as infile:
    train_labels = pickle.load(infile)

with open('../data/cue_detection/dev_features.pkl', 'rb') as infile:
    dev_features = pickle.load(infile)

with open('../data/cue_detection/dev_labels.pkl', 'rb') as infile:
    dev_labels = pickle.load(infile)


feature_vectorizer = DictVectorizer()
target_encoder = LabelEncoder()

transformed_train_features = feature_vectorizer.fit_transform(train_features)
transformed_dev_features = feature_vectorizer.transform(dev_features)

transformed_train_labels = target_encoder.fit_transform(train_labels)
transformed_dev_labels = target_encoder.transform(dev_labels)

models = {
    # 'forest': RandomForestClassifier(),
    'svm': LinearSVC(),
    'logreg': LogisticRegression(),
    'nb': ComplementNB()
}

model_path = '../data/output/cue_detector_outputs/'

with tqdm(total=len(models.keys()), desc='Train and evaluate models: ') as pbar:

    for model_name, model in models.items():

        model.fit(transformed_train_features, transformed_train_labels)
        print(f'{model_name} fitted...')

        predictions = model.predict(transformed_dev_features)
        gold = transformed_dev_labels

        report = classification_report(gold, predictions)

        output_path = model_path + model_name + '.txt'

        with open(output_path, 'w') as outfile:
            outfile.write(report)

        pbar.update(1)
