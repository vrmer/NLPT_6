import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

# Fixing random state for reproducibility
np.random.seed(1995)

plt.style.use('seaborn')


def extract_features_labels(batches):
    """

    :param batches:
    :return:
    """
    gold_labels = []
    feature_list = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    for batch in batches:
        labels = batch.gold
        labels = np.array(labels).reshape(-1, 1)
        scaler.fit(labels)
        features = batch[[attribute for attribute in batch.columns
                          if attribute not in ['gold', 'time']]].values
        features = features.reshape((features.shape[0], 1, features.shape[1]))
        gold_labels.append(labels)
        feature_list.append(features)

    gold_labels = [scaler.transform(label) for label in gold_labels]

    return feature_list, gold_labels, scaler


def train_model(train_features, train_labels, model, epochs=20, verbose=2):
    """

    :param train_features:
    :param train_labels:
    :param model:
    :return:
    """
    histories = []

    train_features, train_labels = shuffle(train_features, train_labels)

    for i in range(epochs):
        for train_feature, train_label in zip(train_features, train_labels):
                history = model.train_on_batch(train_feature, train_label, reset_metrics=False)

        histories.append(history)

    return histories


# Read in the data
train_data = pd.read_csv('../data/train_RNN.csv')
train_data.fillna(float(0), inplace=True)
test_data = pd.read_csv('../data/test_RNN.csv')
test_data.fillna(float(0), inplace=True)

train_indices = train_data.loc[train_data.time == float(0)].index
test_indices = test_data.loc[test_data.time == float(0)].index

# Splitting into training and test batches
train_batches = []
test_batches = []

for e, j in enumerate(train_indices):
    if e == 0:
        train_batches.append(train_data[:j])
    else:
        train_batches.append(train_data[i+1:j])
    i = j

for e, j in enumerate(test_indices):
    if e == 0:
        test_batches.append(test_data[:j])
    else:
        test_batches.append(test_data[i+1:j])
    i = j

filtered_train_batches = []
filtered_test_batches = []

for batch in train_batches:
    filtered_batch = batch.drop(batch[batch.gold == 0.000].index)
    filtered_train_batches.append(filtered_batch)

for batch in test_batches:
    filtered_batch = batch.drop(batch[batch.gold == 0.000].index)
    filtered_test_batches.append(filtered_batch)

# Extracting training and test features, labels, and scalers
train_features, train_labels, train_scaler = extract_features_labels(filtered_train_batches)
test_features, test_labels, test_scaler = extract_features_labels(filtered_test_batches)

# Design network
model = Sequential()
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam')

# Training the model
train_history = train_model(train_features, train_labels, model, epochs=20)

prediction = model.predict(test_features[0])
print(prediction[0][0])

# Creating prediction and gold lists
user_predictions = []
user_golds = []
all_predictions = []
all_golds = []

for tst, gld in zip(test_features, test_labels):
    predict = model.predict(tst)
    predict = np.array(predict).reshape(-1, 1)
    prediction = test_scaler.inverse_transform(predict)
    gold = test_scaler.inverse_transform(gld)
    user_predictions.append(prediction)
    user_golds.append(gold)

    for predict, gld in zip(prediction, gold):
        all_predictions.append(predict)
        all_golds.append(gld)

evs = explained_variance_score(all_golds, all_predictions)
mae = mean_absolute_error(all_golds, all_predictions)  # MAE
mse = math.sqrt(mean_squared_error(all_golds, all_predictions))  # RMSE
r2 = r2_score(all_golds, all_predictions)

# Preparing gold and predictions
all_golds = np.array(all_golds)
print(all_golds)
# median = np.median(golds[golds > 0])
# golds[golds == 0] = median

# Creating elementwise mean squared errors for the gold and prediction values
scores = [math.sqrt(mean_squared_error(gold, prediction)) for gold, prediction in zip(all_golds, all_predictions)]

# golds = golds
all_predictions = np.array(all_predictions)
scores = np.array(scores)

# fig, ax = plt.subplots()
plt.plot(all_predictions, label='predictions', alpha=0.6)
plt.plot(all_golds, label='gold', alpha=0.6)
plt.plot(scores, label='losses', alpha=0.5)
# ax.set_ylabel([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.xlabel('Test day (users merged)')
plt.ylabel('Predicted value')
plt.title('Real and predicted values with loss (LSTM, 20 epochs)')
plt.legend()
plt.savefig('../data/lstm_output/tracked_values.pdf')
plt.show()

print(f'''Explained variance score: {evs}
Mean absolute error: {mae}
Root mean squared error: {mse}
R2 score: {r2}''')

evaluations = {'Explained variance score': [evs],
               'Mean absolute error': [mae],
               'Root mean squared error': [mse],
               'R2 score': [r2]}

evaluations_df = pd.DataFrame.from_dict(evaluations)
evaluations_df.to_csv('../data/lstm_output/lstm_evaluations.csv')

user_predicted = []

for prediction, gold in zip(user_predictions, user_golds):
    predicted = math.sqrt(mean_squared_error(gold, prediction))
    user_predicted.append(predicted)

plt.bar(range(len(user_predicted)), user_predicted)
plt.xlabel('User index')
plt.ylabel('RMSE loss for user')
plt.title('Loss for each user')
plt.savefig('../data/lstm_output/loss_per_user.pdf')
plt.show()
