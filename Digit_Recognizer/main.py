import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler


def open_csv(path):
    return pd.read_csv(path)


def save_csv(predict_array, path):
    df = pd.DataFrame(np.mat(predict_array).T, columns=['Label'])
    df.index += 1
    df.index.name = 'ImageId'
    df.to_csv(path)


def predict(model, test):
    return model.predict(test)


def main():
    simple_df = open_csv('asserts/simples.csv')
    target_series = simple_df['label']
    simple_df = simple_df.drop("label", axis=1)

    simples = simple_df.values
    targets = target_series.values

    standard_scaler = StandardScaler()
    simples_std = standard_scaler.fit_transform(simples)

    # use 70% simples to training model

    train_number = int(simples_std.shape[0] * 0.8)
    simples_train = simples_std[:train_number]
    targets_train = targets[:train_number]
    simples_test = simples_std[train_number:]
    targets_test = targets[train_number:]

    knn = neighbors.KNeighborsClassifier()
    knn.fit(simples_train, targets_train)

    # log model accuracy
    print metrics.accuracy_score(y_true=targets_test, y_pred=predict(knn, simples_test))

    data_df = open_csv('asserts/data.csv')
    data_std = standard_scaler.transform(data_df.values)
    predict_result = predict(knn, data_std)
    save_csv(predict_result, 'asserts/predict.csv')


if __name__ == '__main__':
    main()
