import numpy as np
import pandas as pd

TRFILE = "data/ML-CUP22-TR.csv"
TSFILE = "data/ML-CUP22-TS.csv"


def readTrain(type="regression"):
    with open(TRFILE, "r") as f:
        reader = pd.read_csv(f)
    data = reader.values
    if type == "regression":
        X = pd.DataFrame(data[:, 1:-2])
        y1 = pd.DataFrame(data[:, -2])
        y2 = pd.DataFrame(data[:, -1])
    elif type == "classification":
        X = pd.DataFrame(data[:, 1:-2])
        median1 = np.median(data[:, -2])
        median2 = np.median(data[:, -1])
        data[:, -2] = np.where(data[:, -2] > median1, 1, -1)
        data[:, -1] = np.where(data[:, -1] > median2, 1, -1)
        y1 = pd.DataFrame(np.sign(data[:, -2]))
        y2 = pd.DataFrame(np.sign(data[:, -1]))
    return X, y1, y2


def readTest(type="regression"):
    with open(TSFILE, "r") as f:
        reader = pd.read_csv(f)
    data = reader.values
    if type == "regression":
        X = pd.DataFrame(data[:, 1:])
    elif type == "classification":
        X = pd.DataFrame(data[:, 1:])
    return X


if __name__ == "__main__":
    X, y1, y2 = readTrain("classification")
