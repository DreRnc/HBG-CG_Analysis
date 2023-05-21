import numpy as np
import pandas as pd

def load_monk(file, labels):
    """
    Loads Monk Dataset
    """
    with open(file) as d:
        X = []
        y_true = []
        for line in d.readlines():
            line = line.lstrip()
            row = [int(x) for x in line.split(" ")[1:-1]]
            X.append(row)
            label = [int(line[0])]
            y_true.append(label)
        X = np.array(X, dtype="float16")
        y_true = np.array(y_true, dtype="float16")

        X = pd.DataFrame(X, columns=labels[1:])

        # one hot encoding of X_train and X_test
        X = pd.get_dummies(X, columns=labels[1:])
       
        return X.values, y_true

    
def load_MLCup(file, labels):
    """
    Loads MLCup Datasets

    """
    if len(labels) == 11:

        TR = pd.read_csv(file, sep = ',', header = None, usecols=range(1,12), names = labels, skiprows = 7)

        TR = TR.to_numpy()

        np.random.shuffle(TR)

        TR = np.split(TR, [9], axis = 1)

        X = TR[0]
        y_true = TR[1]

        return X, y_true
    
    if len(labels) == 9:

        TS = pd.read_csv(file, sep = ',', header = None, usecols=range(1,10), names = labels, skiprows = 7)

        TS = TS.to_numpy()

        np.random.shuffle(TS)

        return TS



