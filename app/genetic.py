import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

DATASET = None
TARGET = ""
MODEL = None

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.columns:
        if data[column].dtype == 'object':
            if len(data[column].unique()) == 2:
                data[column] = pd.Categorical(data[column]).codes
            else:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])

    return data

def main(csv: str, target: str, model: int):
    global DATASET, TARGET, MODEL
    DATASET = pd.read_csv(csv)
    TARGET = target

    if model == 0:
        MODEL = DecisionTreeClassifier()
    elif model == 1:
        MODEL = RandomForestClassifier()
    elif model == 2:
        MODEL = SVC()
    elif model == 3:
        MODEL = KNeighborsClassifier()

    DATASET = preprocess(DATASET)

    target = DATASET[TARGET]
    predictors = DATASET.drop(TARGET, axis=1)
