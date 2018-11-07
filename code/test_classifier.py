import os

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

features = np.load("../processed data/phr_test.npy")
labels = np.load("../processed data/sen_test.npy")
model = "../models/classifer.h5"
df = pd.read_csv('../data/train.tsv', delimiter='\t').head(10000)
class_names = set(df['Phrase'])


def test_acc(fea, lab, mod):
    try:
        neural_nework = load_model(mod)
    except:
        print("model not found")
    prediction = neural_nework.evaluate(x=fea, y=lab, verbose=1, batch_size=64)
    return prediction, neural_nework


pred, nn = test_acc(features, labels, model)
print("accuracy", pred[1] * 100)
print("loss", pred[0])
