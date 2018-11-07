import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Read .TSV files
df = pd.read_csv('../data/train.tsv', delimiter='\t').head(10000)
raw_phrases = np.array(df['Phrase'])
raw_sentiment = np.array(df['Sentiment'])


def bag_of_words(sentences):
    vector = CountVectorizer()
    vector.fit(sentences)
    bag = vector.transform(sentences)
    return bag.toarray()


def one_hot_encode(label):
    one_hot = np_utils.to_categorical(label, len(set(label)))
    return one_hot


def save_processed_data(mat, name):
    return np.save("../processed data/" + name + ".npy", mat)


phrases_train, phrases_test, sentiment_train, sentiment_test = train_test_split(bag_of_words(raw_phrases),
                                                                                one_hot_encode(raw_sentiment),
                                                                                test_size=0.20, random_state=0)
save_processed_data(phrases_train, "phr_train")
save_processed_data(phrases_test, "phr_test")
save_processed_data(sentiment_train, "sen_train")
save_processed_data(sentiment_test, "sen_test")
