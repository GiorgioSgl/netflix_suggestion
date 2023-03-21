import pandas as pd
from model.model_distances import ModelDistances
from os import path
import pickle


def train():
    data_folder = path.join("..", "data")
    data_filename = "data.csv"

    model_folder = path.join("..", "model")
    model_filename = "model.pickle"

    df = pd.read_csv(path.join(data_folder,
                               data_filename))

    model = ModelDistances()

    a = model.fit(df)

    print(a.head())

    with open(path.join(model_folder, model_filename)) as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train()