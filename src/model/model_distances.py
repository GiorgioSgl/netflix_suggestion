import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..features.preprocessor import ColumnDropperTransformer, ColumnJoiner


class ModelSimilarity():

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        dropper = ColumnDropperTransformer(["title", "duration", "rating", "date_added"])
        joiner = ColumnJoiner(cols=["director", "cast", "listed_in"])

        preprocess = Pipeline([
            ("drop" ,dropper),
            ("joiner", joiner)
        ])

        X = preprocess.fit_transform(X)

        print(X.head())
