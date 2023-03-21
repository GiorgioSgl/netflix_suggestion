from sklearn.base import BaseEstimator, TransformerMixin


## custom transformer sklearn
class ColumnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self


class ColumnJoiner(BaseEstimator, TransformerMixin):
    def __init__(self, sep=' ', cols=None):
        self.sep = sep

        if cols is None:
            self.cols = []
        else:
            self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.cols:
            X_[col] = X_[col].str.replace(' ', '').str.replace(',', self.sep)
        return X_
