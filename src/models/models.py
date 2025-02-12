from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
import pdb

class RegressionModel:
    def __init__(self):
        self.model = Ridge(alpha=1.0)

    def train(self, X, y):
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        self.model.fit(X, y)


class ElasticNetModel:
    def __init__(self):
        self.model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    
    def train(self, X, y):
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        self.model.fit(X, y)