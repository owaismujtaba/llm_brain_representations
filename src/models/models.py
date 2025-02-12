from sklearn.linear_model import Ridge

class Regression:
    def __init__(self):
        self.model = Ridge(alpha=1.0)

    def train(self, X, y):
        self.model.fit(X, y)