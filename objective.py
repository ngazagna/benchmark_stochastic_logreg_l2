import numpy as np


from benchopt.base import BaseObjective

from sklearn.model_selection import train_test_split

class Objective(BaseObjective):
    name = "L2 Logistic Regression"

    parameters = {
        'fit_intercept': [False],
        'lmbd': [1., 0.01]
    }

    def __init__(self, lmbd=.1, fit_intercept=False):
        self.lmbd = lmbd
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        # Train test split
        # self.random_state = random_state # attribute of the objective ?
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    def compute(self, beta):
        y_X_beta = self.y_test * self.X_test.dot(beta.flatten())
        l2 = 0.5 * np.dot(beta, beta)
        return np.log1p(np.exp(-y_X_beta)).sum() + self.lmbd * l2 # compute loss on test set

    def to_dict(self):
        return dict(X=self.X_train, y=self.y_train, lmbd=self.lmbd) # return train data to solver
