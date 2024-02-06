import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class LinReg():
    def __init__(self):
        pass

    def train(self, data, y):
        """Trains the Linear Regressor object

        Parameters
        ----------
        data : an `n x m` matrix of observations
            Data that should be used for training the model
        y : a vector of length `n` of predictions
            Regression values of observations

        Returns
        -------
        trained model
            Returns the trained model as a `LinearRegression` object
        """
        model = LinearRegression().fit(data, y)
        return model

    def get_fitness(self, x, y, rng=None):
        """Return the error of the trained model

        Parameters
        ----------
        x : an `n x m` matrix of
            Data that should be used for training the model
        y : a vector of length `n`
            Regression values of observarions
        rng : int, optional
            Random seed, by default None

        Returns
        -------
        float
            The square root of the MSE of the model
        """
        if rng is None:
            rng = np.random.default_rng().integers(1000)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            random_state=rng)
        model = self.train(x_train, y_train)
        predictions = model.predict(x_test)
        error = np.sqrt(mean_squared_error(predictions, y_test))

        return error

    def get_columns(self, X, bitstring):
        """Get columns of X according to bitstring

        Parameters
        ----------
        X : A `n x m` matrix
            Data that should be used for training the model
        bitstring : A binary vector of length `m`
            Binary vector indicating which columns to keep

        Returns
        -------
        np.array
            A smaller matrix, subset of `X`, containing only specified columns
        """
        indices = np.where(bitstring == 1)[0]
        return X[:, indices]
