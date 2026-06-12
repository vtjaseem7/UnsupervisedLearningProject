import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class FeatureEngineer(BaseEstimator,TransformerMixin):

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()

        X["UTILIZATION_RATIO"] = (
            X["BALANCE"] /
            (X["CREDIT_LIMIT"] + 1)
        )

        X["AVG_PURCHASE_VALUE"] = (
            X["PURCHASES"] /
            (X["PURCHASES_TRX"] + 1)
        )

        X["PAYMENT_RATIO"] = (
            X["PAYMENTS"] /
            (X["BALANCE"] + 1)
        )

        return X
 
        

    