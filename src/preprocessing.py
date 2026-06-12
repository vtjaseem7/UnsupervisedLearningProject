import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin


#Building Custom Transformers

#1. Remove Customer ID
class RemoveCustomerID(BaseEstimator,TransformerMixin):

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):

        X = X.copy()

        return X.drop(columns=['CUST_ID'])
    

#2.LogTransformer

class LogTransformer(BaseEstimator,TransformerMixin):

    def __init__(self):
        
        self.log_cols = [
            "BALANCE",
            "PURCHASES",
            "ONEOFF_PURCHASES",
            "INSTALLMENTS_PURCHASES",
            "CASH_ADVANCE",
            "PAYMENTS",
            "MINIMUM_PAYMENTS",
            "CREDIT_LIMIT",
            "PURCHASES_TRX",
            "CASH_ADVANCE_TRX"
        ]

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):

        X = X.copy()

        for col in self.log_cols:

            X[col] = np.log1p(X[col])

        return X
    
                       

#2. MedianImputer (fill missing values)

class MedianImputer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.medians_ = X.median()

        return self
    
    def transform(self,X):
        X = X.copy()
        X = X.fillna(self.medians_)

        return X