
# coding: utf-8

# In[4]:

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class CustomLogisticClassifier(BaseEstimator, ClassifierMixin):  

    def __init__(self, seuil=0.9, penalty='l2', C=1.0, random_state=None, solver='liblinear',
                 max_iter=100, verbose=True, n_jobs=-1):
        
        clf = LogisticRegression(penalty=penalty, C=C, random_state=random_state, solver=solver,
                                 max_iter=max_iter, verbose=verbose, n_jobs=n_jobs)
        self.clf = clf
        self.seuil = seuil
        
        


    def fit(self, X, y=None):
        self.clf.fit(X,y)

        return self


    def predict(self, X, y=None):
        y_pred = self.clf.predict(X)
        y_predict_proba = self.clf.predict_proba(X)[:,0]
        
        for i in range(len(y_pred)):
            if (y_predict_proba[i]<self.seuil) and (y_predict_proba[i]>1-self.seuil):
                y_pred[i]=0

        return y_pred

    def score(self, X, y=None):
        y_pred = self.clf.predict(X)
        y_pred_unq =  np.unique(y_pred)
        for i in y_pred_unq:
            if((i != -1) & (i!= 1) & (i!= 0)):
                raise ValueError('The predictions can contain only -1, 1, or 0!')
        y_comp = y * y_pred
        score = float(10*np.sum(y_comp == -1) + np.sum(y_comp == 0))
        score /= y_comp.shape[0]
        return score


# In[5]:

from sklearn.svm import SVC
class CustomSVMClassifier(BaseEstimator, ClassifierMixin):  

    def __init__(self, seuil=0.9, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, 
                 probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=True, 
                 random_state=42):
        
        clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, 
                 probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, 
                 random_state=random_state)
        self.clf = clf
        self.seuil = seuil
        

    def fit(self, X, y=None):
        self.clf.fit(X,y)

        return self


    def predict(self, X, y=None):
        y_pred = self.clf.predict(X)
        y_predict_proba = self.clf.predict_proba(X)[:,0]
        
        for i in range(len(y_pred)):
            if (y_predict_proba[i]<self.seuil) and (y_predict_proba[i]>1-self.seuil):
                y_pred[i]=0

        return y_pred

    def score(self, X, y=None):
        y_pred = self.clf.predict(X)
        y_pred_unq =  np.unique(y_pred)
        for i in y_pred_unq:
            if((i != -1) & (i!= 1) & (i!= 0)):
                raise ValueError('The predictions can contain only -1, 1, or 0!')
        y_comp = y * y_pred
        score = float(10*np.sum(y_comp == -1) + np.sum(y_comp == 0))
        score /= y_comp.shape[0]
        return score


# In[6]:

from sklearn.neural_network import MLPClassifier

class CustomNeuralClassifier(BaseEstimator, ClassifierMixin):  

    def __init__(self, seuil=0.9,hidden_layer_sizes=(100, ),activation='relu', solver='adam', alpha=0.0001, 
                 batch_size='auto',verbose=True, max_iter=200, random_state=None, warm_start=False, 
                 momentum=0.9, nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation=activation, solver=solver, alpha=alpha, 
                            batch_size=batch_size, max_iter=200, random_state=random_state,verbose=verbose,
                            warm_start=warm_start,momentum=momentum, nesterovs_momentum=nesterovs_momentum, 
                            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        self.clf = clf
        self.seuil = seuil
        
        


    def fit(self, X, y=None):
        self.clf.fit(X,y)

        return self


    def predict(self, X, y=None):
        y_pred = self.clf.predict(X)
        y_predict_proba = self.clf.predict_proba(X)[:,0]
        
        for i in range(len(y_pred)):
            if (y_predict_proba[i]<self.seuil) and (y_predict_proba[i]>1-self.seuil):
                y_pred[i]=0

        return y_pred

    def score(self, X, y=None):
        y_pred = self.clf.predict(X)
        y_pred_unq =  np.unique(y_pred)
        for i in y_pred_unq:
            if((i != -1) & (i!= 1) & (i!= 0)):
                raise ValueError('The predictions can contain only -1, 1, or 0!')
        y_comp = y * y_pred
        score = float(10*np.sum(y_comp == -1) + np.sum(y_comp == 0))
        score /= y_comp.shape[0]
        return score


# In[ ]:

from sklearn.naive_bayes import GaussianNB

class CustomNaiveBayesClassifier(BaseEstimator, ClassifierMixin):  

    def __init__(self, seuil=0.9):
        
        clf = GaussianNB()
        self.clf = clf
        self.seuil = seuil
        

    def fit(self, X, y=None):
        self.clf.fit(X,y)
        return self


    def predict(self, X, y=None):
        y_pred = self.clf.predict(X)
        y_predict_proba = self.clf.predict_proba(X)[:,0]
        
        for i in range(len(y_pred)):
            if (y_predict_proba[i]<self.seuil) and (y_predict_proba[i]>1-self.seuil):
                y_pred[i]=0

        return y_pred

    def score(self, X, y=None):
        y_pred = self.clf.predict(X)
        y_pred_unq =  np.unique(y_pred)
        for i in y_pred_unq:
            if((i != -1) & (i!= 1) & (i!= 0)):
                raise ValueError('The predictions can contain only -1, 1, or 0!')
        y_comp = y * y_pred
        score = float(10*np.sum(y_comp == -1) + np.sum(y_comp == 0))
        score /= y_comp.shape[0]
        return score

