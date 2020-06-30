import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from load_data import load_data
import pickle

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Transformation used to explicitly learn or select the columns from a data frame.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns==None:
            self.columns = list(X.columns)
        return self

    def transform(self, X):
        df = X[self.columns].copy()
        return df

    
class NumericFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Summary of class here.
    """
    
    def __init__(self, transformations = [], columns=[]):
        self.transformations = transformations
        self.columns = columns
            
    def fit(self, X, y=None):
        if self.columns==None:
            self.columns = list(X.columns)
        return self
    
    def transform(self, X):
        df = pd.DataFrame(index=X.index)
        for fcn in self.transformations:
            for col in self.columns:
                df[col + '*' +fcn.__name__] = X[col].apply(fcn).apply(float) 
        return df  


class PandasWrapper(BaseEstimator, TransformerMixin):
    """Summary of class here.
    """
    def __init__(self, transformation):
        self.transformation = transformation
            
    def fit(self, X, y=None):
        self.transformation.fit(X)
        return self
    
    def transform(self, X):
        df = pd.DataFrame(self.transformation.transform(X), columns = X.columns, index=X.index)
        return df   

class BucketDistributor(BaseEstimator, TransformerMixin):
    """Summary of class here.
    """
    
    def __init__(self, num_buckets = [], columns=[]):
        self.num_buckets = num_buckets
        self.columns = columns
            
    def fit(self, X, y=None):
        if self.columns==None:
            self.columns = list(X.columns)
        columns = self.columns
        num_buckets = self.num_buckets
        quantile_delta = 1.0/num_buckets
        quantile_vector = quantile_delta*np.linspace(0,num_buckets, num_buckets+1)
        self.quantiles_df = X[columns].quantile(quantile_vector)
        return self
    
    def transform(self, X):
        columns = self.columns
        num_buckets = self.num_buckets
        quantiles_df = self.quantiles_df
        df = pd.DataFrame(index=X.index)
        
        for current_column in columns:
            for current_bucket in range(1,num_buckets+1):
                feature_name = current_column + '*bucket' + str(current_bucket)
                current_lower_bound =  quantiles_df[current_column].values[current_bucket -1]
                current_upper_bound = quantiles_df[current_column].values[current_bucket]

                if current_lower_bound == quantiles_df.loc[0, current_column]:
                    current_lower_bound = -np.inf

                if current_upper_bound == quantiles_df.loc[1, current_column]:
                    current_upper_bound = np.inf
                fcn = lambda x: 1 if (current_lower_bound < x and x <= current_upper_bound) else 0
                df[feature_name]=X[current_column].apply(fcn)
        return df    
    
    
class ModelTransformer(BaseEstimator, TransformerMixin):
    """Transformation used to explicitly apply models coming from a dictionary of models
        to a dataframe.
    """
    def __init__(self, model_dictionary=None):
        assert model_dictionary!=None, 'Must specify a dictionary of models: {"model_name": model}' 
        self.model_dictionary = model_dictionary
        
    def fit(self, X, y=None): 
        return self
        
    def transform(self, X):
        df = pd.DataFrame(index=X.index)
        for model_name, model in self.model_dictionary.items():
            df[model_name] = pd.Series(model.predict_proba(X)[:,1], index=X.index).apply(float)
        return df
    
    
    

def create_base_pipeline(base_pipeline_description, cv=2):
    classifier, parameters, features, filename = base_pipeline_description
    
    pipeline_parameters = {}
    for parameter_name in parameters:
        pipeline_parameters['classifier__'+ parameter_name] = parameters[parameter_name]
    
    pipeline_steps = []
    pipeline_steps.append(('selector', ColumnSelector(columns=features)))
    pipeline_steps.append(('scaler', StandardScaler()))
    pipeline_steps.append(('classifier', classifier))
    pipeline_model = Pipeline(pipeline_steps)
    grid_model = GridSearchCV(pipeline_model, pipeline_parameters, cv=cv, scoring='roc_auc', return_train_score=True);
    return grid_model


### ONLY USED FOR MULTINOMIAL NAIVE BAYES
def create_positive_base_pipeline(base_pipeline_description, cv=2):
    classifier, parameters, features, filename = base_pipeline_description
    
    pipeline_parameters = {}
    for parameter_name in parameters:
        pipeline_parameters['classifier__'+ parameter_name] = parameters[parameter_name]
    
    pipeline_steps = []
    pipeline_steps.append(('selector', ColumnSelector(columns=features)))
    pipeline_steps.append(('scaler', MinMaxScaler())) ## REMOVE THIS LINE!!!
    pipeline_steps.append(('classifier', classifier))
    pipeline_model = Pipeline(pipeline_steps)
    grid_model = GridSearchCV(pipeline_model, pipeline_parameters, cv=cv, scoring='roc_auc', return_train_score=True);
    return grid_model




def create_assembled_pipeline(model_dictionary, topmodel, topmodel_parameters={}, cv=2):   
    pipeline_parameters = {}
    for parameter_name in topmodel_parameters:
        pipeline_parameters['topmodel__'+ parameter_name] = topmodel_parameters[parameter_name]
    
    pipeline_steps = []
    pipeline_steps.append(('ModelTransformer', ModelTransformer(model_dictionary=model_dictionary)))
    pipeline_steps.append(('scaler', StandardScaler())) ## for no reason.
    pipeline_steps.append(('topmodel', topmodel))
    assembled_model= Pipeline(pipeline_steps)
    assembled_pipeline = GridSearchCV(assembled_model, pipeline_parameters, cv=cv, scoring='roc_auc', return_train_score=True);
    return assembled_pipeline   


def study_performance_of_all_available_pickled_models(folder="./pickled_models"):
    all_models = load_all_available_pickled_models(folder=folder)
    X, X_ensemble, X_dropout, y, y_ensemble, y_dropout, train_csv, test_csv = load_data()    
    performance = []
    for model_name in all_models:
        p = list(_get_performance_metrics(all_models[model_name], X, X_ensemble, X_dropout, y, y_ensemble, y_dropout))
        performance.append(p)
    
    index_names = [n for n in all_models]
    column_names = ['train_score_cv', 'test_score_cv','train_score','ensemble_score','dropout_score']
    df = pd.DataFrame(performance, columns = column_names, index = index_names)
    return df


def load_all_available_pickled_models(folder="./pickled_models"):
    import os
    models = {}
    for filename in os.listdir(folder):
        if filename.endswith(".pkl"):
            models[filename] = pickle.load(open(folder+'/'+filename, 'rb'))
    return models


def _get_performance_metrics(model, X, X_ensemble, X_dropout, y, y_ensemble, y_dropout):
    index_best_model = model.best_index_
    cv_train_score = model.cv_results_['mean_train_score'][index_best_model]
    cv_test_score = model.cv_results_['mean_test_score'][index_best_model]
    train_score = model.score(X, y)
    ensemble_score = model.score(X_ensemble, y_ensemble)
    dropout_score = model.score(X_dropout, y_dropout)
    return cv_train_score, cv_test_score, train_score, ensemble_score, dropout_score


def pickle_pipeline(model, filename, folder='pickled_models/'):
    pickle.dump(model, open(folder + filename, "wb"))
    print filename + ' was created'
    return




