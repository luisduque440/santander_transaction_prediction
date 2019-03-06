import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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

    
class ModelTransformer(BaseEstimator, TransformerMixin):
    """Transformation used to explicitly learn or select the columns from a data frame.
    """
    def __init__(self, model_dictionary=None):
        assert model_dictionary!=None, 'you must specify a dictionary of models of the form {"model_name": model}' 
        self.model_dictionary = model_dictionary
        
    def fit(self, X, y=None): ## is this necessary ?? it might not be necessary, I feel.
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



def study_performance_of_all_available_pickled_models(X_dropout, y_dropout, folder="./pickled_models"):
    all_models = load_all_available_pickled_models(folder=folder)
    names = []
    train_scores = []
    test_scores = []
    dropout_scores = []
    for model_name in all_models:
        tr, ts,  dr = _get_performance_metrics_base_pipeline(all_models[model_name], X_dropout, y_dropout)
        names.append(model_name)
        train_scores.append(tr)
        test_scores.append(ts)
        dropout_scores.append(dr)
    return pd.DataFrame({'mean_train_score_cv': train_scores,
                         'mean_test_score_cv' : test_scores, 
                         'dropout_score': dropout_scores}, index=names)


def load_all_available_pickled_models(folder="./pickled_models"):
    import os
    models = {}
    for filename in os.listdir(folder):
        if filename.endswith(".pkl"):
            models[filename] = pickle.load(open(folder+'/'+filename, 'rb'))
    return models


def _get_performance_metrics_base_pipeline(model, X_dropout, y_dropout):
    index_best_model = model.best_index_
    cv_train_score = model.cv_results_['mean_train_score'][index_best_model]
    cv_test_score = model.cv_results_['mean_test_score'][index_best_model]
    dropout_score = model.score(X_dropout, y_dropout)
    return cv_train_score, cv_test_score,  dropout_score


def pickle_pipeline(model, filename, folder='pickled_models/'):
    pickle.dump(model, open(folder + filename, "wb"))
    print filename + ' was created'
    return




