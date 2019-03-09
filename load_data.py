import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(random_state=42):
    train_csv = pd.read_csv('train.csv').set_index('ID_code')
    test_csv = pd.read_csv('test.csv').set_index('ID_code')
    
    Features = train_csv.drop(columns = 'target')
    target = train_csv.target
    X, X_dropout, y, y_dropout = train_test_split(Features, target, test_size=0.50, random_state=random_state)
    X_ensemble, X_dropout, y_ensemble, y_dropout = train_test_split(X_dropout, y_dropout, 
                                                                    test_size=0.50, random_state=random_state)
    return X, X_ensemble, X_dropout, y, y_ensemble, y_dropout, train_csv, test_csv
