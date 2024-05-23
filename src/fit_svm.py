'''
This script tunes hyperparameters for an SVM fitted to the training data using cross-validation.
The type of features used as training data are either eigenvector centralities or time course averages, which can be selected from the terminal.
'''

print('[INFO]: Importing libraries.')
from pathlib import Path
import argparse
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd



def input_parse():
    '''
    For parsing terminal arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", "-f", choices=['centrality', 'average'], required=True, help="Type of feature to use for training.")
    args = parser.parse_args()

    return args



def tune_hyperparameters(X, y):
    '''
    Tune hyperparameters for SVM using grid search.
    '''
    param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001]}
    ]

    grid = GridSearchCV(SVC(), param_grid, refit=True, scoring='f1_weighted', verbose=3)
    grid.fit(X, y)

    return grid.cv_results_, grid.best_score_, grid.best_estimator_, grid.best_params_



if __name__ == '__main__':
    
    args = input_parse()

    # paths
    root = Path(__file__).parents[1]
    data_path = root / 'data' / 'processed' / f'{args.feature}_features_train.pkl'

    # load data
    with open(data_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    y_train = y_train.ravel() 

    print('[INFO]: Tuning hyperparameters.')
    cv_results, best_score, best_estimator, best_params = tune_hyperparameters(X_train, y_train)
    print(f'[INFO]: Best weighted F1 score: {best_score:.4f}.')

    # save results
    model_dir = root / 'models' / f'svm_{args.feature}_f1_{best_score:.4f}'
    model_dir.mkdir(parents=True, exist_ok=True)

    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(model_dir / 'cv_results.csv', index=False)
    
    with open(model_dir / f'best_model.pkl', 'wb') as f:
        pickle.dump(best_estimator, f)
    
    with open(model_dir / 'best_params.txt', 'w') as f:
        f.write(str(best_params))
    
    print(f'[DONE]: Hyperparameters tuned and model saved to {model_dir}.')