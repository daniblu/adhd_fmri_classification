'''
This script performs hyperparameter tuning of a logistic lasso regression that predicts ADHD status from phenotypic data.
'''

print('[INFO]: Importing libraries.')
from pathlib import Path
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



def tune_hyperparameters(X, y):
    '''
    Tune hyperparameters for logistic lasso regression using grid search.
    '''
    param_grid = [
        {'C': [0.1, 1, 10, 50], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
    ]

    grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, scoring='f1_weighted', verbose=3)
    grid.fit(X, y)

    return grid.cv_results_, grid.best_score_, grid.best_estimator_, grid.best_params_



if __name__ == '__main__':

    # paths
    root = Path(__file__).parents[1]
    data_path = root / 'data' / 'processed' / 'pheno_train.pkl'

    # load data (two pd.DataFrames: X_train, y_train)
    with open(data_path, 'rb') as f:
        X_train_df, y_train_df = pickle.load(f)
    
    # convert to numpy arrays with dtype float
    X_train = X_train_df.values.astype(float)
    y_train = y_train_df.values.astype(float).ravel()

    print('[INFO]: Tuning hyperparameters.')
    cv_results, best_score, best_estimator, best_params = tune_hyperparameters(X_train, y_train)
    print(f'[INFO]: Best weighted F1 score: {best_score:.4f}.')

    # save results
    model_dir = root / 'models' / f'logistic_f1_{best_score:.4f}'
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / 'best_estimator.pkl', 'wb') as f:
        pickle.dump(best_estimator, f)

    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(model_dir / 'cv_results.csv', index=False)
    
    with open(model_dir / f'best_model.pkl', 'wb') as f:
        pickle.dump(best_estimator, f)
    
    with open(model_dir / 'best_params_and_coefs.txt', 'w') as f:
        f.write(f'Best parameters: {best_params}\n')
        f.write(f'Coefficients:\n')
        for i, coef in enumerate(best_estimator.coef_[0]):
            f.write(f'{X_train_df.columns[i]}: {coef}\n')

    print(f'[DONE]: Model tuned and saved to {model_dir}.')