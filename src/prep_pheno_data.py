'''
This script creates a data frame where each row is a subject and columns are demographic and phenotypic information. 
One column indicates whether the subject belongs in the training or test set.
'''

from pathlib import Path
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler



def list_train_test_subjects(path: Path) -> list:
    '''
    List all subjects recorded in csv files distributed across site folders at path.
    '''

    # empty list for storing subject IDs
    subjects = []
   
    # search path recursively for csv files and extract subject IDs
    for file in path.rglob('*phenotypic.csv'):
        
        # load csv file
        data = pd.read_csv(file)
        
        # get subject IDs
        try:
            subjects.extend(data['ScanDir ID'].values)
        except KeyError:
            subjects.extend(data['ScanDirID'].values)

    return subjects



if __name__ == '__main__':

    # paths
    root = Path(__file__).parents[1]
    pheno_data_path = root / 'data' / 'ADHD-200' / 'adhd200_preprocessed_phenotypics.tsv'
    train_data_path = root / 'data' / 'ADHD-200' / 'ADHD200_HO_TCs_filtfix'
    test_data_path = root / 'data' / 'ADHD-200' / 'ADHD200_HO_TCs_TestRelease'

    # list training subjects
    train_subjects = list_train_test_subjects(train_data_path)

    # list test subjects
    test_subjects = list_train_test_subjects(test_data_path)

    # load phenotypic data
    pheno_data = pd.read_csv(pheno_data_path, sep='\t')
    drop_columns = ['Secondary Dx', 'QC_Athena', 'QC_NIAK', 'Inattentive', 'Hyper/Impulsive', 'Med Status', 'Verbal IQ', 'Performance IQ', 'Full2 IQ']
    pheno_data.drop(columns=drop_columns, inplace=True)

    # create ADHD column, 0: typically developing (DX=0), 1: ADHD (DX=1,2,3), NaN: missing (DX is not 0,1,2, or 3)
    pheno_data['ADHD'] = pheno_data['DX'].apply(lambda x: 1 if x in ['1', '2', '3'] else 0 if x == '0' else None)

    # delete rows with missing ADHD status
    pheno_data.dropna(subset=['ADHD'], inplace=True)

    # create column indicating whether subject is in training or test set
    pheno_data['Train'] = pheno_data['ScanDir ID'].isin(train_subjects)

    # if the ScanDir ID value contains only 5 digits, add two leading zeros
    pheno_data['ScanDir ID'] = pheno_data['ScanDir ID'].apply(lambda x: f'{x:07d}')

    # convert ID to string to avoid leading zeros being removed
    pheno_data['ScanDir ID'] = pheno_data['ScanDir ID'].astype(str)

    # recode handedness (the values for NYU are based on the Edinbugh Handedness Inventory. It ranges from -1 to 1, with -1 being strong left-handed and 1 being strong right-handed)
    # 2 if in [2.0, 3.0], 1 if >=0, 0 if <0
    pheno_data['Handedness'] = pheno_data['Handedness'].replace('L', 0)
    pheno_data['Handedness'] = pheno_data['Handedness'].astype(float)
    pheno_data['Handedness'] = pheno_data['Handedness'].apply(lambda x: 2.0 if x in [2.0, 3.0] else 1.0 if x >= 0 else 0.0)

    # split
    X_train = pheno_data[pheno_data['Train'] == True].drop(columns=['ScanDir ID', 'Train', 'Site', 'DX', 'ADHD', 'ADHD Measure', 'ADHD Index', 'IQ Measure'])
    y_train = pheno_data[pheno_data['Train'] == True]['ADHD']

    X_test = pheno_data[pheno_data['Train'] == False].drop(columns=['ScanDir ID', 'Train', 'Site', 'DX', 'ADHD', 'ADHD Measure', 'ADHD Index', 'IQ Measure'])
    y_test = pheno_data[pheno_data['Train'] == False]['ADHD']

    # remove rows with NaN values, and where Full4 IQ is -999.0, and where handedness is 2
    X_train = X_train.dropna()
    X_train = X_train[X_train['Full4 IQ'] != -999.0]
    X_train = X_train[X_train['Handedness'] != 2.0]
    y_train = y_train[X_train.index]

    X_test = X_test.dropna()
    X_test = X_test[X_test['Full4 IQ'] != -999.0]
    X_test = X_test[X_test['Handedness'] != 2.0]
    y_test = y_test[X_test.index]

    # get a list of subject ids correspondig to the order of rows in y_test
    subject_ids = pheno_data[pheno_data['Train'] == False]['ScanDir ID']
    test_subject_order_pheno = subject_ids[X_test.index]

    # standardize
    scaler = StandardScaler()

    X_train[['Age', 'Full4 IQ']] = scaler.fit_transform(X_train[['Age', 'Full4 IQ']])
    X_test[['Age', 'Full4 IQ']] = scaler.transform(X_test[['Age', 'Full4 IQ']])

    # save processed phenotypic data
    pheno_data.to_csv(root / 'data' / 'processed' / 'pheno_data_train_test.csv', index=False)

    # save subject order
    with open(root / 'data' / 'processed' / 'test_subject_order_pheno.txt', 'w') as f:
        for subject in test_subject_order_pheno:
            f.write(f'{subject}\n')
    
    # save arrays
    with open(root / 'data' / 'processed' / 'pheno_train.pkl', 'wb') as f:
        pickle.dump((X_train, y_train), f)
    
    with open(root / 'data' / 'processed' / 'pheno_test.pkl', 'wb') as f:
        pickle.dump((X_test, y_test), f)

    print(f"[DONE]: Phenotypic data saved at {root / 'data' / 'processed'}\nShape of X_train: {X_train.shape}\nShape of X_test: {X_test.shape}\nShape of y_train: {y_train.shape}\nShape of y_test: {y_test.shape}")