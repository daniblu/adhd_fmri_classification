'''
This script creates a data frame where each row is a subject and columns are demographic and phenotypic information. 
One column indicates whether the subject belongs in the training or test set.
'''

from pathlib import Path
import pandas as pd



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
    out_path = root / 'data' / 'processed' / 'pheno_data_train_test.csv'

    # list training subjects
    train_subjects = list_train_test_subjects(train_data_path)

    # list test subjects
    test_subjects = list_train_test_subjects(test_data_path)

    # load phenotypic data
    pheno_data = pd.read_csv(pheno_data_path, sep='\t')
    drop_columns = ['Secondary Dx', 'QC_Athena', 'QC_NIAK', 'Inattentive', 'Hyper/Impulsive', 'Med Status']
    pheno_data.drop(columns=drop_columns, inplace=True)

    # create ADHD column, 0: typically developing, 1: ADHD (Combined, Inattentive, Hyperactive/Impulsive)
    pheno_data['ADHD'] = pheno_data['DX'].apply(lambda x: 1 if x != '0' else 0) 

    # create column indicating whether subject is in training or test set
    pheno_data['Train'] = pheno_data['ScanDir ID'].isin(train_subjects)

    # if the ScanDir ID value contains only 5 digits, add two leading zeros
    pheno_data['ScanDir ID'] = pheno_data['ScanDir ID'].apply(lambda x: f'{x:07d}')

    # convert ID to string
    pheno_data['ScanDir ID'] = pheno_data['ScanDir ID'].astype(str)

    # save data frame
    pheno_data.to_csv(out_path, index=False)

    print(f'[DONE]: {out_path.name} saved at {out_path.parent}')
