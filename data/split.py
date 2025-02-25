#!/usr/bin/env python3
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split

def main():
    # ----- Configuration -----
    input_file = './data/annotated/all_fixed.csv'
    random_state = 42

    # Define the drug columns
    drug_cols = ['heroin', 'cocaine', 'methamphetamine', 'benzodiazepine',
                 'rx_opioid_misuse', 'cannabis', 'injection_drug_use', 'general_drug_use']

    # ----- Load Data -----
    df = pd.read_csv(input_file)

    # Create a helper column: True if any drug column is True, else False.
    df['has_drug'] = df[drug_cols].any(axis=1)

    # ----- Separate into Two Groups -----
    # Group of rows with any drug flag
    df_drug = df[df['has_drug']].copy()
    # Group of rows with no drugs (all drug columns False)
    df_no_drug = df[~df['has_drug']].copy()

    # ----- Split Each Group into Train (10%), Validation (10%), Test (80%) -----
    # For the drug group, we want to preserve the multi-label distribution.
    # First, split into a train_val (20%) and test (80%) set.
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=random_state)
    y_drug = df_drug[drug_cols].values
    for train_val_idx, test_idx in msss1.split(df_drug, y_drug):
        df_drug_train_val = df_drug.iloc[train_val_idx]
        df_drug_test = df_drug.iloc[test_idx]

    # Now split the train_val (which is 20% of the drug group) into train (50% of 20% => 10% overall) 
    # and validation (the other 50% of 20% => 10% overall).
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    y_drug_train_val = df_drug_train_val[drug_cols].values
    for train_idx, val_idx in msss2.split(df_drug_train_val, y_drug_train_val):
        df_drug_train = df_drug_train_val.iloc[train_idx]
        df_drug_val = df_drug_train_val.iloc[val_idx]

    # For the no-drug group (all values in drug_cols are False) we don’t need multi-label stratification.
    # We perform simple random splits with the same proportions.
    # First, split into train_val (20%) and test (80%).
    df_no_drug_train_val, df_no_drug_test = train_test_split(
        df_no_drug, test_size=0.8, random_state=random_state, shuffle=True)
    # Then split train_val evenly into train (10% overall) and validation (10% overall).
    df_no_drug_train, df_no_drug_val = train_test_split(
        df_no_drug_train_val, test_size=0.5, random_state=random_state, shuffle=True)

    # ----- Combine the Splits -----
    # For each split, combine drug and no-drug rows and shuffle the result.
    df_train = (pd.concat([df_drug_train, df_no_drug_train[:len(df_drug_train)]])
                  .sample(frac=1, random_state=random_state)
                  .reset_index(drop=True))
    df_val = (pd.concat([df_drug_val, df_no_drug_val[:len(df_drug_val)]])
                .sample(frac=1, random_state=random_state)
                .reset_index(drop=True))
    df_test = (pd.concat([df_drug_test, df_no_drug_test[:len(df_drug_test)]])
                 .sample(frac=1, random_state=random_state)
                 .reset_index(drop=True))

    # ----- Output Info and Save -----
    print("Total rows in original data:", len(df))
    print("Train rows:", len(df_train))
    print("Validation rows:", len(df_val))
    print("Test rows:", len(df_test))

    # Optionally, save the splits to CSV files
    df_train.to_csv('./data/multilabel/train.csv', index=False, encoding='utf-8')
    df_val.to_csv('./data/multilabel/val.csv', index=False, encoding='utf-8')
    df_test.to_csv('./data/multilabel/test.csv', index=False, encoding='utf-8')

if __name__ == '__main__':

    # python -m data.split

    main()
