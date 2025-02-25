import pandas as pd

CONST_COLS = ["doc_id", "sentence_id", "text"]
LABEL_MAP = {
    "heroin": 'Heroin Use_Actual', 
    "cocaine": 'Cocaine Use_Actual', 
    "methamphetamine": "Methamphetamine Use_Actual", 
    "benzodiazepine": "Benzodiazepine Use_Actual",
    "rx_opioid_misuse": "Prescription Opioids Misuse_Actual", 
    "cannabis": "Cannabis Use_Actual", 
    "injection_drug_use": "Injection Drug Use_Actual", 
    "general_drug_use": "General Drug Use_Actual"
}

def process_excel_to_csv(excel_filename):
    # Read all sheets from the Excel file into a dictionary
    sheets = pd.read_excel(excel_filename, sheet_name=None)
    
    processed_frames = []
    
    # Process each sheet by inserting a 'drug' column with the sheet name
    for drug, df in sheets.items():
        # Insert the drug name into each row
        df.insert(0, 'drug', drug)
        # Apply the annotation processing row-wise
        df_fixed = df.apply(process_annotations, axis=1)
        processed_frames.append(df_fixed)
    
    # Concatenate all processed rows vertically
    combined_df = pd.concat(processed_frames, ignore_index=True)

    # Apply merging function to groups defined by CONST_COLS
    combined_df = combined_df.groupby(CONST_COLS).apply(merge_rows).reset_index(drop=True)
    
    return combined_df[CONST_COLS + list(LABEL_MAP.keys())]

def process_annotations(row):
    # Get the corresponding column name for the drug
    target_col = LABEL_MAP[row['drug'].lower()]

    # Determine the value based on whether there's a misclassification reason
    value = not row[target_col] if pd.isna(row['Misclassification Reason']) else row[target_col]

    # Get other drugs
    other_vals = {key: row[value] for key, value in LABEL_MAP.items() if value != target_col}
    
    return pd.Series({
        "doc_id": row["doc_id"],
        "sentence_id": row["sentence_id"],
        "text": row["text"],
        row['drug']: value,
        **other_vals,
        "override": row['drug'].lower()
    })

# Function to merge rows based on override column
def merge_rows(group):
    # Start with all False values
    merged_row = group.iloc[0].copy()
    
    # Iterate over rows to apply overrides
    for _, row in group.iterrows():
        if row["override"] in group.columns:
            merged_row[row["override"]] = row[row["override"]]

    return merged_row

def update_rows(df_base, df_updated):

    # Set the key columns as the index for both dataframes
    df_base.set_index(CONST_COLS, inplace=True)
    df_updated.set_index(CONST_COLS, inplace=True)

    # Update df_base with the values from df_updated (this updates in-place)
    df_base.update(df_updated)

    # If you need the keys back as columns, reset the index
    df_base.reset_index(inplace=True)

    return df_base


if __name__ == '__main__':

    # python -m data.merge_error_annotations

    error_analysis_df = process_excel_to_csv('./data/annotated/error_analysis.xlsx')
    error_analysis_df.to_csv('./data/annotated/error_analysis.csv', index=False, encoding='utf-8')
    print("error_analysis_df:")
    print(error_analysis_df)
    print("Columns:", error_analysis_df.columns)

    df_base = pd.read_csv('./data/multilabel/all.csv')

    # Create a copy of df_base before updating
    df_base_before = df_base.copy()

    for split in ['train', 'val', 'test']:
        path = f'./data/multilabel/{split}.csv'
        print(f'Updating base_df with {path}')
        df_updated = pd.read_csv(path)
        df_base = update_rows(df_base, df_updated)
        
        # Count number of rows that changed
        num_different_rows = (df_base_before != df_base).any(axis=1).sum()
        print(f"Number of rows that changed: {num_different_rows}")

    print(f'Updating base_df with error_analysis_df')
    df_base = update_rows(df_base, error_analysis_df)
    print(df_base)

    # Count number of rows that changed
    num_different_rows = (df_base_before != df_base).any(axis=1).sum()
    print(f"Number of rows that changed: {num_different_rows}")

    print(f"Saving all_fixed!")
    df_base.to_csv('./data/annotated/all_fixed.csv', index=False, encoding='utf-8')
