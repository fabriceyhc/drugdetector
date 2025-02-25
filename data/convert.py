import pandas as pd
import os

def load_and_convert(file_path, text_column, label_columns, id_columns, add_false=False):
    """
    Loads a multi-label classification dataset, converts it to single-label format,
    and optionally adds missing False labels for all categories while preserving all ID columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if add_false:
        # Get all unique combinations of IDs and text
        base_df = df[id_columns + [text_column]].drop_duplicates()
        # Create a DataFrame of all possible labels
        labels_df = pd.DataFrame({"drug": list(label_columns)})
        # Create a full grid (cross join) of each text with every label
        cross_df = base_df.merge(labels_df, how="cross")

        # Melt the original dataframe to get the true labels only
        df_melt = df.melt(
            id_vars=id_columns + [text_column],
            value_vars=label_columns,
            var_name="drug",
            value_name="label"
        )
        df_true = df_melt[df_melt["label"] == 1]

        # Merge the full grid with the true labels so that existing True values remain True
        df_final = cross_df.merge(
            df_true, on=id_columns + [text_column, "drug"], how="left", suffixes=("", "_true")
        )
        df_final["label"] = df_final["label"].fillna(False).astype(bool)
        return df_final
    else:
        # If not adding False labels, simply melt and keep only True labels
        df_melt = df.melt(
            id_vars=id_columns + [text_column],
            value_vars=label_columns,
            var_name="drug",
            value_name="label"
        )
        return df_melt[df_melt["label"] == 1]


if __name__ == "__main__":

    # python -m data.convert

    import glob

    INPUT_DIR = "./data/multilabel"
    OUTPUT_DIR = "./data/singlelabel"

    label_map = {
        'Heroin Use': "heroin", 
        'Cocaine Use': "cocaine", 
        'Methamphetamine Use': "methamphetamine", 
        'Benzodiazepine Use': "benzodiazepine",
        'Prescription Opioids Misuse': "rx_opioid_misuse", 
        'Cannabis Use': "cannabis", 
        'Injection Drug Use': "injection_drug_use", 
        'General Drug Use': "general_drug_use"
    }

    file_paths = glob.glob(f"{INPUT_DIR}/*.csv")

    for file_path in file_paths:
        # Load multilabel dataset with proper formatting
        df = pd.read_csv(file_path, encoding='latin-1')
        df = df.rename(columns=label_map)
        for col in label_map.values():
            df[col] = df[col].astype(bool)
        print(f"[MAIN] Saving multilabel dataset to: {file_path}")
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        # Create single label version of the dataset with all (True and False) labels per text
        save_path = os.path.join(OUTPUT_DIR, os.path.basename(file_path))
        df_transformed = load_and_convert(
            file_path, 
            text_column="text", 
            label_columns=list(label_map.values()), 
            id_columns=["doc_id", "sentence_id"],
            add_false=True
        )
        print(f"[MAIN] Saving single label dataset to: {save_path}")
        df_transformed.to_csv(save_path, index=False, encoding='utf-8')
