import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    """
    Load raw machining dataset
    """
    df = pd.read_csv(path)
    return df



def preprocess_data(df):
    """
    Data preprocessing:
    - Encode categorical columns
    - Remove missing values
    """

    encoder = LabelEncoder()

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col].astype(str))

    # Remove missing values
    df = df.dropna()

    return df



if __name__ == "__main__":

    # Load dataset
    df = load_data(
        "data/processed/master_machining_dataset.csv"
    )

    print("Original data loaded successfully")
    print(df.head())


    # Preprocess data
    processed_df = preprocess_data(df)


    # Save processed dataset
    processed_df.to_csv(
        "data/processed/processed_data.csv",
        index=False
    )


    print("Preprocessing completed successfully")
    print("Processed data saved:")
    print("data/processed/processed_data.csv")