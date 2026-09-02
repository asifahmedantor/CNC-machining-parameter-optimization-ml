import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    df = pd.read_hdf(path)
    return df


def preprocess_data(df):

    encoder = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    df = df.dropna()

    return df