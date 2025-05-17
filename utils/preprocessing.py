import pandas as pd

def load_data(path, limit=None):
    df = pd.read_csv(path)
    df = df.dropna()

    # Drop source column
    df = df.drop("source", axis=1)
    
    if limit:
        df = df.head(limit)

    return df


def get_example(df):
    return list(zip(df["ingredients"], df["title"]))





