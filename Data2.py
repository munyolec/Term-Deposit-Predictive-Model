import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

   
def scaler(df):
    ss=StandardScaler()
    df_scaled = pd.DataFrame(ss.fit_transform(df),columns = df.columns)
    return df_scaled

    
    


