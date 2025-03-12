from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from pythelpers.ml.tfidfvectorizer import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import dill
import h5py
from scipy import sparse


# Get the directory of the executing Python file
script_dir = Path(__file__).parent.resolve()
data_folder= Path(script_dir / "input")
dependence_path = str(data_folder / "ausw_dependence_{}.pkl")

datatrain_path = str(data_folder / "features4ausw4linearsvc_trainscaled.csv")
dataout_folder= str(script_dir / "output/features4ausw4linearsvc_trainsampled.h5")


def get_data():
    df = load_df(datatrain_path)
    vectorizer, selector = load_dependence()
    X, X_feature_names = get_x(df, vectorizer, selector)
    # X, X_feature_names = get_x2(df)
    y = df["impact"]
    
    return X, y, X_feature_names

def get_x(df, vectorizer, selector):
    # Define features and target variable
    X_vec1 = df.drop(columns=["impact", "combined_tks"])  # Drop non-numeric columns
    # Convert boolean columns to numeric (0 and 1)
    X_vec1 = X_vec1.astype(int)

    X_vec2 = vectorizer.transform(df['combined_tks'])
    X_vec2 = selector.transform(X_vec2)

    X_vec1_feature_names = X_vec1.columns
    # Get feature names from the vectorizer
    X_vec2_feature_names = vectorizer.get_feature_names_out()
    # Get the mask of selected features from SelectKBest
    X_vec2_selected_mask = selector.get_support()
    # Apply the mask to get the final selected feature names
    X_vec2_selected_feature_names = X_vec2_feature_names[X_vec2_selected_mask]

    X_feature_names = np.concatenate([X_vec1_feature_names, X_vec2_selected_feature_names])

    X = hstack([X_vec1, X_vec2])

    return X, X_feature_names

def get_x2(df):
    # without combined tks

    # Define features and target variable
    X_vec1 = df.drop(columns=["impact"])  # Drop non-numeric columns
    # Convert boolean columns to numeric (0 and 1)
    X_vec1 = X_vec1.astype(int)

    X_vec1_feature_names = X_vec1.columns

    X = hstack([sparse.csr_matrix(X_vec1)])

    return X, X_vec1_feature_names

def load_df(path):
    # Load the dataset
    df = pd.read_csv(datatrain_path)  # Replace with your actual file path
    df.drop(columns=["id"], inplace=True)
    
    return df

def load_dependence():
    with open(dependence_path.format("featureselector"), "rb") as f:
        selector = dill.load(f)
    vectorizer = TfidfVectorizer.load(dependence_path.format("vectorizer"))

    return vectorizer, selector


def save_data_withh5py(X_res, y_res):
    with h5py.File(dataout_folder, "w") as f:
        # Save sparse matrix X as a compressed dataset
        f.create_dataset("X_data", data=X_res.data)
        f.create_dataset("X_indices", data=X_res.indices)
        f.create_dataset("X_indptr", data=X_res.indptr)
        f.create_dataset("X_shape", data=X_res.shape)

        # Save y as a dense dataset
        f.create_dataset("y", data=y_res)


if __name__ == "__main__":
    X, y = get_data()
    print(X.shape)
    print(y.shape)
    print(X)
    print(y)