from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def load_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / filename)
    return df

def analyse_data(df: pd.DataFrame) -> None:
    print(df.head(10))        # overview of 10 rows
    print(df.columns)         # show column names
    print(df.shape)           # show number of rows and columns
    print(df.info())          # show data types
    print(df.isna().sum())    # check for missing values
    print(df.dtypes)          # show data types of each column

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    text_columns = ['job_title', 'education_level', 'industry', 
                'company_size', 'location', 'remote_work']

    for col in text_columns:
        print(f"Unique values in '{col}': {df[col].unique()}")

    return df

def main():
    df = load_data("job_salary_prediction_dataset.csv")
    analyse_data(df)
    df = encode_data(df)
   

if __name__ == "__main__":
    main()