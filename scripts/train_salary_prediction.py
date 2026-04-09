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


def main():
    df = load_data("job_salary_prediction_dataset.csv")
    analyse_data(df)

if __name__ == "__main__":
    main()