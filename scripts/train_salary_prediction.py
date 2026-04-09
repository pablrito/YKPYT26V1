
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

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
        #print(f"Values in '{col}': {df[col].unique()}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        df.to_csv(DATA_DIR /"encoded_job_salary_prediction_dataset.csv", index=False)

    return df

def train_model(df: pd.DataFrame) -> None:
    X = df[['job_title', 'experience_years', 'education_level', 'skills_count',
        'industry', 'company_size', 'location', 'remote_work', 'certifications']]
    y = df['salary']
    # 80% train o 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"Training: {len(X_train)}")
    print(f"Testing: {len(X_test)}")
 
    model_configs = [
    {
        "display_name": "LinearRegression",
        "model": LinearRegression(),
    },
    {   
        "display_name": "DecisionTree",          
        "model": DecisionTreeRegressor(random_state=42)
    },
    {
        "display_name": "RandomForestRegressor 50 trees",
        "model": RandomForestRegressor(n_estimators=50, random_state=42),
    }]
     
    best_model      = None
    best_model_name = None
    best_r2         = -1

    for config in model_configs:
        model = config["model"]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2  = r2_score(y_test, predictions)
        print(f"Model: {config['display_name']}")
        print(f"MAE: ${mae:,.0f}")
        print(f"R2: {r2:.8f}")

        # save best model from loop
        if r2 > best_r2:
            best_r2         = r2
            best_model      = model
            best_model_name = config['display_name']

        plot_data(y_test, predictions, config['display_name'])

    # Save best model 
    print(f"\nBest model: {best_model_name} (R2: {best_r2:.8f})")
    joblib.dump(best_model, DATA_DIR / "test_salary_model.pkl")
  
     
def plot_data(y_test: pd.Series, predictions, model_name: str) -> None:
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title('Actual vs Predicted Salary for ' + model_name)
    plt.show()

def main():
    df = load_data("job_salary_prediction_dataset.csv")
    analyse_data(df)
    df = encode_data(df)
    #df = load_data("encoded_job_salary_prediction_dataset.csv")
    train_model(df)
    
if __name__ == "__main__":
    main()