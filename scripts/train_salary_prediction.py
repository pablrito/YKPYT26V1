
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
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

# Function to analyze the dataset
def analyse_data(df: pd.DataFrame) -> None:
    print(df.head(10))        # overview of 10 rows
    print(df.columns)         # show column names
    print(df.shape)           # show number of rows and columns
    print(df.info())          # show data types
    print(df.isna().sum())    # check for missing values
    print(df.dtypes)          # show data types of each column

# Function to encode text data to numeric values
def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    text_columns = ['job_title', 'education_level', 'industry', 
                'company_size', 'location', 'remote_work'] # this columns is not numeric, we need to encode it

    for col in text_columns:
        #print(f"Values in '{col}': {df[col].unique()}")  # print values in the column before encoding
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str)) # encode text to numeric
        
    df.to_csv(DATA_DIR /"encoded_job_salary_prediction_dataset.csv", index=False) # save encode dataset for future use

    return df

# Function to train and evaluate models
def train_models(df: pd.DataFrame) -> None:
    X = df[['job_title', 'experience_years', 'education_level', 'skills_count',
        'industry', 'company_size', 'location', 'remote_work', 'certifications']] # fearure values
    y = df['salary'] # target value
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42) # train and test split with 80% train and 20% test, random_state for 42
    print(f"Training: {len(X_train)}") #amount of data used for training
    print(f"Testing: {len(X_test)}") #amount of data used for testing
 
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
        "display_name": "RandomForestRegressor 200 trees",
        "model": RandomForestRegressor(n_estimators=200, random_state=42),
    }] # models to train and compare
   
    for config in model_configs:
        model = config["model"]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2  = r2_score(y_test, predictions)
        print(f"Model: {config['display_name']}")
        print(f"MAE: ${mae:,.0f}") # mean absoulte error in dollars , lower is better , how much avarage prediction is off from actual salary
        print(f"R2: {r2:.8f}") # R2 score, higher is better
   
        plot_data(y_test, predictions, config['display_name']) # plot actual vs predicted graph for each model

# Function to plot actual vs predicted salary    
def plot_data(y_test: pd.Series, predictions, model_name: str) -> None:
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title('Actual vs Predicted Salary for ' + model_name)
    plt.show()

# Main funtion
def main():
    df = load_data("job_salary_prediction_dataset.csv")
    analyse_data(df)
    df = encode_data(df)
    train_models(df)
    
if __name__ == "__main__":
    main()