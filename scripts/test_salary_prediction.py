import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

#load
model = joblib.load(DATA_DIR / "random_forest_model.pkl")

#mock data for testing  
person = pd.DataFrame([{
    'job_title'       : 6,   
    'experience_years': 5,
    'education_level' : 3,   
    'skills_count'    : 8,
    'industry'        : 8,   
    'company_size'    : 2,   
    'location'        : 7,   
    'remote_work'     : 2,   
    'certifications'  : 2
}])

predicted_salary = model.predict(person)
print(f"Predicted salary: ${predicted_salary[0]:,.0f}")

df = pd.read_csv(DATA_DIR / "encoded_job_salary_prediction_dataset.csv")
salary = df[
    (df['job_title'] == 6) &
    (df['experience_years'].between(4, 6)) &
    (df['education_level'].between(2, 3)) 
]['salary']

print(f"Similar people salary range: ${salary.min():,.0f} - ${salary.max():,.0f}")
print(f"Similar people avg salary  : ${salary.mean():,.0f}")