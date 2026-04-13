import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

#mock data for testing  
person = pd.DataFrame([{
    'job_title'       : 0,   
    'experience_years': 5,
    'education_level' : 4,    
    'skills_count'    : 12,
    'industry'        : 9,   
    'company_size'    : 2,   
    'location'        : 7,   
    'remote_work'     : 2,   
    'certifications'  : 2
}])

df = pd.read_csv(DATA_DIR / "encoded_job_salary_prediction_dataset.csv")
X = df[['job_title', 'experience_years', 'education_level', 'skills_count',
        'industry', 'company_size', 'location', 'remote_work', 'certifications']] # fearure values
y = df['salary'] # target value

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42) #RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
results = pd.DataFrame({
    'Actual'    : y_test.values[:5],
    'Predicted' : y_pred[:5].round(0),
    'Diff': (y_test.values[:5] - y_pred[:5]).round(0)
})

print(results.to_string(index=False))
print(f"MAE : ${mean_absolute_error(y_test, y_pred):,.0f}")
print(f"R2  : {r2_score(y_test, y_pred):.8f}")

predicted_salary = model.predict(person)
print(f"Predicted salary: ${predicted_salary[0]:,.0f}")

salary = df[
    (df['job_title'] == 0) &
    (df['experience_years'].between(4, 6)) 
]['salary']
#print(salary);
#print(salary.describe())
print(f"Similar salaries range: ${salary.min():,.0f} - ${salary.max():,.0f}")
print(f"Similar avg salaries  : ${salary.mean():,.0f}")

filtered = df[
    (df['job_title'] == 0) &
    (df['experience_years'].between(4, 6))
]

salary_highest = filtered.loc[filtered['salary'].idxmax()]
print(salary_highest) # Name: 135581, dtype: int64 , row with highest salary with similar job 
#AI Engineer,5,PhD,12,Telecom,Enterprise,USA,Hybrid,5,277135