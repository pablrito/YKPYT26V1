# YKPYT26V1
Assignment for YKPYT26V1

# How to run
pip install -r requirements.txt

./scripts/train_salary_prediction.py

# Steg 1: Välj problem och dataset
Valde att hämta dataset från kaggle med årsinkomster för olika yrken/klassifikationer.

Källa https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset

Column	Description
job_title	The job role or position (e.g., Data Analyst, AI Engineer)
experience_years	Number of years of professional experience
education_level	Highest level of education completed
skills_count	Number of technical or professional skills
industry	Industry sector where the job belongs
company_size	Size of the company (small, medium, large)
location	Job location or region
remote_work	Whether the job allows remote work
certifications	Number of professional certifications
salary	Annual salary of the employee

Total 250000 rader

Vill kunna förutsäga olika yrken årlöner genom testa olika klassifikationer
Dataset är regression då det är salary (numerisk), salary är vårat target 

# Steg 2: Förberedelse av data
Dataset ser fint ut, inga saknade värden, dock så måste kategorierna göras till numeriska värden ex

job_title             str
experience_years    int64
education_level       str
skills_count        int64
industry              str
company_size          str
location              str
remote_work           str
certifications      int64
salary              int64
dtype: object
Unique values in 'job_title': <StringArray>
[              'AI Engineer',              'Data Analyst',
        'Frontend Developer',          'Business Analyst',
           'Product Manager',         'Backend Developer',
 'Machine Learning Engineer',           'DevOps Engineer',
         'Software Engineer',     'Cybersecurity Analyst',
            'Data Scientist',            'Cloud Engineer']
Length: 12, dtype: str 

# Steg 3: Träna en maskininlärningsmodell

# Steg 4: Utvärdera modellen