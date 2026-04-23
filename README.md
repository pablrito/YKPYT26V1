# YKPYT26V1
Assignment 1

# How to run
```bash
pip install -r requirements.txt
python .\scripts\train_salary_prediction.py
```
# Steg 1: Välj problem och dataset
Valde att hämta dataset från kaggle med årsinkomster för olika yrken/klassifikationer.

Källa https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset

```bash
From kaggle
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

Total 250000 row and 10 col
```

Vill kunna förutsäga olika yrken årslöner genom testa olika klassifikationer, dataset är regression då det är salary (numerisk), salary är vårat target 

# Steg 2: Förberedelse av data
Dataset ser bra ut (kanske för bra), inga saknade värden, dock så måste kategorierna göras till numeriska värden.

Om det saknas värdern så kunde vi antigen tagit bort de raderna eller använd fill in metoden

```bash
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
```
Exempel så enkodas job_title till numeriska värden [0,1,2,3,4,5,6,7,8...], detta görs för samtliga kategorierna som ej är numeriska.

Våra feature o target som vi ska använda av oss för träning av modelen.

```bash
Features : 'job_title', 'experience_years', 'education_level', 'skills_count',
        'industry', 'company_size', 'location', 'remote_work', 'certifications'

Target : 'salary'
```
Vi vill kunna förutsäga yrken årslöner i dollars $$$

# Steg 3: Träna en maskininlärningsmodell
Trännade med 3 olika modeller, har testat med olika värden ex för RandomForestRegressor med antalet träd, 50-300 , efter 200 så tar det bara längre tid men ingen skillnad på R2.

Model: LinearRegression
MAE: $21,741
R2: 0.45597583

Model: DecisionTree
MAE: $7,595
R2: 0.93116522

Model: RandomForestRegressor 200 trees
MAE: $5,166
R2: 0.96929295

MAE genomsnittet hur mycket fel i dollars har den 
R2 hur mycket variation , mellan 0-1 , 1 är bäst

Datasetet är inte linjärt , lönen o erfarenheten ökar inte linjert , först året kanske du får en höjning men nästa år kanske det är tredubbla , men tredje är det samma ökning som första året etc.

Valde gå vidare med RandomForestRegressor har bästa värderna

Har en graf i output som visar resultat, ju närmar linjen plupparna är desto bättre

# Steg 4: Utvärdera modellen
Sparade ner RandomForestRegressor modellen men det är ca 1 G stor så jag simulerar istället att förutsäga vilken lön ett yrke med olika kategorier, dels att söka i datasetet o sedan att använda modellen men en dummy person.

Dessa features viktar av modellens beslut
location            0.33
experience_years    0.20
company_size        0.17
job_title           0.16
education_level     0.10
skills_count        0.03
certifications      0.01
industry            0.01
remote_work         0.01


```bash
python ./scripts/test_salary_prediction.py    

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
#en person som jobbar med Machine Learning Engineer, 5 år erfaranhet , Master

Predicted salary: $170,787
Similar salaries range: $70,328 - $277,135
Similar avg salaries  : $160,661

```