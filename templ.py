print("today is Tuesday")
print("content for git sandeep local")
# added additional comment 
for i in range(10):
    print(i)
from sklearn.datasets import load_iris
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
data = load_iris()


X = data.data
y = data.target




num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale',MinMaxScaler())
])
cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])






input_col_trans = ColumnTransformer(transformers=[
    ('num_pipeline',num_pipeline,[0,1] ),
    
    ],
    remainder='passthrough',
    n_jobs=-1)



output_col_trans = ColumnTransformer(transformers=[
    ('num_pipeline',cat_pipeline, [0]),
    
    ],
    remainder='drop',
    n_jobs=-1)


input_col_trans.fit_transform(X)[:4], X[:4]

num_pipeline.fit_transform(X)[:4], X[:4]
cat_pipeline.fit_transform(y.reshape(-1,1))
output_col_trans.fit_transform(y.reshape(-1,1))

t = [('num', SimpleImputer(strategy='median'), [0, 1]), ('cat', SimpleImputer(strategy='most_frequent'), [2, 3])]
transformer = ColumnTransformer(transformers=t)






from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0)
clf_pipeline = Pipeline(steps=[
    ('col_trans', col_trans),
    ('model', clf)
])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

le = preprocessing.LabelEncoder()
le.fit(y_train)








