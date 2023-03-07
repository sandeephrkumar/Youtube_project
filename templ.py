
from sklearn.datasets import load_iris
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler
data = load_iris()


X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale',MinMaxScaler()),
    #('std_scale', StandardScaler())
])
cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


grid_step_params = [{'col_trans__num_pipeline__minmax_scale': ['passthrough']},
                    {'col_trans__num_pipeline__std_scale': ['passthrough']}]



input_col_trans = ColumnTransformer(transformers=[
    ('num_pipeline',num_pipeline,[0,1,2,3] ),
    
    ],
    remainder='passthrough',
    n_jobs=-1)



output_col_trans = ColumnTransformer(transformers=[
    ('num_pipeline',cat_pipeline, [0]),
    
    ],
    remainder='drop',
    n_jobs=-1)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)

clf_pipeline1 = Pipeline(steps=[
    ('col_trans', input_col_trans),
    ('model', clf)
]) 

clf_pipeline1.fit(X_train, y_train)
# preds = clf_pipeline.predict(X_test)
score = clf_pipeline1.score(X_test, y_test)
print(f"Model score: {score}") # model accuracy
#y_train = output_col_trans.fit(y_train)

import joblib

# Save pipeline to file "pipe.joblib"
joblib.dump(clf_pipeline1,"pipe.joblib")

# Load pipeline when you want to use
same_pipe = joblib.load("pipe.joblib")

clf_pipeline1.get_params()


grid_params = {'model__penalty' : ['none', 'l2'],
               'model__C' : np.logspace(-4, 4, 20)}



from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(clf_pipeline1, grid_params, cv=5, scoring='accuracy')
gs.fit(X_train, y_train)

print("Best Score of train set: "+str(gs.best_score_))
print("Best parameter set: "+str(gs.best_params_))
print("Test Score: "+str(gs.score(X_test,y_test)))



