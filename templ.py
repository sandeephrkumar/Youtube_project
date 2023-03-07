print("today is Tuesday")
print("content for git sandeep local")
# added additional comment 
for i in range(10):
    print(i)
from sklearn.datasets import load_iris
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
data = load_iris()


X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
le = preprocessing.LabelEncoder()






