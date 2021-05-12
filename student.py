import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import numpy as np


data = pd.read_csv('C:/Users/TATAN/Downloads/student_scores - student_scores.csv')
feature = data['Hours']
target = data['Scores']

X_train , X_test, y_train, y_test = train_test_split(feature, target, random_state=1, test_size=0.2)
X_train = X_train.values.reshape(20, 1)
y_train = y_train.values.reshape(20, 1)

reg = LinearRegression()
reg.fit(X_train, y_train)

X_test = X_test.values.reshape(5, 1)
y_test = y_test.values.reshape(5, 1)

def study_score(n):
    """
    study_score(n = Not Null)
    ----------------------------------------------------------
    Parameters :
    n : No. of hours studied .
    returns : A float value, the corresponding marks obtained .
    """
    
    n = float(n)
    
    if n > 10 or n < 1:
        print("Unrealistic Figures")
        return
    
    arr = np.array([n]).reshape((1,1))
    print(f'Studying for {arr[0][0]} hours a student can score around {np.round((reg.predict(np.array([n]).reshape((1,1)))[0][0]), 2)}%')