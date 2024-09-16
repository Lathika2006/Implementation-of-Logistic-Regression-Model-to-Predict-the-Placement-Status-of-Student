# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Get the data and use label encoder to change all the values to numeric.
Drop the unwanted values,Check for NULL values, Duplicate values.
Classify the training data and the test data.
Calculate the accuracy score, confusion matrix and classification report.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: LATHIKA LJ

RegisterNumber:  212223220050

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
*/
```

## Output:
![image](https://github.com/user-attachments/assets/920775f0-427e-4568-afad-ef12eeb02c07)

![image](https://github.com/user-attachments/assets/d8d30dda-1499-4ca9-b0e9-4e8031133b74)

![image](https://github.com/user-attachments/assets/410fb885-3ec4-4c5b-89b8-715e09bf447b)

![image](https://github.com/user-attachments/assets/b243c8be-0486-4dfd-9d90-457dda38003f)

![image](https://github.com/user-attachments/assets/0e84a34f-afc0-48f0-88a5-dff0547f560f)

![image](https://github.com/user-attachments/assets/816ae8e9-b912-4cd5-ae62-bdd6737fc749)

![image](https://github.com/user-attachments/assets/3d52a343-46ac-43c1-bb09-d52a7725257f)

![image](https://github.com/user-attachments/assets/4f898811-2021-4b72-ae0b-113f9e875bb2)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
