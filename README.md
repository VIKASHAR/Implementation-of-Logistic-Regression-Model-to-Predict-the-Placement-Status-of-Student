# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vikash A R
RegisterNumber:  212222040179
*/
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(20)
dataset.tail(20)

#droping tha serial no salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('salary',axis=1)
dataset

dataset = dataset.drop('gender',axis=1)
dataset = dataset.drop('ssc_b',axis=1)
dataset = dataset.drop('hsc_b',axis=1)
dataset

dataset.shape
(215, 10)

dataset.info()

#catgorising for further labeling
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset.dtypes

dataset.info()

dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset
dataset.info()
dataset

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)
dataset.head()
x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

clf.predict([[0, 87, 0, 95, 0, 2, 78, 2, 0]])


```

## Output:
![269283140-aff9831a-1c7b-4891-bf2e-17daa68b980f](https://github.com/VIKASHAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405655/3512d1f6-b9d7-46f5-8bc8-7aadb2d77976)

![269278613-aad8cdc4-a792-4849-83fd-cbe34de30b25](https://github.com/VIKASHAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405655/d0dcdb58-c4d9-4484-8cca-456231e56d34)

![269278696-7adf4de4-ae75-4b47-90c5-b28495b39fa8](https://github.com/VIKASHAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405655/ee4957ea-6300-4971-b74e-013d80f29af2)

![269278745-17fc7934-c1cb-43d0-86da-368cea0eea7f](https://github.com/VIKASHAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405655/6a58228b-9ad5-411a-8fc1-a6f0746fe296)

![269278851-40b87c44-cf69-4657-b075-99e0c85fa0eb](https://github.com/VIKASHAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405655/889073e4-391c-41bd-ba66-10b60c16445f)

![269278942-fffbed93-8031-4f67-91ac-9ec19560663f](https://github.com/VIKASHAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405655/fdffd7ae-e49d-4b5b-9b08-e8d5701d5354)

![269279028-cb305b0b-2ec1-4d36-8641-c93fec2f969a](https://github.com/VIKASHAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405655/0f30b2f0-2673-4210-aa7c-4693c37db28a)

![269279197-1f0f866c-d427-4e84-916b-7efe7ea7e2a2](https://github.com/VIKASHAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405655/e215b95e-cca1-476c-8ba1-48163cd6326e)

![269279120-aa254c36-3140-462f-87b4-0bf12e8db9e0](https://github.com/VIKASHAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405655/dffe2379-ceda-4367-b899-186e732b6dd0)


![269279293-1d9052be-3a5a-4ce6-96d4-941314a7bbbe](https://github.com/VIKASHAR/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119405655/30885704-7612-4b84-8570-0248d98422f7)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
