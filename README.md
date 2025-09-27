# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary.

6.Define a function to predict the Regression value.

Program and Output:

      import pandas as pd 
      import numpy as np 
      import matplotlib.pyplot as plt

      data=pd.read_csv("/content/Placement_Data (1).csv")
      data.head()

<img width="1203" height="200" alt="443582194-2a70578e-a2bb-41b7-987e-d934c0c823e8" src="https://github.com/user-attachments/assets/ca4e19ad-29e1-40e9-bbd0-84300379d026" />

    data = data.drop(['sl_no', 'salary'], axis=1)
    data
        


<img width="1203" height="200" alt="443582194-2a70578e-a2bb-41b7-987e-d934c0c823e8" src="https://github.com/user-attachments/assets/3144bb21-7ff4-4441-ada9-ea2318af31f2" />

    data["gender"]=data["gender"].astype('category') 
    data["ssc_b"]=data["ssc_b"].astype('category') 
    data["hsc_b"]=data["hsc_b"].astype('category') 
    data["degree_t"]=data["degree_t"].astype('category') 
    data["workex"]=data["workex"].astype('category') 
    data["specialisation"]=data["specialisation"].astype('category') 
    data["status"]=data["status"].astype('category') 
    data["hsc_s"]=data["hsc_s"].astype('category') 
    data.dtypes



<img width="281" height="307" alt="443582420-7ce2a409-8b62-4c72-b9d0-2b1b62fd80d1" src="https://github.com/user-attachments/assets/1b42ccc9-5d06-48cf-8e7e-a90da42f80d6" />


    data["gender"]=data["gender"].cat.codes 
    data["ssc_b"]=data["ssc_b"].cat.codes 
    data["hsc_b"]=data["hsc_b"].cat. codes
    data["degree_t"]=data["degree_t"].cat.codes 
    data["workex"]=data["workex"].cat.codes 
    data["specialisation"]=data["specialisation"].cat.codes 
    data["status"]=data["status"].cat.codes 
    data["hsc_s"]=data["hsc_s"].cat.codes 
    data



<img width="956" height="438" alt="443582591-7cb25b08-041e-4917-bd60-212610c97e4e" src="https://github.com/user-attachments/assets/cdd33cad-4b8e-4238-bd58-be42593bf355" />


    x = data.iloc[:, :-1].values 
    y = data.iloc[:, -1].values 
    y

<img width="748" height="225" alt="443582667-0e3fb59c-bab4-4ef2-9f31-37f50805dee7" src="https://github.com/user-attachments/assets/38dd326c-c7e4-40fc-803e-b8cb8c068a90" />

      theta = np.random.randn(x.shape[1]) 
      Y = y
      def sigmoid(z): 
          return 1 / (1 + np.exp(-z))
      def loss(theta, X, y): 
          h = sigmoid(X.dot(theta))
          return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
      def gradient_descent(theta, X, y, alpha, num_iterations): 
          m = len(y)
          for i in range(num_iterations): 
              h = sigmoid(X.dot(theta)) 
              gradient = X.T.dot(h - y) / m 
              theta -= alpha * gradient 
          return theta
      theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)
      def predict(theta, X): 
          h = sigmoid(X.dot(theta)) 
          y_pred=np.where(h>=0.5,1,0) 
          return y_pred
      
      y_pred = predict(theta, x) 
      accuracy = np.mean(y_pred.flatten() == y)
      print("Accuracy: ", accuracy) 
      print(y_pred)


<img width="747" height="159" alt="443582787-bf4560eb-fba5-4cad-ab24-203156005520" src="https://github.com/user-attachments/assets/a625101c-664d-49a0-b1a2-4877af61c737" />

    xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
    y_prednew = predict(theta, xnew) 
    print(y_prednew)

<img width="50" height="35" alt="443582882-dd89e227-3983-4599-bbe0-4d565e7d0c21" src="https://github.com/user-attachments/assets/e26910ee-6d67-4602-bda9-824717de964b" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

