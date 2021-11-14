import variable as v 
from typing import TextIO
from sklearn.preprocessing import PolynomialFeatures as pf
import numpy as np

def get_polynomial_features(x: list, degree: int):
    
    # remove the target variable
    x.pop(-1)
    
    # get the polynomial features
    x = np.array(x)
    poly = pf(degree)
    x = poly.fit_transform([x])
    x = x.tolist()[0]

    return x

#returns the features in the first k columns and the targets in last columns
def get_features(line: list)->list:
    
    x = []
    line = line.strip().split(",")
    for l in line:
        x.append(float(l))
    #now x contains the corresponding features
    return x

def get_number_parameters(degree: int, f: int)->int:
    
    #the fake features
    x = np.array([0]*f)
    #specify degree of the polynomial
    poly = pf(degree)
    #returns the list of the parameter coef
    arr = poly.fit_transform([x])
    #return the length
    return len(arr.tolist()[0])

def initialize_matrix(n: int)->list:

    x = []
    
    #append n empty lists to x
    for i in range(n):
        x.append([])

    #fill each list with n 0s
    for i in range(n):
        for j in range(n):
            x[i].append(0)

    return x

def initialize_column(n: int)->list:
    x = []

    for i in range(n):
        x.append(0)
    
    return x

def assign_sum(degree: int, x: list, A: list, B: list):
  
    # remove the target variable and store it seperatly
    t = x[-1]
    # get the polynomial features
    x = get_polynomial_features(x, degree)
    
    #add the sums
    n = len(x)
    for i in range(n):
        for j in range(n):
            A[i][j] =  x[i]*x[j]

    for i in range(n):
        B[i] = x[i]*t
    

def add_sum(degree: int, x: list, A: list, B: list):
 
    # remove the target variable and store it seperatly
    t = x[-1]
    # get the polynomial features
    x = get_polynomial_features(x, degree)
    
    #add the sums
    n = len(x)
    for i in range(n):
        for j in range(n):
            A[i][j] = A[i][j] + x[i]*x[j]

    for i in range(n):
        B[i] = B[i] + x[i]*t
    

def feature_coefficient_matrix(degree: int):

    #get number of parameters
    n  = get_number_parameters(degree, v.features)
    #get an empty nXn matrix
    A = initialize_matrix(n)    
    #and a column matrix
    B = initialize_column(n)

    #open the training file to construct the matrix
    with open(v.train_path, 'r') as input_file:
        for line in input_file:
            x = get_features(line)
            add_sum(degree, x, A, B)

    return [A, B]            


if __name__ == "__main__":
    [A, B] = feature_coefficient_matrix(5)

    for i in A:
        print(i)
    print("****")
    print(B)