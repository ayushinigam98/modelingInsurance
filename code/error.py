import model
import variable as v
import matrix
from sklearn.preprocessing import PolynomialFeatures as pf
import numpy as np
import matplotlib.pyplot as plt

def plot(x: list, err: list, strX, strY):
    plt.plot(x, err)
    plt.xlabel(strX) 
    plt.ylabel(strY)
    plt.show()  

def calculate_error(w: list, degree: int, path)->float:

    error = 0
    n = 0

    with open(path, 'r') as input_file:
        for line in input_file:
            #get the features
            x = matrix.get_features(line)                   
            #store target value
            t = x[-1]                  
            #get polynomial features                     
            x = matrix.get_polynomial_features(x, degree)   
            #the predicted value
            y = model.predicted_value(w, x)                 
            #calculate error
            error = error + (y - t)**2                      
            #add 1 to number of test data points 
            n = n + 1                                          
            

    error = error/2
    rmse = (2*error/n)**0.5

    return rmse
