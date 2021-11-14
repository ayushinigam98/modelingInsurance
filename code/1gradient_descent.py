import split
import variable as v
import matrix
import numpy as np
import error

def lasso(l: float, w: np.ndarray, n: int):
    k = []
    for i in range(n):
        if(w[i]>0):
            k.append(l)
        else:
            k.append(-l)
    k = np.array(k)
    return k

def ridge(l: float, w: np.ndarray, n: int):
    return l*w

def gradient_descent(degree: int, func)->list:
    """
    |w0(k+1)|        |w0(k)|         |f0f0 f0f1 f0f2 ...| |w0(k)|     |t  |          |w0(k)|
    |w1(k+1)|        |w1(k)|         |f1f0 f1f1 f1f2 ...| |w1(k)|     |tf1|          |w1(k)|
    |w2(k+1)|        |w2(k)|         |f2f0 f2f1 ...     | |w2(k)|     |tf2|          |w2(k)|  
    |w3(k+1)|    =   |w3(k)|  - eta( |. . . . . .       | |w3(k)| -   |tf3| + lambda*|w3(k)| )
    | . . . |        |. . .|         |. . . . . .       | |. . .|     |...|          | ... |   
    | . . . |        |. . .|         |. . . . . .       | |. . .|     |...|          | ... |
    | . . . |        |. . .|         |. . . . . .       | |. . .|     |...|          | ... |   
                                             ^A                         ^B 
                                     |______________________________________________________|
                                                                ^dE/dw      
    """
    #get the A matrix and B matrix
    [A, B] = matrix.feature_coefficient_matrix(degree) 
    
    #hyperparameters
    eta = 0.0001
    l = 1
    n = len(B) 
    
    #the guess value of w
    w = [v.guess]*n
    w = np.array(w)
    #the error to tell when to stop iteration
    err = v.tolerance + 1

    while(err > v.tolerance):
        #calculate the partial derivative according to the equation above
        dEdw = np.add(np.subtract(np.dot(A, w), B), func(l, w, n))
        #update the parameters
        w = np.subtract(w, eta*dEdw)  
        #calculate the error
        err = eta*(np.linalg.norm(dEdw))**2
        print(err,degree)
    
    return w

if __name__ == "__main__":
    
    err_degree_test = []
    err_degree_train = []
    
    for degree in range(1, 10):
        w = gradient_descent(degree, ridge)
        
        #error calculation
        err_test = error.calculate_error(w, degree, v.test_path)
        err_train = error.calculate_error(w, degree, v.train_path)
        err_degree_test.append(err_test)
        err_degree_train.append(err_train)
  
    #plot the error
    error.plot(range(1, 10), err_degree_test, "degree", "Root Mean square error")
    error.plot(range(1, 10), err_degree_train, "degree", "Root Mean square error")
    #print the error
    print(err_degree_test)
    print(err_degree_train)

    
