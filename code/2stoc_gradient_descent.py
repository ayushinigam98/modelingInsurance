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
    n = matrix.get_number_parameters(degree, v.features)
    #get an empty nXn matrix
    A = matrix.initialize_matrix(n)    
    #and a column matrix
    B = matrix.initialize_column(n)

    #hyperparameters
    eta = 0.1
    l = 0
    
    #the guess value of w
    w = [v.guess]*n
    w = np.array(w)

    for i in range(v.epochs):
        with open(v.train_path, 'r') as input_file:
            for line in input_file:
                #get the feature matrix
                x = matrix.get_features(line)
                matrix.assign_sum(degree, x, A, B)
                #now get dE/dw
                dEdw = np.add(np.subtract(np.dot(A, w), B), func(l, w, n))
                #update the parameters
                w = np.subtract(w, eta*dEdw) 
    
    return w

if __name__ == "__main__":
    err_degree = []
    
    for degree in range(1, 7):
        
        w = gradient_descent(degree, ridge)
        err = error.calculate_error(w, degree, v.val_path)
        err_degree.append(err)
        print(err)
    
    error.plot(range(1, 7), err_degree, "Root Mean square error", "degree")
    
    
