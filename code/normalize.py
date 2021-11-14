import variable as v
import random 
from typing import TextIO

#the total number of seperate data for each datapoint
n = v.features + v.targets

def normalize(x: list, max_val: list):
    for data_point in x:
        for i in range(n):
            data_point[i] = str(float(data_point[i])/max_val[i])    

def find_max(max_val: list, line: list):
    for i in range(n):
        max_val[i] = max(max_val[i], float(line[i]))

def initialize(max_val: list, n: int):
    for i in range(n):
        max_val.append(0)

def write_organized_data(output_file: TextIO, x: list):
    for data_point in x:
        str = ""
        for i in range(n):
            
            str = str + data_point[i]
            if(i<n-1):
                str = str + ","
            
        str = str + "\n"
        output_file.write(str)

def create_list(input_file: TextIO):
    #list of lists to store the data
    x = []
    
    #get the maximum value of each feature which can be used to normalize the data
    max_val = []
    initialize(max_val, n)

    #first line is the header, ignore it
    input_file.readline()

    for line in input_file:
        #make a list of each element seperated by a comma and append it to the list
        line = line.strip().split(",")
        x.append(line)
        #find the maximum value
        find_max(max_val, line)  
    
    #now shuffle the list
    random.shuffle(x)

    #normalize
    normalize(x, max_val)
    print(x)
    #input the shuffled, normalized data into another file
    with open(v.data_path, 'w') as data:
        write_organized_data(data, x)


if __name__ == "__main__":
    with open(v.main_path + 'insurance.txt', 'r') as input_file:
        create_list(input_file)
