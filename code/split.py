import random
import variable as v
from typing import TextIO

def split_70_20_10(path):

    with open(path, 'r') as input_file, open(v.train_path, 'w') as training, open(v.test_path, 'w') as test, open(v.val_path, 'w') as val:
        for line in input_file:  
            r = random.random()
            if(r < 0.7):
                training.write(line)        
            elif(0.7 <= r and r < 0.9):
                val.write(line)
            else:
                test.write(line)

def shuffle(path):
    
    with open(path, 'r') as input_file:
        x = []
        for line in input_file:    
            x.append(line)
        random.shuffle(x)
    
    with open(path, 'w') as output_file:
        for line in x:
            output_file.write(line)

def split_data():

    shuffle(v.data_path)
    split_70_20_10(v.data_path)

if __name__ == "__main__":

    split_data()