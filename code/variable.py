#path variables
#the main path
main_path_ = 'd:/ayushi/Documents/BTech/sem7/fods/assignment2/'
main_path = ''
train_path = main_path + "training_data\\training_data.txt"
test_path = main_path + "test_data\\test_data.txt"
data_path = main_path + 'data\\data.txt'
val_path = main_path + "validation_data\\val_data.txt"

#number of features
features = 3

#number of passes
epochs = 20

#number of models
models = 5

#number of target variables
targets = 1

#initial guess in gradient descent
guess = 20

#the tolerance is the maximum error tolerated
tolerance = 10**(-5)

#the learning rate
learning_rate = 0.001