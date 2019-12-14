import numpy as np
import pickle

path_to_load = './traffic-signs-preprocessed/data6.pickle'
path_to_save = './datasets/80k_no_priorityroad_grey.pickle'

with open(path_to_load, 'rb') as f:
    dataset = pickle.load(f)

# load the dataset into variables
x_train = dataset['x_train']
x_val = dataset['x_validation']
x_test = dataset['x_test']
y_train = dataset['y_train']
y_val = dataset['y_validation']
y_test = dataset['y_test']

# move channels from first position to last position (i.e. from (1, 32, 32) to (32, 32, 1))
x_train = np.moveaxis(x_train, 1, -1)
x_val = np.moveaxis(x_val, 1, -1)
x_test = np.moveaxis(x_test, 1, -1)

# delete the priority road sign from the dataset, as it biased the network
# delete from training data
to_delete = []
for i in range(len(x_train)):
    if y_train[i] == 12:
        to_delete.append(i)
x_train = np.delete(x_train, to_delete, 0)
y_train = np.delete(y_train, to_delete, 0)

# delete from validation data
to_delete = []
for i in range(len(x_val)):
    if y_val[i] == 12:
        to_delete.append(i)
x_val = np.delete(x_val, to_delete, 0)
y_val = np.delete(y_val, to_delete, 0)

# delete from test data
to_delete = []
for i in range(len(x_test)):
    if y_test[i] == 12:
        to_delete.append(i)
x_test = np.delete(x_test, to_delete, 0)
y_test = np.delete(y_test, to_delete, 0)

# write the new data to a file
with open(path_to_save, 'wb') as f:
    pickle.dump([(x_train, y_train), (x_val, y_val), (x_test, y_test)], f)

