import numpy as np

training_data = open("Data_number_train.csv", "r")
traning_data_list = training_data.readlines()
training_data.close()

test_data = open("Data_number_test.csv", "r")
test_data_list = test_data.readlines()
test_data.close()

weights = np.zeros(15)
lr = 1
epochs = 5
bias = 3

for e in range(epochs):
    for i in traning_data_list:
        all_values = i.split(',')
        inputs_x = np.asfarray(all_values[1:])
        target_Y = int(all_values[0])

        if target_Y == 0:
            target_Y = 1
        else:
            target_Y = 0

        y = np.sum(weights * inputs_x)

        if y > bias:
            y = 1
            E = -(target_Y - y)
            weights -= lr * E * inputs_x
        else:
            y = 0
            E = -(target_Y - y)
            weights -= lr * E * inputs_x

        print('Весовые коэффициенты:\n', weights)

for i in traning_data_list:
    all_values = i.split(',')
    inputs_x = np.asfarray(all_values[1:])
    print(i[0], ' это 0? ', np.sum(weights * inputs_x) >= bias)

t = 0
for i in test_data_list:
    all_values = i.split(',')
    inputs_x = np.asfarray(all_values[1:])
    t += 1
    print('Узнал 0 - ', t, '?', np.sum(weights * inputs_x) >= bias)