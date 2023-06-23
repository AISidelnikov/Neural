import numpy as np

training_data = open("Data_train.csv", "r")
training_data_list = training_data.readlines()
training_data.close()

test_data = open("Data_test.csv", "r")
test_data_list = test_data.readlines()
test_data.close()

weights = np.zeros(15)
lr = 0.1
epochs = 50000

for e in range(epochs):
    for i in training_data_list:
        all_values = i.split(',')
        inputs_x = np.asfarray(all_values[1:])
        target_Y = int(all_values[0])

        if target_Y == 0:
            target_Y = 1
        else:
            target_Y = 0

        x = np.sum(weights * inputs_x)
        y = 1/(1 + np.exp(-x))
        E = -(target_Y - y)
        weights -= lr * E * y * (1.0 - y) * inputs_x

print('Весовые коэффициенты:\n',weights)

for i in training_data_list:
    all_values = i.split(',')
    inputs_x = np.asfarray(all_values[1:])
    x = np.sum(weights * inputs_x)
    print(i[0], 'Вероятность что 0: ', 1/(1 + np.exp(-x)))

t = 0
for i in test_data_list:
    all_values = i.split(',')
    inputs_x = np.asfarray(all_values[1:])
    t += 1
    x = np.sum(weights * inputs_x)
    print('Вероятность что узнал 0 -', t, '?', 1/(1 + np.exp(-x)))