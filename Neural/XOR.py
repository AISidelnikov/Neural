import numpy as np

training_data = open("XOR/Data_train.csv", 'r')
training_data_list = training_data.readlines()
training_data.close()


test_data = open("XOR/Data_test.csv", 'r')
test_data_list = test_data.readlines()
test_data.close() 

class neuron_Net:
    def __init__(self, input_num, neuron_num, learningrate):
        self.weights = (np.random.rand(neuron_num, input_num) - 0.5)
        self.lr = learningrate
        pass

    def train(self, inputs_list, targets_list):
        inputs_x = np.array(inputs_list, ndmin=2).T
        targets_Y = np.array(targets_list, ndmin=2).T
        x = np.dot(self.weights, inputs_x)
        y = 1/(1 + np.exp(-x))
        E= -(targets_Y - y)
        self.weights -= self.lr * np.dot(E * y * (1.0 - y), np.transpose(inputs_x))
        pass
    
    def query(self, inputs_list):
        inputs_x = np.array(inputs_list, ndmin=2).T
        x = np.dot(self.weights, inputs_x)
        y = 1/(1 + np.exp(-x))
        return y
    
data_input = 2
data_neuron = 2
learningrate = 0.1
n = neuron_Net(data_input, data_neuron, learningrate)
epochs = 10000

for e in range(epochs):
    for i in training_data_list:
        all_values = i.split(',')
        inputs_x = np.asfarray(all_values[1:])
        inputs_Y = int(all_values[0])
        target_Y = np.zeros(data_neuron) + 0.01
        target_Y[int(all_values[0])] = 0.99
        n.train(inputs_x, target_Y)

print('Весовые коэффициенты:\n', n.weights)

for i in training_data_list:
    all_values = i.split(',')
    inputs_x = np.asfarray(all_values[1:])
    outputs = n.query(inputs_x)