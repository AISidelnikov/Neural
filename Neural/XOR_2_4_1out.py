import numpy as np

training_data = open("dataset/Data_train.csv", 'r') 
training_data_list = training_data.readlines()
training_data.close()

class neuron_Net:
    def __init__(self, input_num, neuron_num, output_num, learningrate):
        self.weights = (np.random.rand(neuron_num, input_num) +0.0)
        self.weights_out = (np.random.rand(output_num, neuron_num) +0.0)
        self.lr = learningrate
        pass

    def train(self, inputs_list, targets_list):
        inputs_x = np.array(inputs_list, ndmin=2).T 
        targets_Y = np.array(targets_list, ndmin=2).T 
        x1 = np.dot(self.weights, inputs_x)
        y1 = 1/(1+np.exp(-x1))
        x2 = np.dot(self.weights_out, y1)
        E = -(targets_Y - x2)
        E_hidden = np.dot(self.weights_out.T, E)
        self.weights_out -= self.lr * np.dot((E * x2), np.transpose(y1))
        self.weights -= self.lr * np.dot((E_hidden * y1 * (1.0 - y1)), np.transpose(inputs_x))
        pass

    def query(self, inputs_list):
        inputs_x = np.array(inputs_list, ndmin=2).T
        x1 = np.dot(self.weights, inputs_x)
        y1 = 1/(1 + np.exp(-x1))
        x2 = np.dot(self.weights_out, y1)
        return x2
    
data_input = 2
data_neuron = 4
data_output = 1
learningrate = 0.2
n = neuron_Net(data_input, data_neuron, data_output, learningrate)
epochs = 70000

for e in range(epochs):
    for i in training_data_list:
        all_values = i.split(',') 
        inputs_x = np.asfarray(all_values[1:])
        targets_Y = int(all_values[0])
        n.train(inputs_x, targets_Y)                      

print('Весовые коэффициенты:\n', n.weights)

for i in training_data_list:
    all_values = i.split(',')
    inputs_x = np.asfarray(all_values[1:])
    outputs = n.query(inputs_x)
    print(int(all_values[1]), 'XOR', int(all_values[2]), '=' , float(outputs), '\n')      