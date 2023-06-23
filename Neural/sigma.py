import numpy as np
import math
import matplotlib.pyplot as plt

from time import time, sleep #Для замера времени выполнения обучения
from tqdm import tqdm #Для вывода прогресса обучения
from tqdm import tqdm_notebook

plt.title("Функция - sin(x)")
plt.xlabel("X")
plt.ylabel("Y = sin(X)")

X_sin = []
Y_sin = []
x = 0 
while x < 1:
    Y_sin += [ (0.48*(math.sin(x*10))+0.48) ]
    X_sin += [x] 
    fx = open ('dataset/Data_sin_x.csv', 'w')
    fy = open ('dataset/Data_sin_y.csv', 'w')
    fx.write (str(X_sin))
    fy.write (str(Y_sin))     
    x += 0.025
    fx.close()
    fy.close()

X_sin2 = np.zeros(len(X_sin))
Y_sin2 = np.zeros(len(Y_sin))
X_sin2 = np.asfarray(X_sin)
Y_sin2 = np.asfarray(Y_sin)

plt.plot(X_sin, Y_sin, color = 'blue', linestyle = 'solid', label = 'sin(x)')
plt.legend(loc=4)
plt.grid(True, linestyle='-', color='0.75')
plt.show()

training_data = open("dataset/Data_sin_x.csv", 'r')
training_data_list = training_data.readlines()
training_data.close()

target_data = open("dataset/Data_sin_y.csv", 'r')
target_data_list = target_data.readlines()
target_data.close()

for i in training_data_list:
        all_values = i.split(',')
        ty = len(all_values)-2
        inputs_ = np.asfarray(all_values[1:ty]) 

for i in target_data_list:
        all_values_t = i.split(',')
        targets_ = np.asfarray(all_values_t[1:ty])

x_data = inputs_
print(len(x_data))
y_data = targets_
print(len(y_data))

plt.title("Проверка данных - sin(x)")
plt.xlabel("X")
plt.ylabel("Y = sin(X)")

plt.plot(x_data, y_data, 'b', label = 'Входные данные - sin(x)')
plt.legend(loc=4)
plt.grid(True, linestyle='-', color='0.75')
plt.show()

class neuron_Net:
    def __init__(self, input_num, neuron_num, output_num, learningrate):
        self.weights = np.random.normal(+0.0, pow(input_num, +0.0), (neuron_num, input_num))
        self.weights_out = np.random.normal(+0.0, pow(neuron_num, +0.0), (output_num, neuron_num))
        self.weights_out_bias = 0.01
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
        y1 = 1/(1+np.exp(-x1))
        x2 = np.dot(self.weights_out, y1)   
        return x2
    
    def querynum(self, inputs_list, numnet): 
        inputs_x = np.array(inputs_list, ndmin=2).T 
        x1 = np.dot(self.weights, inputs_x)
        y1 = 1/(1+np.exp(-x1))
        return y1[numnet]
    
    def querynum2(self, inputs_list, numnet):
        inputs_x = np.array(inputs_list, ndmin=2).T 
        x1 = np.dot(self.weights, inputs_x)
        y1 = 1/(1+np.exp(-x1))
        y12 = np.dot(self.weights_out[0,numnet], y1[numnet])         
        return y12
    
data_input = 2
data_neuron = 6
data_output = 1
learningrate = 0.01
n = neuron_Net(data_input, data_neuron, data_output, learningrate)
epochs = 60000

start = time()
for e in tqdm(range(epochs)):
    for i in range(len(x_data)):
        inputs_x = x_data[i]
        inputs_x = np.append(inputs_x, 1)
        targets_Y = y_data[i]   
        n.train(inputs_x, targets_Y) 
time_out = time() - start
print("Время выполнения: ", time_out, " сек" )

print('Весовые коэффициенты:\n', n.weights)
print('Весовые коэффициенты от скрытого слоя:\n', n.weights_out)

outputs_ = np.array([])
for i in range(len(x_data)):
    inputs_x = x_data[i]
    inputs_x = np.append(inputs_x, 1)
    outputs_ = np.append(outputs_, n.query(inputs_x))

outputs_num0 = np.array([])
outputs_num1 = np.array([])
outputs_num2 = np.array([])
outputs_num3 = np.array([])
outputs_num4 = np.array([])
outputs_num5 = np.array([])
for i in range(len(x_data)):
    inputs_num = x_data[i]
    inputs_num = np.append(inputs_num, 1)
    outputs_num0 = np.append(outputs_num0, n.querynum2(inputs_num, 0))
    outputs_num1 = np.append(outputs_num1, n.querynum2(inputs_num, 1))
    outputs_num2 = np.append(outputs_num2, n.querynum2(inputs_num, 2))
    outputs_num3 = np.append(outputs_num3, n.querynum2(inputs_num, 3))
    outputs_num4 = np.append(outputs_num4, n.querynum2(inputs_num, 4))
    outputs_num5 = np.append(outputs_num5, n.querynum2(inputs_num, 5))

X_sin = []
Y_sin = []
x = 0.0 
while x < 1:
    Y_sin += [ (0.48*(math.sin(x*10))+0.48) ]
    X_sin += [x]
    x += 0.025

X_sin2 = np.zeros(len(X_sin))
Y_sin2 = np.zeros(len(Y_sin))
X_sin2 = np.asfarray(X_sin)
Y_sin2 = np.asfarray(Y_sin)
plt.plot(X_sin, Y_sin, color = 'b', linestyle = 'solid', label = 'Входные данные - sin(x)')
plt.title("Функция - sin(x)")
plt.xlabel("X")
plt.ylabel("Y = sin(X)")
plt.plot(x_data, outputs_, color = 'red', label = 'Обученная сеть - sin(x)')

X_sin = []
Y_sin = []
x = 0.0 
while x < 1:
    Y_sin += [ (0.48*(math.sin(x*10))+0.48) ]
    X_sin += [x]
    x += 0.025

X_sin2 = np.zeros(len(X_sin))
Y_sin2 = np.zeros(len(Y_sin))
X_sin2 = np.asfarray(X_sin)
Y_sin2 = np.asfarray(Y_sin)

plt.plot(X_sin, Y_sin, color = 'b', linestyle = 'solid', label = 'Входные данные - sin(x)')
plt.title("Функция - sin(x)")
plt.xlabel("X")
plt.ylabel("Y = sin(X)")

plt.plot(x_data, outputs_, color = 'red', label = 'Обученная сеть - sin(x)')

plt.plot(x_data, outputs_num0, color = 'b', linestyle = 'solid')
plt.plot(x_data,outputs_num0, label='wout1 * tanh(w1 * x + b1)') 
plt.plot(x_data, outputs_num1, color = 'b', linestyle = 'solid')
plt.plot(x_data,outputs_num1, label='wout2 * tanh(w2 * x + b2)')

plt.plot(x_data, outputs_num2, color = 'b', linestyle = 'solid')
plt.plot(x_data,outputs_num2, label='wout3 * tanh(w3 * x + b3)')

plt.plot(x_data, outputs_num3, color = 'b', linestyle = 'solid')
plt.plot(x_data,outputs_num3, color = 'g',label='wout4 * tanh(w4 * x + b4)')

plt.plot(x_data, outputs_num4, color = 'b', linestyle = 'solid')
plt.plot(x_data,outputs_num4, color = 'c', label='wout5 * tanh(w5 * x + b5)')

plt.plot(x_data, outputs_num5, color = 'b', linestyle = 'solid')
plt.plot(x_data,outputs_num5, color = 'c', label='wout6 * tanh(w6 * x + b6)')
plt.legend(loc=4)
plt.grid(True, linestyle='-', color='0.75')
plt.show()