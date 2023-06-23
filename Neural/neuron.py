import matplotlib.pyplot as plt

# Коэффициент крутизны наклона прямой
A = 0.4
A_vis = A
print('Начальная переменная: ', A, ' * X')

# Скорость обучения
lr = 0.001
# Количество эпох
epochs = 3000

# Массив входных данных x
arr_x = [1,2,3,3.5,4,6,7.5,8.5,9]
# Массив целевых данных y
arr_y = [2.4, 4.5, 5.5, 6.4, 8.5, 11.7, 16.1, 16.5, 18.3]

# Прогон по выборке
for e in range(epochs):
    for i in range(len(arr_x)):
        x = arr_x[i]
        y = A * x
        target_Y = arr_y[i]
        E = target_Y - y
        A += lr*(E/x)

print('Готовая прямая: y = ', A, ' * X')
# Отображение входных данных
def func_data(arr_y):
    return [arr_y[i] for i in range(len(arr_y))]
# Отображение начальной прямой
def func_begin(x_begin):
    return [A_vis * i for i in x_begin]
# Отображение готовой прямой
def func(x):
    return [A * i for i in x]
# Значения по Х входных данных
x_data = arr_x
# Значения по X начальной прямой
x_begin = [i for i in range(0, 11)]
# Значения по X готовой прямой
x = [i for i in range(0, 11)]
# Значения по Y входных данных
y_data = func_data(arr_y)
# Значения по Y начальной прямой
y_begin = func_begin(x_begin)
# Значения по Y готовой прямой
y = func(x)
# Имена графика и числовых координат
plt.title("Neuron")
plt.xlabel("X")
plt.ylabel("Y")
# Имена графиков и числовых координат
plt.plot(x, y, label='Входные данные', color = 'g')
plt.plot(x, y, label='Готовая прямая', color = 'r')
plt.plot(x, y, label='Начальная прямая', color = 'b')
plt.legend(loc=2)
# Точки данных 
plt.scatter(x_data, y_data, color = 'g', s = 10)
# Начальная прямая
plt.plot(x_begin, y_begin, 'b')
# Готовая прямая
plt.plot(x, y, 'r')
# Сетка
plt.grid(True, linestyle='-', color='0.75')

plt.show()

x = input("Введите значнеи ширины X: ")
x = int(x)
T = input("Введите значение высоты Y: ")
T = int(T)
y = A * x

if T > y:
    print('Это жираф')
else:
    print('Это крокодил')