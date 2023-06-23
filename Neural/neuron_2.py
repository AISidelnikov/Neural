import matplotlib.pyplot as plt
import random
# Коэффициент крутизны наклона прямой
w1 = 0.4
w1_vis = w1
w2 = random.uniform(-4, 4)
w2_vis = w2
print('Начальная переменная: ', w1, ' * X + ', w2)

# Скорость обучения
lr = 0.001
# Количество эпох
epochs = 3000

# Массив входных данных x
arr_x1 = [1,2,3,3.5,4,6,7.5,8.5,9]
# Значение входных данных второго входа
x2 = 1
# Массив целевых данных y
arr_y = [2.4, 4.5, 5.5, 6.4, 8.5, 11.7, 16.1, 16.5, 18.3]

# Прогон по выборке
for e in range(epochs):
    for i in range(len(arr_x1)):
        # Получить x координату точки
        x1 = arr_x1[i]
         # Получить расчетную y, координату точки
        y = w1 * x1 + w2
        # Получить целевую Y, координату точки
        target_Y = arr_y[i]
        # Ошибка E = -(целевое значение - выход нейрона)
        E = -(target_Y - y)
        # Меняем вес при x, в соответствии с правилом обновления веса
        w1 -= lr * E * x1
        # Меняем вес при x2 = 1 
        w2 -= lr * E

print('Готовая прямая: ', w1, '* X + ', w2)
# Отображение входных данных
def func_data(arr_y):
    return [arr_y[i] for i in range(len(arr_y))]
# Отображение начальной прямой
def func_begin(x_begin):
    return [w1_vis * i + w2_vis for i in x_begin]
# Отображение готовой прямой
def func(x):
    return [w1 * i + w2 for i in x]
# Значения по Х входных данных
x_data = arr_x1
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