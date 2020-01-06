from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

#Hiep 09/16/2019

def derivate(x):
    return 2*(x + 5)
    # return  3* (x**2) - (6 * (x))

def cost(x):
    return (x + 5) ** 2
    # return (x ** 3)-(3 *(x ** 2))+7

def gradient_descent():
    alpha = 0.05
    theta = 0.001
    x_old = 0
    x_new = -1

    x_list, y_list = [x_new], [cost(x_new)]
    while (abs(x_new - x_old) > theta):
        # print ("x_new = ", x_new, ". derivate(x_new)", derivate(x_new))
        # print ("abs(cost(x_new) - gradient(x_new) = ", abs(cost(x_new) - derivate(x_new)))
        x_old = x_new
        x_new = x_old - alpha * derivate(x_old)

        #add to the list
        x_list.append(x_new)
        y_list.append(cost(x_new))
    return (x_new, cost(x_new), x_list, y_list)

def main():
    print("Hello")
    # Loading housing dataset
    # (train_data, train_y), (test_data, test_y) = boston_housing.load_data()
    # print("Training data ", train_data.shape)
    # print("Training data sample", train_data[0])
    # print("Training target: ", train_y.shape)
    # print("Training target sample", train_y[0])
    # print("Test data %d %d" %test_data.shape)

    (a,b, x_list, y_list) = gradient_descent()
    print("a = ", a)
    print("b = ", b)

    x = np.linspace(-10,10,500)

    # plt.subplot(1,2,2)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    plt.plot(x,cost(x), c="r")
    plt.title("Gradient descent")
    plt.show()

    plt.subplot(1,2,1)
    plt.scatter(x_list,y_list,c="g")
    plt.plot(x_list,y_list,c="g")
    # plt.plot(x,cost(x), c="r")
    plt.xlim([1.0,2.1])
    plt.title("Zoomed in Gradient descent to Key Area")
    plt.show()

if __name__ == "__main__":
    main()
