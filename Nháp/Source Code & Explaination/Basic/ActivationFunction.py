
import numpy as np
import matplotlib.pyplot as plt

# apply activation func to scalar, vector, or matrix

def sigmoid(z):
        return 1/(1+np.exp(-z))

def relu(z):
    return z * (z > 0)

def softmax(z):
        return np.exp(z) / float(sum(np.exp(z)))

# #show 
testInput = np.arange(-5,5,0.01)
plt.plot(testInput,sigmoid(testInput),linewidth=2)
plt.plot(testInput,relu(testInput),linewidth=2)
# plt.plot(testInput,softmax(testInput),linewidth=2)
plt.grid(1)
plt.legend(['Sigmoid','ReLU'])
plt.title("Activation Function")
plt.show()

# #try calculator
# print(sigmoid(1))

# print(sigmoid(np.array([-1,0,1])))
# print(relu(np.array([-1,0,1])))
# print(softmax(np.array([-1,0,1])))

# print(sigmoid(np.random.randn(3,4)))