from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

model = load_model('E:\Research\Semester 7\Machine Learning\Git\Source Code & Explaination\minh\mnistCNN.h5')
# load data
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#test = X_test[0].reshape(1,28,28,1)

# Show
# print(X_test.shape)
# plt.imshow(X_test[0])
# plt.show()
#--------------------------------------------------C2
# Show2
# Mode "L" :greyscale (8-bit pixels, black and white)
img = Image.open("E:/Research/Semester 7/Machine Learning/Git/Source Code & Explaination/minh/test2.png").convert('L')
img = img.resize((28,28))
imgArr = np.array(img)
# print(imgArr.shape)

imgArr = imgArr.reshape(1,imgArr.shape[0],imgArr.shape[1],1)

# print(imgArr)
# plt.imshow('imgArr')
# plt.show()

# Predicting the Test set results
y_pred = model.predict(imgArr)
print("Predict results: \n", y_pred[0])
print("Predict label: \n", np.argmax(y_pred[0]))
# print(y_pred)

