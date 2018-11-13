from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('E:/HK7/ML/Pyth/CNN-keras/models/mnistCNN.h5')

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshaping: (batch, height, width, channels)
test = X_test[0].reshape(1,28,28,1)

# Show
# print(X_test.shape)
# plt.imshow(X_test[0])
# plt.show()

# Show2
# Mode "L" :greyscale (8-bit pixels, black and white)
img = Image.open("E:/HK7/ML/Pyth/CNN-keras/test2.png").convert('L')
img = img.resize((28,28))
imgArr = np.array(img)
# print(imgArr.shape)

# Reshaping: (batch, height, width, channels)
imgArr = imgArr.reshape(1,imgArr.shape[0],imgArr.shape[1],1)

# print(imgArr)
# plt.imshow('imgArr')
# plt.show()

# Predicting the Test set results
y_pred = model.predict(test)
print("Predict results: \n", y_pred[0])
print("Predict label: \n", np.argmax(y_pred[0]))
# print(y_pred)

