from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('E:/HK7/ML/Pyth/CNN-keras/models/mnistCNN.h5')

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Show
# plt.imshow(X_test[0])
# plt.show()

# print(X_test.shape)

# # Reshaping: (batch, height, width, channels)
test = X_test[0].reshape(1,28,28,1)

# Predicting the Test set results
y_pred = model.predict(test)
print(y_pred)

