import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from IPython.display import SVG

# load data
# bộ dữ liệu được load vào là mảng 3D 
# với 60000 28x28 grayscale với X_train và 10000 ứng với X_test
# image data with shape (num_samples, 28, 28) -> (60000,28,28)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Input shape that CNN expects is a 4D array (batch, height, width, channels) <batch: số samples>
# X_train có shape(60000,28 ,28) -> X_train.shape[0] = 60000 ...tương tự.
# channel =1 trong trường hợp này Grayscale, =3 RGB
# convert our data type to float32 (giá trị se là số thực)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

#normalize inputs from 0-255 to 0-1
#mỗi pixel trên ả có value từ 0-255 ta sẽ quy tỉ lệ thành 0-1  (ảnh trắng -đen)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# one hot encode
#output cuối cùng cần phải phân lọai ra chữ số trong 10 số từ 0-9 --> multiple classification
# onhot encode converts a class vector (integers) to binary class matrix
number_of_classes = 10
y_train = to_categorical(y_train, number_of_classes)
y_test = to_categorical(y_test, number_of_classes)

# create model

#khai bao sequential is a linear stack of layers
model = Sequential()
# the first hidden layer is a convolutional layer
# have 32filter/output channels, size 5x5  & input size 28x28 grayscale & activation for output là ReLU
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))

#the next layer Pooling layer --> reduce 5x5 -> 2x2  <downscale (vertical, horizontal)>
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#have 32filter/output channels, size 3x3, activation for output là ReLU
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#converts the 2D matrix data to a vector
#Output này là standard input cho fully connected layers
model.add(Flatten())

#Next layer is a fully connected layer with 128 neurons
# output = activation(dot(input, kernel) + bias)
model.add(Dense(128, activation='relu'))

#Next(last) layer is output layer with 10 neurons
model.add(Dense(number_of_classes, activation='softmax'))

# Compile model
# categorical_crossentropy as a loss function that the model will try to minimize
#  optimizer để đảm bảo trọng số của chúng tôi được tối ưu hóa đúng cách
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Fit the model
history= model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# Save the model
model.save('D:\Research\Semester 7\Machine Learning\Git\Source Code & Explaination\minh\mnistCNN.h5')
# Final evaluation of the model
metrics = model.evaluate(X_test, y_test)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)

plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))