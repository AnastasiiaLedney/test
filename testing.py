import cv2 as cv 
import numpy as np
IMG_SIZE = 28
import matplotlib.pyplot as plt
import tensorflow as tf 

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')

img = cv.imread('D:/neural/three.png')
plt.imshow(img, cmap=plt.cm.binary)
plt.show()
img.shape
gray =  cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray.shape
resized = cv.resize(gray, (28,28), interpolation = cv.INTER_AREA)
resized.shape
newing = tf.keras.utils.normalize(resized, axis = 1)
newing = np.array(newing).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
newing.shape
predicions = model.predict(newing)
print(" Це цифра :",np.argmax(predicions))



