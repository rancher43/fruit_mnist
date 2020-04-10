import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model
import pickle

print('tensorflow version: ' + tf.__version__)

train_images_file = open('train_images', 'rb')
train_images = pickle.load(train_images_file)
train_images_file.close()

train_labels_file = open('train_labels', 'rb')
train_labels = pickle.load(train_labels_file)
train_labels_file.close()

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
# print(train_ds)
del train_images
del train_labels
import gc
gc.collect()

class CNNModel(Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = Conv2D(32, 3, padding='same', activation='relu')
        self.pool1 = MaxPool2D((2, 2))
        self.conv2 = Conv2D(64, 3, padding='same', activation='relu')
        self.pool2 = MaxPool2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.dropout1 = Dropout(0.4)
        self.d2 = Dense(128, activation='relu')
        self.dropout2 = Dropout(0.4)
        self.d3 = Dense(3, activation='softmax')
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout1(x)
        x = self.d2(x)
        x = self.dropout2(x)
        x = self.d3(x)
        return x

model = CNNModel()

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

epochs = 5
for epoch in range(epochs):
    for images, labels in train_ds:
        train_step(images, labels)
    model.save_weights('saved_models', save_format='tf')
    print('Epoch: ' + str(epoch + 1) + ' Loss: ' + str(train_loss.result()), ' Acc: ' + str(train_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()