from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.utils import to_categorical
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

wandb.init(project="speechrec")
config = wandb.config

config.max_len = 11
config.buckets = 20

modelPath = r"C:\Coding\Models\audioModels/"

# Save data to array file first
#save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

labels=["bed", "happy", "cat"]

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
config.epochs = 100
config.batch_size = 100

num_classes = 3

X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len, channels)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

'''model = Sequential([
    Conv2D(32,
          (3, 3),
          input_shape=(config.buckets, config.max_len, channels),
          activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32,
          (3, 3),
          activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.3),
    Flatten(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])


model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

model.fit(X_train, y_train_hot, epochs=config.epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])

model.save(modelPath + f"speechModel{config.epochs}.model")'''

model = tf.keras.models.load_model(r"C:\Coding\Models\audioModels\speechModel100.model")
#model.summary()

check = 100

print("\n\nExpected: {expected}\nPredicted: {predicted}".format(
                                                     expected=labels[int(y_train[check])],
                                                     predicted=labels[np.argmax(model.predict(X_train)[check])]))