import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from imblearn.over_sampling import RandomOverSampler 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPool2D,Flatten,Dense
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import time
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten

X = np.load('siamese_x2.npy')
Y = np.load('siamese_y2.npy')
print('shape of X',X.shape)
print('shape of Y',Y.shape)

label_encoder = LabelEncoder()

# Fit the encoder to the names and transform them into integers
y_encoded = label_encoder.fit_transform(Y)

# Print the encoded labels
print("Encoded Labels:", y_encoded)

# Print the mapping of names to integers (classes)
print("Class Mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# If you need one-hot encoding for CNN
num_classes = len(label_encoder.classes_)  # Number of unique classes
y_onehot = np.eye(num_classes)[y_encoded]


print("The original distribution of data")
unique,counts = np.unique(y_encoded, return_counts=True)
print(np.asarray((unique, counts)).T)

ii = input("Enter any number to continue")

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(160,160,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(num_classes,activation='softmax'))
# from keras.applications.vgg16 import VGG16
# vgg16_weight_path = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# vgg = VGG16(weights='imagenet',include_top=False)
# for layer in vgg.layers:
#     layer.trainable = False
# from tensorflow.keras import Sequential
# from keras import layers
# from tensorflow.keras.layers import Flatten,Dense
# model = Sequential()
# model.add(vgg)
# model.add(Dense(256, activation='relu'))
# model.add(layers.Dropout(rate=0.5))
# #model.add(Dense(64, activation='sigmoid'))
# #model.add(layers.Dropout(rate=0.2))
# model.add(Dense(16, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(Flatten())
# model.add(Dense(31,activation="sigmoid"))
# model.summary()



X_train, X_test, Y_train, Y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='Siamese_CNN_attempt1.h5',  # Filepath to save the model
    monitor='val_accuracy',                     # Monitor validation accuracy (correct metric name)
    mode='max',                                 # Save the model with the highest validation accuracy
    verbose=1,                                  # Print messages when the model is saved
    save_best_only=True                         # Save only the best model
)

# Define the Adam optimizer
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-3)

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer labels
    optimizer=opt,
    metrics=['accuracy']                     # Track accuracy during training
)

# Train the model
history = model.fit(
    x_train,                                 # Training data (features)
    y_train,                                 # Training data (labels)
    validation_data=(x_val, y_val),          # Validation data
    batch_size=64,                          # Batch size
    epochs=50,                              # Number of epochs
    shuffle=True,                            # Shuffle the data during training
    callbacks=[callback]                     # Add the ModelCheckpoint callback
)
