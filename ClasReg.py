import os
import cv2
import numpy as np
import random
from collections import defaultdict
from glob import glob
import matplotlib.pyplot as plt
import h5py
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models, losses, backend
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


########################################################
###############LOADING HYPERPARAMETERS########################
########################################################

# Load hyperparameters from a text file
file = open("HyperparamMulticlass.txt")
lines = file.readlines()

# Batch size
BATCH_SIZE = int(lines[0].split(";")[1])
print("Batch Size: " + str(BATCH_SIZE))

# Number of classes
NUM_CLASSES = int(lines[7].split(";")[1])
print("Number of classes: " + str(NUM_CLASSES))

# Images folder
DATA_DIR_IMAGES = str(lines[8].split(";")[1].split("\n")[0])
# Correct writing issues
DATA_DIR_IMAGES = DATA_DIR_IMAGES.replace("\\", "/")
print("Training Images Directory: " + str(DATA_DIR_IMAGES))

# Masks folder
DATA_DIR_MASKS = str(lines[9].split(";")[1].split("\n")[0])
DATA_DIR_MASKS = DATA_DIR_MASKS.replace("\\", "/")
print("Training Masks Directory: " + str(DATA_DIR_MASKS))

# Debug flag
DEBUG = str(lines[10].split(";")[1].split("\n")[0]) != "0"

# Model name
MODEL = str(lines[11].split(";")[1].split("\n")[0])

# Dropout
DROPOUT = 0.0
DROPOUT = float(lines[12].split(";")[1])
print("Dropout: " + str(DROPOUT))


# Learning Rate
LEARNING_RATE = float(lines[3].split(";")[1])
print("Learning rate: " + str(LEARNING_RATE))

# Epochs
EPOCHS = int(lines[4].split(";")[1])
print("Number of epochs: " + str(EPOCHS))



MODELO = str(lines[11].split(";")[1].split("\n")[0])

# Image size
IMAGE_SIZE = int(lines[5].split(";")[1])
print("Image size: " + str(IMAGE_SIZE))

# Early Stopping patience
EARLY_STOPPING = int(lines[6].split(";")[1])
print("Early Stopping patience: " + str(EARLY_STOPPING))

NUM_FILTERS = int(lines[17].split(";")[1])
print("Number of filters: " + str(NUM_FILTERS))

alpha = float(lines[19].split(";")[1])
print("Alpha: " + str(alpha))

file.close()
input("Press ENTER to continue")



########################################################
###############IMAGE LOADING FUNCTIONS#################
########################################################
########################################################


# Define a function to load data and create a tuple with the image, mask, and value
def load_data(image_path, class_value, value):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    
    class_value = tf.cast(class_value, tf.int32)
    class_number = tf.one_hot(class_value, NUM_CLASSES)
    
    value = tf.cast(value, tf.float32)
    
    class_with_value = tf.concat([class_number, tf.expand_dims(value, axis=-1)], axis=-1)

    return image, class_with_value



########################################################
###############LOSS FUNCTIONS###########################
########################################################
########################################################

#Segmentation Loss Function
def dice_loss(y_true, y_pred, smooth=1e-5):
    mask_true = tf.slice(y_true, [0, 0, 0, 0], [-1, -1, -1, 1])
    mask_pred = y_pred
    
    if DEBUG:
        tf.print("\nmask_true[40,40, :]", mask_true[0, 40, 40, :])
        tf.print("\nmask_true[20, 20, :]", mask_true[0, 20, 20, :])
        tf.print("mask_pred[20, 20, :]", tf.argmax(mask_pred[0, 20, 20, :]))
        
        
        tf.print("mask_true[40, 40, :]", mask_true[0, 40, 40, :])
        tf.print("mask_pred[40, 40, :]", tf.argmax(mask_pred[0, 40, 40, :]))

        tf.print("mask_true[60, 60, :]", mask_true[0, 60, 60, :])
        tf.print("mask_pred[60, 60, :]", tf.argmax(mask_pred[0, 60, 60, :]))

        tf.print("mask_true[79, 79, :]", mask_true[0, 79, 79, :])
        tf.print("mask_pred[79, 79, :]", tf.argmax(mask_pred[0, 79, 79, :]))
    
    mask_true_f = K.flatten(K.one_hot(K.cast(mask_true, 'int32'), num_classes=NUM_CLASSES)[..., 1:NUM_CLASSES])
    mask_pred_f = K.flatten(mask_pred[..., 1:NUM_CLASSES])
    intersect = K.sum(mask_true_f * mask_pred_f, axis=-1)
    denom = K.sum(mask_true_f + mask_pred_f, axis=-1)
    dice_loss = 1 - K.mean((2. * intersect / (denom + smooth)))
     
    return dice_loss

#Categorical_cross_entropy
def categorical_cross_entropy_loss(y_true, y_pred):
    # Elimina la información adicional al final de y_true
    value_true = y_true[:, :-1]

    # Calcula la pérdida de entropía cruzada categórica entre y_true y y_pred
    loss = -tf.reduce_sum(value_true * tf.math.log(y_pred + 1e-10), axis=-1)
    
    #tf.print(value_true)
    #tf.print(y_pred)
    
    # Devuelve la pérdida promedio sobre el lote
    return tf.reduce_mean(loss)

#Regression Loss Function
def mean_squared_error(y_true, y_pred):
    #Firstly, we split the regression part of the y_true tensor
    value_true = y_true[:, -1]

    value_pred = y_pred
    #tf.print(value_true)
    #tf.print(value_pred)
    #Same format
    value_true = tf.cast(value_true, dtype=tf.float64)
    value_pred = tf.cast(value_pred, dtype=tf.float64)
         
    mse_loss = tf.keras.losses.mean_squared_error(value_true, value_pred[:, 0])
    
    if DEBUG:
        tf.print("\nvalue_true", value_true[:])
        tf.print("value_pred", value_pred[:])
        tf.print("\nmae_loss", mae_loss)
        
    return mse_loss






########################################################
###############CONSTRUCCIÓN DE DEEPLAB##################
########################################################
########################################################

def ConvolutionalNetwork(image_size):
    model_input = layers.Input(shape=(image_size, image_size, 3))
    
    # Arquitectura más ligera como encoder
    x = layers.Conv2D(NUM_FILTERS, 3, strides=2, activation='relu', padding='same')(model_input)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Conv2D(NUM_FILTERS*2, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Conv2D(NUM_FILTERS*4, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Conv2D(NUM_FILTERS*8, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x) 
    x = layers.Dropout(DROPOUT)(x)
    
     # Agregar capas Dense para procesar las características extraídas
    regression_output = layers.GlobalAveragePooling2D()(x)
    regression_output = layers.Dense(64, activation='relu')(regression_output)
    regression_output = layers.Dense(32, activation='relu')(regression_output)
    regression_output = layers.Dense(1, name='regression_output')(regression_output)  # Salida con una sola dimensión para la estimación de cantidades
    
    
    # Agregar capas Dense para procesar las características extraídas
    classification_output = layers.GlobalAveragePooling2D()(x)
    classification_output = layers.Dense(NUM_FILTERS*2, activation='relu')(classification_output)
    classification_output = layers.Dense(NUM_FILTERS, activation='relu')(classification_output)
    classification_output = layers.Dense(NUM_CLASSES, activation='softmax', name='classification_output')(classification_output)  # Salida con una dimensión por clase para la clasificación

    model = models.Model(inputs=model_input, outputs=[classification_output, regression_output])
    
    return model








# Define el Callback personalizado para guardar los pesos
class SaveWeightsCallback(tf.keras.callbacks.Callback):
    def on_test_begin(self, epoch, logs=None):
        # Este método se ejecutará al final de cada época de entrenamiento
        # Aquí guardamos los pesos del modelo en la lista
        tf.print("\n\n\n\n\n\n\n\n\tEMPIEZA VALIDACION\n\n\n\n\n\n")
        for layer in self.model.layers:
            for weight in layer.get_weights():
                tf.print("Weight:", weight)
    def on_test_end(self, epoch, logs=None):
        # Este método se ejecutará al final de cada época de entrenamiento
        # Aquí guardamos los pesos del modelo en la lista
        tf.print("\n\n\n\n\n\n\n\n\tTERMINA VALIDACIÓN\n\n\n\n\n\n")
        for layer in self.model.layers:
            for weight in layer.get_weights():
                tf.print("Weight:", weight)
        input()



########################################################
###############EXPERIMENT##################
########################################################
########################################################
hist_loss_segmentation = []
hist_loss_regression = []
hist_val_loss_segmentation = []
hist_val_loss_regression = []
hist_test_loss_segmentation = []
hist_test_loss_regression = []
               
k = 5
for i in range(42, 42+k):
    ########################################################
    ###############TRAIN, TEST AND VAL DATASETS#############
    ########################################################
    ########################################################
    # Instantiate a seed
    random.seed(i)

    # Input file
    input_file = "./Pancake/Pancake_tags_formated.txt"

    # Dictionary to store images by class
    image_dict = defaultdict(list)

    # Read the file and store images in the dictionary by class
    with open(input_file, "r") as file:
        for line in file:
            parts = line.strip().split(";")
            image_name = DATA_DIR_IMAGES + "/" + parts[0] + ".jpg"
            class_label = float(parts[1])
            class_number = class_label
            final_value = float(parts[3])
            image_dict[class_label].append((image_name, class_number, final_value))

    # Lists to store training, validation, and test images and values
    train_images = []
    val_images = []
    test_images = []
    value_train = []
    value_val = []
    value_test = []
    train_class = []
    val_class = []
    test_class = []

    # Separate images into training, validation, and test sets
    for class_label, class_images in image_dict.items():
        # Shuffle images randomly for each class
        random.shuffle(class_images)
        # Extract all images except the last two for training under random order
        train_images.extend([image[0] for image in class_images[:-2]])
        train_class.extend([image[1] for image in class_images[:-2]])
        value_train.extend([image[2] for image in class_images[:-2]])
        # Extract the penultimate image for validation
        val_images.append(class_images[-2][0])
        val_class.append(class_images[-2][1])
        value_val.append(class_images[-2][2])
        # Extract the last image for testing
        test_images.append(class_images[-1][0])
        test_class.append(class_images[-1][1])
        value_test.append(class_images[-1][2])

    # Print the results
    print("Training Images:")
    print(train_images)
    print("Training Class:")
    print(train_class)
    print("Training Real Values:")
    print(value_train)
    print("Validation Images:")
    print(val_images)
    print("Validation Class:")
    print(val_class)
    print("Validation Real Values:")
    print(value_val)
    print("Test Images:")
    print(test_images)
    print("Test Class:")
    print(test_class)
    print("Test Real Values:")
    print(value_test)


    # Generating train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_class, value_train))
    train_dataset = train_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    #train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

    # Generating val dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_class, value_val))
    val_dataset = val_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

    # Generating test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_class, value_test))
    test_dataset = test_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)
    print("Test Dataset:", test_dataset)

    ########################################################
    ###############TRAINING AND EVALUATION#############
    ########################################################
    ########################################################

    model = ConvolutionalNetwork(image_size=IMAGE_SIZE)

    optimizer = Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer, loss=[categorical_cross_entropy_loss, mean_squared_error], loss_weights = [alpha, 1-alpha])
    
    model.summary()
    
    history = model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=val_dataset)


    # Obtaining segmentation and regression validation losses
    loss_segmentation = history.history['classification_output_loss']
    loss_regression = history.history['regression_output_loss']
    val_loss_segmentation = history.history['val_classification_output_loss']
    val_loss_regression = history.history['val_regression_output_loss']

    # Saving validation historics
    hist_loss_segmentation.append(loss_segmentation)
    hist_loss_regression.append(loss_regression)
    hist_val_loss_segmentation.append(val_loss_segmentation)
    hist_val_loss_regression.append(val_loss_regression)

    # Getting evaluation losses
    test_loss = model.evaluate(test_dataset)
    test_loss_segmentation = test_loss[1]
    test_loss_regression = test_loss[2]
    
    print("Test:" + str(test_loss[1]) + " "  + str(test_loss[2]))

    # Saving evaluation historics
    hist_test_loss_segmentation.append(test_loss_segmentation)
    hist_test_loss_regression.append(test_loss_regression)



########################################################
###############SAVING MODEL##################
########################################################
########################################################

#Model saving.
model.save(MODELO)
print(MODELO + " saved")


# Cross-validation alike losses
mean_loss_segmentation = np.mean(hist_loss_segmentation, axis=0)
mean_loss_regression = np.mean(hist_loss_regression, axis=0)
mean_val_loss_segmentation = np.mean(hist_val_loss_segmentation, axis=0)
mean_val_loss_regression = np.mean(hist_val_loss_regression, axis=0)
mean_test_loss_segmentation = np.mean(hist_test_loss_segmentation, axis=0)
mean_test_loss_regression = np.mean(hist_test_loss_regression, axis=0)

#Saving extra info in the model.
with h5py.File(MODELO, 'r+') as file:
    history_group = file.create_group('history')
    history_group.create_dataset('loss_classification', data=mean_loss_segmentation)
    history_group.create_dataset('loss_regression', data=mean_loss_regression)
    history_group.create_dataset('val_loss_classification', data=mean_val_loss_segmentation)
    history_group.create_dataset('val_loss_regression', data=mean_val_loss_regression)
    history_group.create_dataset('test_loss_classification', data=mean_test_loss_segmentation)
    history_group.create_dataset('test_loss_regression', data=mean_test_loss_regression)
