import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

import general_settings as s
import load_data
import plottings

if __name__ == '__main__':

    # -- EDITABLE -- Change this to any model name you want to print on the plotted graph
    displayed_name = 'Base Model'

    # The Base model:
    # Normalization layer to rescale the input
    # -- EDITABLE -- 3 Convolutional + MaxPooling layers with ReLU: 16-32-64 filters with 3x3 kernel
    # 1 Flattening layer to unstack the compressed image and serve it to the FC
    # -- EDITABLE -- 1 Dense layer activated by ReLU with 128 neurons
    # Output layer with 10 neurons (model_type of classes to predict)
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(s.IMG_WIDTH, s.IMG_HEIGHT, 3)),
        layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(len(s.CLASSES))
    ])

    # ---------------- DO NOT EDIT FROM HERE ON ---------------- #

    # Build the model with categorical cross-entropy loss (needed for integer labels)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics='accuracy')

    # Load both training set and validation set with an external function
    ts, vs = load_data.generate_training_set(s.TRAINING_DIR, s.CLASSES, s.IMG_WIDTH, s.IMG_HEIGHT, s.BATCH_SIZE)

    # Train the model and store the results in a variable
    history = model.fit(ts, validation_data=vs, epochs=s.EPOCHS)

    # Load both test set and test labels. The latter are not needed
    test_set, test_labels = load_data.generate_test_set(s.TEST_DIR,
                                                        s.CLASSES,
                                                        s.IMG_WIDTH,
                                                        s.IMG_HEIGHT,
                                                        s.BATCH_SIZE)

    # Evaluate the test set and store the results in a variable
    test_results = model.evaluate(test_set, verbose=1)
    accuracy_percentage = round(test_results[1], 4) * 100

    # Plot training, validation and test accuracy
    plottings.plot_accuracy(model_name=displayed_name,
                            epochs_range=range(s.EPOCHS),
                            tr_acc=history.history['accuracy'],
                            val_acc=history.history['val_accuracy'],
                            acc=accuracy_percentage)

    plt.show()

