# Module for creating a dataset from a folder tree of classes and samples and plotting samples

# The folder structure must be:
# Root
#  |_____ Class1
#           |_______ Sample 1
#           |_______ Sample 2
#           |_______ Sample 3
#  |_____ Class2
#           |_______ Sample 1
#           |_______ Sample 2
#           |_______ Sample 3
#  |_____ Class3
#           |_______ Sample 1
#           |_______ Sample 2
#           |_______ Sample 3

import os
import tensorflow as tf

import random


# Takes a folder containing classes and samples and returns a list of numeric labels ordered with os.walk.
def generate_labels(root_dir, classes_dict):

    labels = list()
    classes_dirs = os.walk(root_dir).__next__()[1]  # Every item in this list is a class folder

    # Assigns a label automatically based on folder name
    for directory in classes_dirs:
        label = 'None'
        for cl in classes_dict:
            if cl in directory:
                label = classes_dict[cl]  # Uses a numerical value for the string label

        directory = os.path.join(root_dir, directory)

        # Inserts the same label in the list for every image contained in the class directory
        for every_file in os.walk(directory).__next__()[2]:
            labels.append(label)

    return labels


# Use this function instead of generate_labels to build totally random sample-label pairs.
# Used to test if the latter is correct, with this one the model should fail terribly.
def generate_fake_labels():
    labels = list()
    for i in range(33589):
        labels.append(random.randint(0, 9))
    return labels


# Returns two TF Datasets of rescaled RGB images. The first is training set (80%) and the second is validation (20%).
# Both datasets are cached and prefetched to boost performance, but not normalized (there's a layer for it).
# The training set directory must follow the structure above.
def generate_training_set(training_set_dir, classes, width, height, batch_size):
    training_set = tf.keras.preprocessing.image_dataset_from_directory(directory=training_set_dir,
                                                                       labels=generate_labels(training_set_dir,
                                                                                              classes),
                                                                       label_mode="int",
                                                                       image_size=(width, height),
                                                                       batch_size=batch_size,
                                                                       subset="training",
                                                                       validation_split=0.2,
                                                                       seed=12345)

    validation_set = tf.keras.preprocessing.image_dataset_from_directory(directory=training_set_dir,
                                                                         labels=generate_labels(training_set_dir,
                                                                                                classes),
                                                                         label_mode="int",
                                                                         image_size=(width, height),
                                                                         batch_size=batch_size,
                                                                         subset="validation",
                                                                         validation_split=0.2,
                                                                         seed=12345)

    training_set = training_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_set = validation_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return training_set, validation_set


# Returns the test set and the respective labels list.
# IMPORTANT: width and height for rescaling MUST match ones used in training.
# The test set directory must follow the structure above.
def generate_test_set(test_set_dir, classes, width, height, batch_size):
    test_labels = generate_labels(test_set_dir, classes)
    test_set = tf.keras.preprocessing.image_dataset_from_directory(directory=test_set_dir,
                                                                   labels=test_labels,
                                                                   label_mode="int",
                                                                   image_size=(width, height),
                                                                   batch_size=batch_size,
                                                                   seed=1234)
    return test_set, test_labels

