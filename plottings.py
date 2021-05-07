# Use this module to create different pyplot functions based on what's needed.

import matplotlib.pyplot as plt


# Plots a graph representing a model accuracy during training, validation and test phase
def plot_accuracy(model_name, epochs_range, tr_acc, val_acc, acc):
    plt.figure(figsize=(6, 6))
    plt.plot(epochs_range, tr_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.plot([], [], ' ', label='Test Accuracy: ' + str(acc) + '%')
    plt.legend(loc='lower right')
    plt.title('Accuracy evaluation for ' + model_name)
