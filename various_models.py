from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from general_settings import IMG_WIDTH, IMG_HEIGHT, CLASSES


# Without parameters, it loads the Base Model CNN.
# Else, you can adjust the model_type of filters, kernel size and activation functions.
def base_model(units_multiplier=1, kernel_size=3, activation='relu'):

    if units_multiplier == 1 and kernel_size == 3 and activation == 'relu':
        name = 'CNN_Base_Model'
    else:
        if not units_multiplier == 1:
            name = 'Base_Model_with_' + str(units_multiplier) + 'x_Filters'
        elif not kernel_size == 3:
            name = 'Base_Model_with_' + str(kernel_size) + 'x' + str(kernel_size) + '_Kernels'
        elif not activation == 'relu':
            name = 'Base_Model_with_' + str.capitalize(activation)
        else:
            name = 'Bugged_CNN_Should_Not_Be_Here'

    return Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            layers.Conv2D(filters=16 * units_multiplier,
                          kernel_size=kernel_size,
                          padding='same',
                          activation=activation),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=32 * units_multiplier,
                          kernel_size=kernel_size,
                          padding='same',
                          activation=activation),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=64 * units_multiplier,
                          kernel_size=kernel_size,
                          padding='same',
                          activation=activation),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(units=128*units_multiplier, activation=activation),
            layers.Dense(len(CLASSES))],
            name=name
    )


# Loads different configurations born from the Base Model.
# The parameter is just a switch and is valid in a range [0,6].
def predefined_base_model(num_layers):

    if num_layers == 0:  # Model without Max Pooling
        return Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(units=128, activation='relu'),
            layers.Dense(len(CLASSES))],
            name='CNN_no_Pooling'
        )
    elif num_layers == 1:  # Single convolutional layer
        return Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(units=128, activation='relu'),
            layers.Dense(len(CLASSES))],
            name='CNN_Single_Conv'
        )
    elif num_layers == 2:  # Two convolutional layers
        return Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(units=128, activation='relu'),
            layers.Dense(len(CLASSES))],
            name='CNN_double_Conv'
        )
    elif num_layers == 3:  # Three convolutional layers, but no Dense at the end
        return Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(len(CLASSES))],
            name='CNN_without_FC_layer'
        )
    elif num_layers == 4:  # Four convolutional layers
        return Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(units=128, activation='relu'),
            layers.Dense(len(CLASSES))],
            name='CNN_4_Conv_Layers'
        )
    elif num_layers == 5:  # Five convolutional layers (max model_type of valid pooling)
        return Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            layers.Conv2D(filters=4, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(units=128, activation='relu'),
            layers.Dense(len(CLASSES))],
            name='CNN_5_Conv_Layers'
        )
    elif num_layers == 6:  # Three subsequent convolutional layers
        return Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(units=128, activation='relu'),
            layers.Dense(len(CLASSES))],
            name='CNN_subsequent_Convs'
        )
    else:  # Input error
        print('Unreachable point, need to bugfix! Quitting...')
        quit()


# Loads different configurations of an MLP.
# By default, it loads a vanilla MLP with 1 hidden layer of 128 neurons.
# num_layers is valid in a range [2-4].
# Use direction='inc' or direction='dec' to set an increasing or decreasing model_type of units in the layers.
def mlp_model(num_layers, direction='none'):

    if direction == 'inc':
        if num_layers == 2:
            return Sequential([
                layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                layers.Flatten(),
                layers.Dense(units=64, activation='relu'),
                layers.Dense(units=128, activation='relu'),
                layers.Dense(len(CLASSES))],
                name='MLP_64-128'
            )
        elif num_layers == 3:
            return Sequential([
                layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                layers.Flatten(),
                layers.Dense(units=32, activation='relu'),
                layers.Dense(units=64, activation='relu'),
                layers.Dense(units=128, activation='relu'),
                layers.Dense(len(CLASSES))],
                name='MLP_32-64-128'
            )
        elif num_layers == 4:
            return Sequential([
                layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                layers.Flatten(),
                layers.Dense(units=16, activation='relu'),
                layers.Dense(units=32, activation='relu'),
                layers.Dense(units=64, activation='relu'),
                layers.Dense(units=128, activation='relu'),
                layers.Dense(len(CLASSES))],
                name='MLP_16-32-64-128'
            )
    elif direction == 'dec':
        if num_layers == 2:
            return Sequential([
                layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                layers.Flatten(),
                layers.Dense(units=128, activation='relu'),
                layers.Dense(units=64, activation='relu'),
                layers.Dense(len(CLASSES))],
                name='MLP_128-64'
            )
        elif num_layers == 3:
            return Sequential([
                layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                layers.Flatten(),
                layers.Dense(units=128, activation='relu'),
                layers.Dense(units=64, activation='relu'),
                layers.Dense(units=32, activation='relu'),
                layers.Dense(len(CLASSES))],
                name='MLP_128-64-32'
            )
        elif num_layers == 4:
            return Sequential([
                layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                layers.Flatten(),
                layers.Dense(units=128, activation='relu'),
                layers.Dense(units=64, activation='relu'),
                layers.Dense(units=32, activation='relu'),
                layers.Dense(units=16, activation='relu'),
                layers.Dense(len(CLASSES))],
                name='MLP_128-64-32-16'
            )
    else:  # Vanilla MLP with 1 hidden layer of 128 neurons
        return Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            layers.Flatten(),
            layers.Dense(units=128, activation='relu'),
            layers.Dense(len(CLASSES))],
            name='Vanilla_MLP_128'
        )
