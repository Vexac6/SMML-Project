import tensorflow as tf
import matplotlib.pyplot as plt

import general_settings as settings
import load_data
import various_models
import plottings

ARCHITECTURES = 2
USER_OPTIONS = 3
selected_model = ''


def main_menu(model_type):

    menu_loop = True

    while menu_loop:
        print('\t(1) [' + model_type + ']Show me the current model\n'
              '\t(2) [' + model_type + ']Choose or edit a model\n',
              '\t(3) [' + model_type + ']Train and Test current model\n',
              '\t(0) Back\n')
        menu_choice = input()
        if valid_choice(menu_choice, USER_OPTIONS):
            if int(menu_choice) == 0:
                menu_loop = False
            else:
                model_menu(model_type, int(menu_choice))
        else:
            print('Invalid input, please retry.\n')


def model_menu(model_type, menu_choice):

    global selected_model

    if menu_choice == 1:
        if selected_model == '':
            print('Select a model to see its details!')
        else:
            selected_model.summary()
    elif menu_choice == 2:
        if model_type == 'CNN':
            selected_model = cnn_submenu()
        elif model_type == 'MLP':
            selected_model = mlp_submenu()
    elif menu_choice == 3:
        if selected_model == '':
            print('Select a model before asking to train it!')
        else:
            train_and_test(selected_model)


def cnn_submenu():
    print('\t(1) Scale number of filters\n'
          '\t(2) Change Kernel size\n',
          '\t(3) Change activation function\n',
          '\t(4) Pick a predefined CNN model\n',
          '\t(0) Pick the Base Model\n')
    cnn_choice = input()
    if valid_choice(cnn_choice, 4):

        if int(cnn_choice) == 0:
            return various_models.base_model()

        elif int(cnn_choice) == 1:
            print('Insert a positive integer to upscale the filters by that number.\n'
                  'Use a negative integer (with "-") to downscale the filters instead: ')
            filters_scale = input()
            try:
                if '-' in filters_scale:
                    filters_scale = filters_scale.lstrip('-')
                    if not int(filters_scale) == 0:
                        return various_models.base_model(units_multiplier=1 / int(filters_scale))
                    else:
                        print('Please don\'t choose zero, you moron. I\'m watching you.\n')
                        return various_models.base_model()
                else:
                    if not int(filters_scale) == 0:
                        return various_models.base_model(units_multiplier=int(filters_scale))
                    else:
                        print('Please don\'t choose zero, you moron. I\'m watching you.\n')
                        return various_models.base_model()
            except ValueError:
                print('Invalid input, please retry.\n')
                return various_models.base_model()

        elif int(cnn_choice) == 2:
            print('Choose another kernel size for the Base: ')
            kernel_choice = input()
            try:
                if int(kernel_choice) > 0:
                    return various_models.base_model(kernel_size=int(kernel_choice))
                else:
                    print('Please choose a positive kernel size.\n')
                    return various_models.base_model()
            except ValueError:
                print('Invalid input, please retry.\n')
                return various_models.base_model()
        elif int(cnn_choice) == 3:
            print('\t(0) Softplus\n'
                  '\t(1) Hyperbolic Tangent\n',
                  '\t(2) Sigmoid\n')
            activation_choice = input()
            if valid_choice(activation_choice, 3):
                if int(activation_choice) == 0:
                    return various_models.base_model(activation='softplus')
                elif int(activation_choice) == 1:
                    return various_models.base_model(activation='tanh')
                elif int(activation_choice) == 2:
                    return various_models.base_model(activation='sigmoid')
            else:
                print('Invalid input, please retry.\n')
                return various_models.base_model()

        elif int(cnn_choice) == 4:
            print('\t(0) Base Model without Max Pooling layers\n'
                  '\t(1) Base Model with only 1 convolutional layer\n',
                  '\t(2) Base Model with 2 convolutional layers\n',
                  '\t(3) Base Model without the FC layer\n'
                  '\t(4) Base Model with 4 convolutional layers\n',
                  '\t(5) Base Model with 5 convolutional layers\n',
                  '\t(6) Base Model with 3 layers with two subsequent convolutions\n'
                  )
            predefined_model_choice = input()
            if valid_choice(predefined_model_choice, 6):
                return various_models.predefined_base_model(int(predefined_model_choice))
            else:
                print('Invalid input, please retry.\n')
                return various_models.base_model()
    else:
        print('Invalid input, please retry.\n')
        return various_models.base_model()


def mlp_submenu():
    print('\t(1) ↑ MLP with two layers 64-128 neurons\n'
          '\t(2) ↑ MLP with three layers 32-64-128 neurons\n',
          '\t(3) ↑ MLP with four layers 16-32-64-128 neurons\n',
          '\t(4) ↓ MLP with two layers 128-64 neurons\n'
          '\t(5) ↓ MLP with three layers 128-64-32 neurons\n',
          '\t(6) ↓ MLP with four layers 128-64-32-16 neurons\n',
          '\t(0) Pick the Vanilla MLP with 128 neurons\n'
          )
    mlp_choice = input()
    if valid_choice(mlp_choice, 6):
        if int(mlp_choice) == 0:
            return various_models.mlp_model(num_layers=1)
        elif int(mlp_choice) == 1:
            return various_models.mlp_model(2, 'inc')
        elif int(mlp_choice) == 2:
            return various_models.mlp_model(3, 'inc')
        elif int(mlp_choice) == 3:
            return various_models.mlp_model(4, 'inc')
        elif int(mlp_choice) == 4:
            return various_models.mlp_model(2, 'dec')
        elif int(mlp_choice) == 5:
            return various_models.mlp_model(3, 'dec')
        elif int(mlp_choice) == 6:
            return various_models.mlp_model(4, 'dec')
    else:
        print('Invalid input, please retry.\n')
        return various_models.mlp_model(num_layers=1)


def valid_choice(ch, options):
    try:
        number = int(ch)
        if 0 <= number <= options:
            return True
        else:
            return False
    except ValueError:
        return False


def train_and_test(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics='accuracy')

    history = model.fit(training_set, validation_data=validation_set, epochs=settings.EPOCHS)

    test_results = model.evaluate(test_set, verbose=1)
    accuracy_percentage = round(test_results[1], 4) * 100

    # Plot training, validation and test accuracy
    plottings.plot_accuracy(model_name='User-Defined Model',
                            epochs_range=range(settings.EPOCHS),
                            tr_acc=history.history['accuracy'],
                            val_acc=history.history['val_accuracy'],
                            acc=accuracy_percentage)

    plt.show()


if __name__ == '__main__':
    training_set, validation_set = load_data.generate_training_set(settings.TRAINING_DIR,
                                                                   settings.CLASSES,
                                                                   settings.IMG_WIDTH,
                                                                   settings.IMG_HEIGHT,
                                                                   settings.BATCH_SIZE)

    test_set, test_labels = load_data.generate_test_set(settings.TEST_DIR,
                                                        settings.CLASSES,
                                                        settings.IMG_WIDTH,
                                                        settings.IMG_HEIGHT,
                                                        settings.BATCH_SIZE)

    print('Welcome user. Please choose the type of model you want to train and test: ')
    looping = True

    while looping:
        print('\t(1) Convolutional Neural Network\n'
              '\t(2) Multilayer Perceptron\n'
              '\t(0) Close\n')
        choice = input()
        if valid_choice(choice, ARCHITECTURES):
            if int(choice) == 0:
                looping = False
            elif int(choice) == 1:
                main_menu('CNN')
            elif int(choice) == 2:
                main_menu('MLP')
        else:
            print('Invalid input, please retry.\n')
