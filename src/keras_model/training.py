from datetime import datetime
from src.keras_model.params import network_parameters_1_0, config_parameters_1_0
import numpy as np
from keras.layers import Input, Dense, BatchNormalization, AlphaDropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder


def build_model(input_dim, hidden1, hidden2, output_dim, drpout_rate, lr):
    inputs = Input(shape=(input_dim,))
    dense_layer1 = Dense(name='first_dense_layer',
                         units=hidden1,
                         activation='relu')
    dense_layer2 = Dense(name='second_hidden_layer',
                         units=hidden2,
                         activation='relu')
    dense_layer3 = Dense(name='final_layer',
                         units=output_dim,
                         activation='softmax')
    input_bn_layer = BatchNormalization(name='input_batchnorm_layer')
    bn_layer1 = BatchNormalization(name='first_batchnorm_layer')
    bn_layer2 = BatchNormalization(name='second_batchnorm_layer')
    dropout_layer = AlphaDropout(rate=drpout_rate)

    y = input_bn_layer(inputs)
    y = dense_layer1(y)
    y = bn_layer1(y)
    y = dropout_layer(y)
    y = dense_layer2(y)
    y = bn_layer2(y)
    y = dropout_layer(y)
    outputs = dense_layer3(y)

    model = Model(inputs=inputs, outputs=outputs)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def create_input_and_output(df):
    data_array = df[['x0', 'x1', 'x2', 'x3']].values
    target_array = df['y'].values
    return data_array, target_array


def get_num_classes(targets):
    return np.unique(targets).shape[0]


def get_input_dim(inputs):
    return inputs.shape[-1]


def get_classes(targets):
    return np.unique(targets)


def create_ohe(targets):
    ohe = OneHotEncoder()
    ohe.fit(targets)
    return ohe


def train(network_parameters, config_parameters):
    train_df = pd.read_csv(os.path.join(network_parameters['dataset_path'], 'train', 'train.csv'))
    validation_df = pd.read_csv(os.path.join(network_parameters['dataset_path'], 'validation', 'validation.csv'))

    training_data_array, training_target_array = create_input_and_output(train_df)
    validation_data_array, validation_target_array = create_input_and_output(validation_df)

    nn_input_dim = get_input_dim(training_data_array)
    nn_output_unique = get_classes(training_target_array)
    nn_output_dim = get_num_classes(training_target_array)

    ohe = create_ohe(nn_output_unique.reshape(-1, 1))
    training_target_array = ohe.transform(training_target_array.reshape(-1, 1))
    validation_target_array = ohe.transform(validation_target_array.reshape(-1, 1))

    model = build_model(nn_input_dim,
                        network_parameters['hidden1'],
                        network_parameters['hidden2'],
                        nn_output_dim,
                        network_parameters['drp_rate'],
                        network_parameters['lr'])

    model.fit(x=training_data_array,
              y=training_target_array,
              batch_size=network_parameters['batch_size'],
              epochs=network_parameters['epochs'],
              validation_data=(validation_data_array,validation_target_array),
              shuffle=False)

    if config_paramters['save_model']:
        model_save_path = os.path.join(config_paramters['model_save_path'], timestamp + '_keras_' + type(model).__name__)
        os.makedirs(model_save_path)
        model.save(os.path.join(model_save_path, timestamp + '_model.h5'))


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    network_parameters = network_parameters_1_0
    config_paramters = config_parameters_1_0
    np.random.seed(network_parameters['random_seed'])
    train(network_parameters, config_paramters)