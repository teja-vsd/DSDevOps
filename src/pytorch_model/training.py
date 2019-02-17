import torch
import torch.nn as nn
from datetime import datetime
from src.pytorch_model.params import network_parameters_1_0, config_parameters_1_0
import pandas as pd
import os
from sklearn import preprocessing
import torch.optim as optim
import time
import copy
import numpy as np


def get_linear_layer(in_dim, out_dim, drp_rate):
    return nn.Sequential(nn.Linear(in_dim, out_dim),
                         nn.ReLU(inplace=True),
                         nn.BatchNorm1d(out_dim),
                         nn.Dropout(p=drp_rate))

class pytorchNN(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim, drp_rate):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.linear_layer1 = get_linear_layer(input_dim, hidden1, drp_rate)
        self.linear_layer2 = get_linear_layer(hidden1, hidden2, drp_rate)
        self.linear_layer3 = get_linear_layer(hidden2, output_dim, 0)

    def forward(self, input):
        y = self.input_bn(input)
        y = self.linear_layer1(y)
        y = self.linear_layer2(y)
        y = self.linear_layer3(y)
        return y


def create_input_and_output(df):
    data_array = df[['x0', 'x1', 'x2', 'x3']].values
    target_array = df['y'].values
    return data_array, target_array


def create_batches(data_array, target_array, batch_size, inp_dim):
    data_array = data_array.reshape(batch_size, -1, inp_dim)
    target_array = target_array.reshape(batch_size, -1)
    return data_array, target_array


def get_num_classes(targets):
    return np.unique(targets).shape[0]


def get_input_dim(inputs):
    return inputs.shape[-1]


def get_classes(targets):
    return np.unique(targets)


def create_label_encoder(targets):
    le = preprocessing.LabelEncoder()
    le.fit(targets)
    return le


def train(network_parameters, config_parameters):
    train_df = pd.read_csv(os.path.join(network_parameters['dataset_path'], 'train', 'train.csv'))
    validation_df = pd.read_csv(os.path.join(network_parameters['dataset_path'], 'validation', 'validation.csv'))

    training_data_array, training_target_array = create_input_and_output(train_df)
    validation_data_array, validation_target_array = create_input_and_output(validation_df)

    nn_input_dim = get_input_dim(training_data_array)
    nn_output_unique = get_classes(training_target_array)
    nn_output_dim = get_num_classes(training_target_array)

    le = create_label_encoder(nn_output_unique)
    training_target_array = le.transform(training_target_array)
    validation_target_array = le.transform(validation_target_array)

    training_data_array, training_target_array = create_batches(training_data_array, training_target_array, network_parameters['batch_size'], nn_input_dim)
    validation_data_array, validation_target_array = create_batches(validation_data_array, validation_target_array, network_parameters['batch_size'], nn_input_dim)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = pytorchNN(nn_input_dim, network_parameters['hidden1'], network_parameters['hidden2'], nn_output_dim, network_parameters['drp_rate'])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=network_parameters['lr'], weight_decay=network_parameters['weight_decay'])

    training_data_array = torch.Tensor(training_data_array).to(device)
    training_target_array = torch.Tensor(training_target_array).to(device)
    validation_data_array = torch.Tensor(validation_data_array).to(device)
    validation_target_array = torch.Tensor(validation_target_array).to(device)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(network_parameters['epochs']):
        print('Epoch {}/{}'.format(epoch, network_parameters['epochs'] - 1))
        print('-' * 10)
        model.train()

        running_loss = 0.0
        running_corrects = 0

        model.train()

        for batch_num in range(training_data_array.shape[0]):
            batch_input = training_data_array[batch_num].float()
            batch_targets = training_target_array[batch_num].long()
            optimizer.zero_grad()
            outputs = model(batch_input)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss
            running_corrects += torch.sum(preds == batch_targets)

        epoch_train_loss = running_loss / training_data_array.shape[0]
        epoch_train_acc = running_corrects.float() / training_target_array.numel()

        running_loss = 0.0
        running_corrects = 0

        model.eval()

        for v_batch_num in range(validation_data_array.shape[0]):
            batch_input = validation_data_array[v_batch_num].float()
            batch_targets = validation_target_array[v_batch_num].long()
            outputs = model(batch_input)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, batch_targets)

            running_loss += loss
            running_corrects += torch.sum(preds == batch_targets)

        epoch_val_loss = running_loss / validation_data_array.shape[0]
        epoch_val_acc = running_corrects.float() / validation_target_array.numel()

        print('Train Loss: {:.6f} Acc: {:.6f}'.format(epoch_train_loss, epoch_train_acc))
        print('Val Loss: {:.6f} Acc: {:.6f}'.format(epoch_val_loss, epoch_val_acc))
        print()

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    if config_parameters['save_model']:
        model_save_path = os.path.join(config_paramters['model_save_path'], timestamp + '_' + type(model).__name__)
        os.makedirs(model_save_path)
        torch.save(model.state_dict(), os.path.join(model_save_path, timestamp + '_model.pt'))


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    network_parameters = network_parameters_1_0
    config_paramters = config_parameters_1_0
    np.random.seed(network_parameters['random_seed'])
    torch.manual_seed(network_parameters['random_seed'])
    train(network_parameters, config_paramters)

