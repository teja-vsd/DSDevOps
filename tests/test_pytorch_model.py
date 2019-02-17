import pytest
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join('..'))
import src.pytorch_model.training as training


@pytest.fixture(scope='module')
def load_data():
    df = pd.read_csv(os.path.join('..', 'data', 'processed_data', 'test', 'test.csv'))
    return df


def test_create_input_and_output(load_data):
    data_array, target_array = training.create_input_and_output(load_data)
    assert data_array.shape == (25, 4)
    assert target_array.shape == (25,)


def test_create_batches(load_data):
    data_array, target_array = training.create_input_and_output(load_data)
    data_array, target_array = training.create_batches(data_array, target_array, 5, 4)
    assert data_array.shape == (5, 5, 4)
    assert target_array.shape == (5, 5)


def test_get_num_classes(load_data):
    num_classes = training.get_num_classes(load_data['y'].values)
    assert num_classes == 3


def test_get_input_dim(load_data):
    input_dim = training.get_input_dim(load_data[['x0', 'x1', 'x2', 'x3']].values)
    assert input_dim == 4

