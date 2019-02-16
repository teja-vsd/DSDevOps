import pandas as pd
import os


def load_data(rdp):
    df = pd.read_csv(rdp, header=None)
    return df

def get_columns():
    return ['x0', 'x1', 'x2', 'x3', 'y']

def create_datasets(rdf):
    rdf = rdf.sample(frac=1)
    train_df = rdf.iloc[:100, :]
    validation_df = rdf.iloc[100:125, :]
    test_df = rdf.iloc[125:, :]
    return train_df, validation_df, test_df

def main(rdp):
    print('Preprocessing Data')
    raw_data_df = load_data(rdp)
    raw_data_df.columns = get_columns()
    train_df, validation_df, test_df = create_datasets(raw_data_df)
    dataset_save_dir = os.path.join('..', '..', 'data', 'processed_data')
    train_df.to_csv(os.path.join(dataset_save_dir, 'train', 'train.csv'))
    validation_df.to_csv(os.path.join(dataset_save_dir, 'validation', 'validation.csv'))
    test_df.to_csv(os.path.join(dataset_save_dir, 'test', 'test.csv'))


if __name__ == '__main__':
    raw_data_path = os.path.join('..', '..', 'data', 'raw_data', 'iris_data.csv')
    main(raw_data_path)