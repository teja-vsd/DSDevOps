import requests
import os


def download_file(url, filename):
    print('Downloading from {} to {}'.format(url, filename))
    response = requests.get(url)
    with open(filename,  'wb') as ofile:
        ofile.write(response.content)

if __name__ == '__main__':
    download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                  os.path.join('..', '..', 'data', 'raw_data', 'iris_data.csv'))