import numpy as np
import csv
from sklearn import datasets


def loading(file_name):
    '''
    Loading in dataset
    :param file_name: name of the file
    :return: d which is a 2 dimensional array consisting the sample of the dataset
             ts which is a 1 dimensional array consisting labels
    '''
    # Loading in Iris data from sklearn library
    if file_name == 'iris.csv':
        iris = datasets.load_iris()
        d = iris.data
        ts = iris.target
    elif file_name == 'test.txt':
        d, ts = [], []

        with open('./datasets/' + file_name, 'r') as file:
            row_count = 0
            for row in file:
                try:
                    if row_count == 0:
                        row_count += 1
                    else:
                        temp_d = []
                        row_sep = row.split('\t')
                        for i in range(len(row_sep)):
                            if i < len(row_sep) - 1:
                                temp_d.append(float(row_sep[i]))
                            else:
                                ts.append(int(row_sep[i]))

                        d.append(temp_d)
                except ValueError:  # Skipping the lines with N/A or any other invalid values
                    continue

        d = np.array(d)
        ts = np.array(ts)
    else:
        # Loading in other datasets from local csv files
        d, ts = csv_loading('./datasets/' + file_name)  # d is data, ts is targets

    return d, ts


def loading_test_file_without_label(file_name):
    d = []
    csv_name = './datasets/' + file_name
    with open(csv_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            try:
                if row_count == 0:
                    row_count += 1
                else:
                    temp_d = []
                    for i in range(len(row)):
                        if i < len(row):
                            temp_d.append(float(row[i]))
                    d.append(temp_d)
            except ValueError:  # Skipping the lines with N/A or any other invalid values
                continue

    return np.array(d)

def loading_test(file_name):
    '''
    Loading data when a seperate testing data file provided
    :param file_name: the name of testing data file
    :return: d which is a 2 dimensional array consisting the sample of the dataset
             ts which is a 1 dimensional array consisting labels
    '''
    d, ts = csv_loading('./datasets/' + file_name)  # d is data, ts is targets

    return d, ts


def csv_loading(csv_name):
    '''
    Loading in CSV files
    :param csv_name: name of CSV file
    :return: Numpy array of data and targets,
            data: 2 dimensional numpy array with shape (x, y)
            targets: 1 dimensional numpy array with shape (x,)
    '''
    d, ts = [], []

    with open(csv_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            try:
                if row_count == 0:
                    row_count += 1
                else:
                    temp_d = []
                    for i in range(len(row)):
                        if i < len(row) - 1:
                            try:
                                temp_d.append(float(row[i]))
                            except ValueError:
                                temp_d.append(float("nan"))
                        else:
                            try:
                                ts.append(int(row[i]))
                            except ValueError:
                                ts.append(float("nan"))

                    d.append(temp_d)
            except ValueError: # Skipping the lines with N/A or any other invalid values
                continue

    return np.array(d), np.array(ts)