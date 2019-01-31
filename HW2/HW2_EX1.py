import numpy as np


def load_csv_data(filename):  # load data from given csv file to a numpy array
    a = []
    index = True
    reader = np.genfromtxt(filename, delimiter=',')
    for row in reader:
        if index:
            index = False
        else:
            a.append(row)
    return np.array(a)


if __name__ == '__main__':
    male_train_data = load_csv_data("data/male_train_data.csv")
    female_train_data = load_csv_data("data/female_train_data.csv")
    print(male_train_data)
    print(female_train_data)
