import csv
import numpy as np


def load_csv_data(filename):
    a = []
    index = True
    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            if index:
                index = False
            else:
                a.append(row)
    csv_file.close()
    return np.array(a)


male_train_data = load_csv_data("data/male_train_data.csv")
female_train_data = load_csv_data("data/female_train_data.csv")
print(male_train_data)
print(female_train_data)
