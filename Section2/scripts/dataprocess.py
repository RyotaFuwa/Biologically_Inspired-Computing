import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from tensorflow.keras.utils import to_categorical

class DataProcess(object):

    @staticmethod
    def import_data(file):
        if "csv" in file:
            data = pd.read_csv(file)
        elif "arff" in file:
            arff_data = arff.loadarff(file)
            if type(arff_data) == tuple:
                for d in arff_data:
                    if type(d) == np.ndarray:
                            data = pd.DataFrame(d)
                            break
            else:
                data = pd.DataFrame(file)
        else:
            print("Not Compatible File Loader Found")
        return data 

    @staticmethod 
    def preProcessData(data):
        #normalization 
        train_in = data.loc[:, ['x', 'y']].to_numpy().astype("float32")
        train_in /= train_in.max() 
        #one hot encoding 
        train_label = data.loc[:, ['class']]
        train_label = np.where(train_label == b'black', 0, 1) 
        train_label = to_categorical(train_label)
        return train_in, train_label


