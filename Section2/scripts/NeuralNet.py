import os
import time
import yaml

import numpy as np
import pandas as pd

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

from dataprocess import DataProcess
import matplotlib.pyplot as plt



# singleton
class NeuralNet(DataProcess):
    OPTIMIZERSCOLLECTION = {
        'Adadelta': optimizers.Adadelta,
        'Adagrad': optimizers.Adagrad,
        'Adam': optimizers.Adam, 
        'Adamax': optimizers.Adamax, 
        'Ftrl': optimizers.Ftrl, 
        'Nadam': optimizers.Nadam, 
        'Optimizer': optimizers.Optimizer, 
        'RMSprop': optimizers.RMSprop, 
        'SGD': optimizers.SGD
    }

    def __init__(self, train_filename, test_filename, config_filename):
        # store filenames
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.config_filename = config_filename
        # preprocess train data
        self.train_data = DataProcess.import_data(train_filename)
        self.train_in, self.train_label = DataProcess.preProcessData(self.train_data)
        # preprocess test data
        self.test_data = DataProcess.import_data(test_filename)
        self.test_in, self.test_label = DataProcess.preProcessData(self.test_data)
        # Network Config
        self.config = self.import_config(config_filename)
        self.summary = True
        self.setUpNetWork()
        # Network Variables
        self.epoch_count = 0
        self.time = 0.0
        self.accuracy = []
        self.loss_value = []
        self.val_accuracy = []
        self.val_loss_value = []

    def import_config(self, config_filename):
        with open(config_filename, "r") as fin:
            config = yaml.load(fin, Loader=yaml.SafeLoader)
        self.hidden_layers = config['hidden_layers']
        self.layersNum = len(self.hidden_layers) + 1
        self.activation = config['activation']
        self.dropout = config['dropout']
        self.optimizer_info = config['optimizer']
        self.loss = config['loss']
        self.metrics = config['metrics']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.validation_split = config['validation_split']
        return config

    def setUpNetWork(self):
        # Model Topology
        self.model = Sequential()
        for i, num in enumerate(self.hidden_layers): # input layer
            if(i == 0):
                self.model.add(Dense(num, activation=self.activation,
                    input_shape=(2, ), kernel_initializer='uniform'))
                continue
            if self.dropout[i] != 0.0:
                self.model.add(Dropout(self.dropout[i]))
            self.model.add(Dense(num, activation=self.activation)) # hidden layers
        self.model.add(Dense(2, activation='softmax')) # output layers
        if(self.summary): # summary
            self.model.summary()
        # Optimizer Definition
        self.optimizer = NeuralNet.OPTIMIZERSCOLLECTION[self.optimizer_info['name']](**self.optimizer_info['config'])
        # Training Configuration SetUp
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def trainNet(self, epochs=False, verbose=True):
        if epochs is False:
            epochs = self.epochs
        if verbose is True:
            verbose = 1
        else:
            verbose = 0
        start = time.time() 
        self.model.fit(self.train_in, self.train_label,
            epochs=epochs, batch_size=self.batch_size,
            verbose=verbose, validation_split=self.validation_split)
        stop = time.time()
        self.time += stop - start
        self.epoch_count += epochs
        self.loss_value.extend(self.model.history.history['loss'])
        self.accuracy.extend(self.model.history.history['accuracy'])
        try:
            self.val_loss_value.extend(self.model.history.history['val_loss'])
            self.val_accuracy.extend(self.model.history.history['val_accuracy'])
        except:
            pass
    
    def showAccuracy(self):
        plt.plot(self.accuracy)
        plt.plot(self.val_accuracy)
        plt.title("Accuracy in Each Epoch")
        plt.show()

    def showLoss(self):
        plt.plot(self.loss_value)
        plt.plot(self.val_loss_value)
        plt.title("Loss in Each Epoch")
        plt.show()
    
    def visualizeClassification(self, data='combine', check=True):
        if data is 'combine':
            data = np.concatenate((self.train_in, self.test_in), axis=0)
            data_label = np.concatenate((self.train_label, self.test_label), axis=0)
        elif data is 'test':
            data = self.test_in
            data_label = self.test_label
        else:
            data = self.train_in
            data_label = self.train_label

        tmp_cls = self.model.predict(data)
        enum = range(len(tmp_cls))
        maxargs = np.argmax(tmp_cls, axis=1)
        tmp_cls[enum, maxargs] = 1
        tmp_cls = np.where(tmp_cls < 1, 0, 1)
        if check is True:
            class_color = np.where(tmp_cls[:, 0] != data_label[:, 0], -1, tmp_cls[:, 0])
        else:
            class_color = tmp_cls[:, 0]
        
        cond = [class_color == 0, class_color == 1, class_color == -1]
        col = ['black', 'grey', 'red']
        class_color = np.select(cond, col)
        plt.scatter(data[:, 0], data[:, 1], c=class_color)
        plt.show()

    def DefaultClasses(self):
        data = self.train_in
        tmp_cls = self.train_label
        class_color = np.where(tmp_cls[:, 0] == 1, 'grey', 'black')  # since white colour can't be seen
        plt.scatter(data[:, 0], data[:, 1], c=class_color)
        plt.show()

    
    def testNet(self):
        return self.model.evaluate(self.test_in, self.test_label)

    def resetNet(self):
        self.import_config(self.config_filename)
        self.setUpNetWork()
        self.epoch_count = 0
        self.accuracy = []
        self.val_accuracy = []
        self.loss_value = []
        self.val_loss_value = []
        self.time = 0
    
    def saveNet(self, filename):
        if not os.path.exists("models"):
            os.mkdir("models")
        if not os.path.exists("config"):
            os.mkdir("config")
        
        self.model.save("models/{}.h5".format(filename))
        with open("config/{}.yaml".format(filename), 'w') as fout:
            yaml.dump(self.config, fout)
    
    def importModel(self, filename):
        self.model.load_weights("models/{}.h5".format(filename))
        self.config = self.import_config("config/{}.yaml".format(filename))
    
    def MetricesToCSV(self, filename):
        data = pd.DataFrame({'Iteration':range(nn.epoch_count), 'Loss':nn.model.history.history['loss'],
            'Accuracy':nn.model.history.history['accuracy']})
        data.to_csv(filename)
            

if __name__ == "__main__":
    nn = NeuralNet('../training.arff', '../test.arff', "../NN/config/neuralnet.yaml")