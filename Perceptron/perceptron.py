#!/usr/bin/python3
# -*- coding: utf-8 -*-

## SPDX-License-Identifier: Unlicense
##
## Done in 2019 by Patrick Rećko. The other guy dissapeared (I heard that he's at full-time job, why choosing day studies?)
##
## To run this mess, Install pandas (used for data loading), PyQT5 (gui), numpy (maths and arrays) and matplotlib (graph drawing)
## and make sure thet you're using python 3 since python 2.x will be dead in a year.
## then run the file using command: python perceptron.py 
##
## There might be still bugs and limitations:
##     - It only loads .csv (and .txt) data with the ';' separator. Probably needs to add import option in QFileDIalog for that
##     - It might explode with less than 2 variables and when the text is in data
##     - Tangens might be finnicky, IDK if ReLU works.
##     - This was only tested on Windows 10, Linux might works but it wasn't that much tested.
##
## Frankly this was mostly done in a few days. Plus most of the time was for learning basics of external APIs.
## The network was implemented in day of time.
##
## Feel free to use it whatever you want, I don't care.

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal, pyqtSlot, QAbstractTableModel, QModelIndex, QVariant
from PyQt5.QtGui import QPainter, QColor 
import sys, random
from enum import Enum

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

class PercGUI(QMainWindow):
###
    def __init__(self):
    ## 
        super().__init__()
        self.isTrained = False
        self.isTested = False
        self.isLoaded = False
        self.initUI()
    ##
    
    def initUI(self):
        '''Initializes the GUI'''
    ##
        self.setFixedSize(1000, 450)
        self.center()
        self.setWindowTitle('Perceptron')

        
        self.graph = PercPointGraph(self, width=5, height=4)
        self.graph.move(0,0)
        
        self.compute_button = QPushButton('Train', self)
        self.compute_button.move(500,400)
        self.compute_button.setToolTip("Performs training on train data set. <br/><br/><b>NOTE:</b>Will reset the state of the previously trained network.")
        
        self.load_button = QPushButton('Load', self)
        self.load_button.move(600,400)
        self.load_button.setToolTip("Loads the new data set and randomly separate them by train and test one. <br/><br/><b>NOTE:</b>Will reset the state of the trained network.")
        
        self.test_button = QPushButton('Test', self)
        self.test_button.move(700,400)
        self.test_button.setToolTip("Tests the trained network.")
        
        self.save_button = QPushButton('Save', self)
        self.save_button.move(800,400)
        self.save_button.setToolTip("Save all results.")
        
        self.tableView = QTableView(self)
        self.tableView.move(500, 20)
        self.tableView.resize(250, 100)
         
        self.tableViewTest = QTableView(self)
        self.tableViewTest.move(750, 20)
        self.tableViewTest.resize(250, 100)        

        self.weightTableView = QTableView(self)
        self.weightTableView.move(500, 200)
        self.weightTableView.resize(250, 100)
        
        self.resultsTableView = QTableView(self)
        self.resultsTableView.move(750, 200)
        self.resultsTableView.resize(250, 100)
        
        self.dsetLabel = QLabel("Train Data Set", self)
        self.dsetLabel.move(500, -5)
        
        self.tsetLabel = QLabel("Test Data set", self)
        self.tsetLabel.resize(300, 20)
        self.tsetLabel.move(750, -5)

        self.actLabel = QLabel("Activation function", self)
        self.actLabel.move(500, 120)
        
        self.iterLabel = QLabel("Number of iterations", self)
        self.iterLabel.move(500, 155)
        
        self.weightChangeLabel = QLabel("Weight Change", self)
        self.weightChangeLabel.move(500, 175)
        
        self.resultsLabel = QLabel("Network answers (from test data)", self)
        self.resultsLabel.resize(300, 20)
        self.resultsLabel.move(750, 182)
        
        self.iterLineEdit = QLineEdit(self)
        self.iterLineEdit.setText('10')
        self.iterLineEdit.move(800, 155)
        
        self.selectCombo = QComboBox(self)
        self.selectCombo.addItem("Sigmoid")
        self.selectCombo.addItem("Tangens")
        self.selectCombo.addItem("ReLU")
        self.selectCombo.setToolTip("Type of activation function to use <br/><br/><b>NOTE:</b>This will reset the state of the trained network.")
        self.selectCombo.move(800, 120)
        
        self.compute_button.clicked.connect(self.on_click_compute)
        self.load_button.clicked.connect(self.on_click_load)
        self.test_button.clicked.connect(self.on_click_test)
        self.save_button.clicked.connect(self.on_click_save)
        
        self.statusBar().showMessage('Ready')
        
        QApplication.processEvents()
        
        self.show()
        
    ##
    
    def center(self):
        '''centers the window on the screen'''
    ##
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width()-size.width())/2, 
            (screen.height()-size.height())/2)
    ##
    
    @pyqtSlot()
    def on_click_compute(self):
        '''handles the train button'''
    ##
        if (self.isLoaded == False):
            self.mb = QMessageBox()
            self.mb.setIcon(QMessageBox.Warning)
            self.mb.setWindowTitle('Error')
            self.mb.setText('Data set not loaded.')
            self.mb.setStandardButtons(QMessageBox.Ok)
            self.mb.show()
            return
        self.statusBar().showMessage('Computing...')
        QApplication.processEvents()
        tmparr = self.trainDataSet.values
        input = tmparr[:,:-1]
        tmpdf = self.trainDataSet[self.trainDataSet.columns[tmparr.shape[1]-1]]
        output = np.array([tmpdf.values]).T
        if self.selectCombo.currentText() == "Sigmoid":
            self.nn = PercOneNeuronNN(PercNeuronActivationType.Sigmoid, tmparr.shape[1]-1)
        elif self.selectCombo.currentText() == "Tangens":
            self.nn = PercOneNeuronNN(PercNeuronActivationType.TanH, tmparr.shape[1]-1)
        elif self.selectCombo.currentText() == "ReLU":
            self.nn = PercOneNeuronNN(PercNeuronActivationType.Real, tmparr.shape[1]-1)
        self.nn.perform_train(input, output, int(self.iterLineEdit.text()))
        self.graph.plot_with_data(self.nn.error_set)
        model = NumpyModel(self.nn.weight_change_set)
        self.weightTableView.setModel(model)
        self.statusBar().showMessage('Ready')
        self.isTrained = True
        self.isTested = False
        
        # show final values
        self.mb = QMessageBox()
        self.mb.setIcon(QMessageBox.Information)
        self.mb.setWindowTitle('Results')
        str = 'Trained network input weights are: \n' + np.array2string(self.nn.weights)
        self.mb.setText(str)
        self.mb.setStandardButtons(QMessageBox.Ok)
        self.mb.show()
        
    ##
    @pyqtSlot()
    def on_click_load(self):
        '''handles the load button '''
    ##
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Open data set", "","All Files (*);;Text File (*.txt);;CSV File (*.csv)", options=options)
        if fileName:
            ## TODO: Hardcoded, this should be setable
            self.dataset = pd.read_csv(fileName, sep = ';', header = None)
            self.testDataSet = self.dataset.sample(frac = 0.5)
            self.trainDataSet = self.dataset[~self.dataset.index.isin(self.dataset.merge(self.testDataSet.assign(a='key'),how='left').dropna().index)]
                        
            model = NumpyModel(self.trainDataSet.to_numpy())
            testModel = NumpyModel(self.testDataSet.to_numpy())
            self.tableView.setModel(model)
            self.tableViewTest.setModel(testModel)
            self.isLoaded = True
            self.isTrained = False
            self.isTested = False
            
    ##
    
    @pyqtSlot()
    def on_click_test(self):
        ''' Handles the test button '''
        if (self.isTrained == False):
            self.mb = QMessageBox()
            self.mb.setIcon(QMessageBox.Warning)
            self.mb.setWindowTitle('Error')
            self.mb.setText('Neural Network not trained.')
            self.mb.setStandardButtons(QMessageBox.Ok)
            self.mb.show()
            return
        tmp = self.testDataSet.values[:,:-1]
        self.resultsStore = np.empty([tmp.shape[0], 1])
        for i in range(tmp.shape[0]):
            self.resultsStore[i] = self.nn.update_weight(tmp[i])
        
        model = NumpyModel(self.resultsStore)
        self.resultsTableView.setModel(model)
        self.isTested = True
    
    @pyqtSlot()
    def on_click_save(self):
        ''' Handles the save button'''
        self.mb = QMessageBox()
        self.mb.setIcon(QMessageBox.Warning)
        self.mb.setWindowTitle('Error')
        self.mb.setStandardButtons(QMessageBox.Ok)
        if (self.isLoaded == False):
            self.mb.setText('Data set not loaded.')
            self.mb.show()
            return
        if (self.isTrained == False):
            self.mb.setText('Neural Network not trained.')
            self.mb.show()
            return
        if (self.isTested == False):
            self.mb.setText('Neural Network not tested.')
            self.mb.show()
            return
        dirName = QFileDialog.getExistingDirectory(None, 'Select a folder to save:', './', QFileDialog.ShowDirsOnly)
        if dirName:
            np.savetxt(dirName + '/' + 'test_set.txt', self.testDataSet.to_numpy(), delimiter=';')
            np.savetxt(dirName + '/' + 'train_set.txt', self.trainDataSet.to_numpy(), delimiter=';')
            np.savetxt(dirName + '/' + 'weight_change.txt', self.nn.weight_change_set, delimiter=';')
            np.savetxt(dirName + '/' + 'results_from_test_set.txt', self.resultsStore, delimiter=';')
            self.graph.save(dirName + '/' + 'error_change.png')
###


class PercNeuronActivationType(Enum):
    ''' Type of the activation function '''
    Sigmoid = 0
    Real = 1
    TanH = 2
    

class PercOneNeuronNN():
    '''Implements the entire logic for the one neuron-based neural network'''
###
    def __init__(self, act_func, num_weights):
        np.random.seed(1)
        self.num_weights = num_weights
        self.weights = 2 * np.random.random((num_weights, 1)) - 1
        self.act_func = act_func
        self.error_set = []
        self.weight_change_set=[]
    
    def activate(self, act_func, total):
        ''' Return the value siphoned through the activation function'''
        if act_func == PercNeuronActivationType.Sigmoid:
            return 1 / (1 + np.exp(-total))
        elif act_func == PercNeuronActivationType.Real:
            return np.maximum(0, total)
        elif act_func == PercNeuronActivationType.TanH:
            return np.tanh(total)
        
    def update_weight(self, inputs):
        ''' Update the weights for entire neural network '''
        total = np.dot(inputs, self.weights)
        return self.activate(self.act_func, total)
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2
        
    def real_derivative(self, x):
        #return np.heaviside(x, 0)
        return x * (x >= 0)
    
    def mse(self, y_true, y_pred):
        ''' Get the mean square error from the weights (from dataset and network)'''
        return np.mean(np.power(y_true-y_pred, 2));
    
    def perform_train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        '''Performs the actual train loop.
           Inputs: training_set_inputs - array containing the input (weights) variables data
                   training_set_outputs - one-column array containing the response/decision for the data set
                   number_of_training_iterations - self explanatory
        '''
        self.error_set = []
        self.weight_change_set = np.array(np.swapaxes(self.weights, 0, 1))
        for iteration in range(number_of_training_iterations):
            output = self.update_weight(training_set_inputs)
            error = training_set_outputs - output
            tmperr = 0
            for i in range(error.shape[0]):
                tmperr += self.mse(error[i], output[i])
            
            tmperr /= error.shape[0]
            
            self.error_set.append(tmperr)
            
            if self.act_func == PercNeuronActivationType.Sigmoid:
                adjustment = np.dot(training_set_inputs.T, error * self.sigmoid_derivative(output))
            elif self.act_func == PercNeuronActivationType.TanH:
                adjustment = np.dot(training_set_inputs.T, error * self.tanh_derivative(output))
            elif self.act_func == PercNeuronActivationType.Real:
                adjustment = np.dot(training_set_inputs.T, error * self.real_derivative(output))
            
            self.weights += adjustment
            self.weight_change_set = np.append(self.weight_change_set, np.swapaxes(self.weights, 0, 1), axis=0)
            
    def think(self, inputs):
        ''' outputs the result of the network on given set of inputs '''
        return self.update_weight(self, inputs)
    
    def reset(self, act_func, num_weights):
        self.act_func = act_func
        self.weights = []
        self.weights = 2 * np.random.random((num_weights, 1)) - 1
        

class PercPointGraph(FigureCanvas):
    ''' Implements the error rate graph '''
###
    def __init__(self, parent=None, width=5, height=4, dpi=100):
    ##
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        
        self.axes.set_title('Error rate', fontsize=20)
        self.axes.set_xlabel('num. of iterations', fontsize=10)
        self.axes.set_ylabel('error value', fontsize=10)
        
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    ##

    def plot(self):
    ##
        self.axes.clear()
        data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        self.axes.set_title('Error rate', fontsize=20)
        self.axes.set_xlabel('num. of iterations', fontsize=10)
        self.axes.set_ylabel('error value', fontsize=10)
        ax.plot(data, 'r-')
        self.draw()
    ##
    
    def plot_with_data(self, data):
    ##
        self.axes.clear()
        ax = self.figure.add_subplot(111)
        self.axes.set_title('Error rate', fontsize=20)
        self.axes.set_xlabel('num. of iterations', fontsize=10)
        self.axes.set_ylabel('error value', fontsize=10)
        ax.plot(data, 'r-')
        self.draw()
    ##
    
    def save(self, fileName):
        self.figure.savefig(fileName)
###

## copied with changes from http://www.riverbankcomputing.com/pipermail/pyqt/attachments/20090528/8ecfadc3/demo_view.py
class NumpyModel(QAbstractTableModel):
    ''' used to present data from NumPy's Arrays to Qt's table view '''
    def __init__(self, narray, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._array = narray

    def rowCount(self, parent=None):
        return self._array.shape[0]

    def columnCount(self, parent=None):
        return self._array.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                row = index.row()
                col = index.column()
                return QVariant("%.5f"%self._array[row, col])
        return QVariant()

if __name__ == '__main__':
    
    app = QApplication([])
    perc = PercGUI()    
    sys.exit(app.exec_())
    
    # TEST CODE BEGIN
    # nn = PercOneNeuronNN(PercNeuronActivationType.Sigmoid, 3)
    
    # print ("Random starting weights: ")
    # print (nn.weights)
    
    # training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    # print(training_set_inputs)
    # training_set_outputs = np.array([[0, 1, 1, 0]]).T
    # print(training_set_outputs)
    
    # nn.perform_train(training_set_inputs, training_set_outputs, 10000)
    
    # print("Ending Weights After Training: ")
    # print(nn.weights)
    # print("err list")
    # print(nn.error_set)
    
    
    # print ("Considering new situation [1, 0, 0] -> ?: ")
    # print (nn.update_weight(np.array([1, 0, 0])))
    
    # TEST CODE END
    
##
