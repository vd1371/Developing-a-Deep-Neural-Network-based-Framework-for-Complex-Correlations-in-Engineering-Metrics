import logging
import os, sys
import time
import pprint

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import keras


class Report(object):
    def __init__(self, name = None):
        super(Report, self).__init__()
        
        self.name = name
        
        self.directory = os.path.dirname(__file__) + "/" + self.name
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        logging_address = os.path.join(self.directory, 'Report.log')
        self.log = Logger(logger_name = self.name + '-Logger', address = logging_address , mode='a')
    
    
    def evaluate_regression(self, y_true, y_pred, inds, label):
        
        # Saviong into csv file
        report = pd.DataFrame()
        report['Actual'] = y_true
        report['Predicted'] = y_pred
        report['Ind'] = inds
        report.set_index('Ind', inplace=True)
        report.to_csv(self.directory + "/"+ f'{label}.csv')
        
        report_str = f"{label}, CorCoef= {CorCoef(y_true, y_pred):.2f}, R2= {R2(y_true, y_pred):.2f}, RMSE={mean_squared_error(y_true, y_pred)**0.5:.2f}, MSE={mean_squared_error(y_true, y_pred):.2f}, MAE={mean_absolute_error(y_true, y_pred):.2f}, MAPE={MAPE(y_true, list(y_pred)):.2f}%"
        self.log.info(report_str)
        print(report_str)
    
        
        # let's order them
        temp_list = []
        for true, pred in zip(y_true, y_pred):
            temp_list.append([true, pred])
        temp_list = sorted(temp_list , key=lambda x: x[0])
        y_true, y_pred = [], []
        for i, pair in enumerate(temp_list):
            y_true.append(pair[0])
            y_pred.append(pair[1])
            
        # Actual vs Predicted plotting
        plt.clf()
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(label+'-Actual vs. Predicted')
        ac_vs_pre = plt.scatter(y_true, y_pred, s = 0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=0.75)
        plt.grid(True)
        plt.savefig(self.directory + '/ACvsPRE-' + label + '.png')
        plt.close()
        
        # Actual and Predicted Plotting
        plt.clf()
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title(label + "-Actual and Predicted")
        x = [i for i in range(len(y_true))]
        act = plt.plot(x, y_true, label = "actual")
        pred = plt.plot (x, y_pred, label = 'predicted')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.directory + '/ACandPRE-' + label + '.png')
        plt.clf()
        plt.close()
    
    def shap_deep_regression(self, model, x_train, x_test, cols, num_top_features = 10, label = 'DNN-OnTest'):
    
        explainer = shap.DeepExplainer(model, x_train.values)
        shap_values = explainer.shap_values(x_test.values)
    
        shap_values = pd.DataFrame(shap_values[0], columns = list(x_train.columns)).abs().mean(axis = 0)
        self.log.info(f"SHAP Values {label}\n" + pprint.pformat(shap_values.nlargest(num_top_features)))
        
        ax = shap_values.nlargest(num_top_features).plot(kind='barh', title = 'Shap Values - Features Importance')
        fig = ax.get_figure()
        
        fig.savefig(self.directory + "/"+ f'FS-{label}.png')
        del fig
        plt.close()


class PlotLosses(keras.callbacks.Callback):
    
    def __init__(self, num):
        self.num = num
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        plt.ion()
        plt.clf()
        plt.title(f'Step {self.num}-epoch:{epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        pointer = int(self.i/200)*200
        plt.plot(self.x[int(pointer/2):], self.losses[int(pointer/2):], label="loss")
        plt.plot(self.x[int(pointer/2):], self.val_losses[int(pointer/2):], label = 'cv_loss')
        
        plt.legend()
        plt.grid(True, which = 'both')
        plt.draw()
        plt.pause(0.00001)
    
    def closePlot(self, dir= '', should_save = False):
        if should_save:
            plt.savefig(dir)
        plt.close()

class Logger(object):
    
    instance = None

    def __init__(self, logger_name = 'Logger', address = '',
                 level = logging.DEBUG, console_level = logging.ERROR,
                 file_level = logging.DEBUG, mode = 'w'):
        super(Logger, self).__init__()
        if not Logger.instance:
            logging.basicConfig()
            
            Logger.instance = logging.getLogger(logger_name)
            Logger.instance.setLevel(level)
            Logger.instance.propagate = False
    
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            Logger.instance.addHandler(console_handler)
            
            file_handler = logging.FileHandler(address, mode = mode)
            file_handler.setLevel(file_level)
            formatter = logging.Formatter('%(asctime)s-%(levelname)s- %(message)s')
            file_handler.setFormatter(formatter)
            Logger.instance.addHandler(file_handler)
    
    def _correct_message(self, message):
        output = "\n----------------------------------------------------------\n"
        output += message
        output += "\n---------------------------------------------------------\n"
        return output
        
    def debug(self, message):
        Logger.instance.debug(self._correct_message(message))

    def info(self, message):
        Logger.instance.info(self._correct_message(message))

    def warning(self, message):
        Logger.instance.warning(self._correct_message(message))

    def error(self, message):
        Logger.instance.error(self._correct_message(message))

    def critical(self, message):
        Logger.instance.critical(self._correct_message(message))

    def exception(self, message):
        Logger.instance.exception(self._correct_message(message))

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        print (f'---- {method.__name__} is about to start ----')
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print (f'---- {method.__name__} is done in {te-ts:.2f} seconds ----')
        return result
    return timed


def MAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    try:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except ZeroDivisionError:
        return 0

def R2(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]**2

def CorCoef(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]
    

