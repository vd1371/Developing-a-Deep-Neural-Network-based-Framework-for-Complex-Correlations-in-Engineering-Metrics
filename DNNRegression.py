import os, sys
from Reporter import *

from keras.models import Sequential, load_model
import keras.losses
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from keras.models import model_from_json

        
class Regressor(Report):

    def __init__(self, file_name,
                 name = None,
                 should_shuffle = True,
                 split_size = 0.4):
        
        super(Regressor, self).__init__(name)
        
        #splitting data into train, cross_validation, test
        
        df = pd.read_csv(file_name+".csv", index_col = 0)
        self.X = df.iloc[:,:-1]
        self.Y = df.iloc[:,-1]
        dates = df.index
        
        self.X_train, X_temp, self.Y_train, Y_temp, self.dates_train, dates_temp = train_test_split(self.X, self.Y, dates, test_size=split_size, shuffle=should_shuffle, stratify = None)
        self.X_cv, self.X_test, self.Y_cv, self.Y_test, self.dates_cv, self.dates_test = train_test_split(X_temp, Y_temp, dates_temp, test_size = 0.5, shuffle = should_shuffle)
        
        self.input_dim = len(self.X_train.columns)
        
        self.log.info('-------------- Regression is about to be fit on %s '%self.name)
        
        
    def setLayers(self,layers):
        self.layers=layers
    
    def setInputActivationFunction(self, activation_function):
        self.input_activation_func = activation_function
    
    def setHiddenActivationFunction(self, hidden_activation_func):
        self.hidden_activation_func = hidden_activation_func
    
    def setFinalActivationFunction(self, final_activation_func):
        self.final_activation_func = final_activation_func
    
    def setLossFunction(self, loss_func):
        self.loss_func = loss_func
    
    def setEpochs(self, epochs):
        self.epochs = epochs
        
    def setMinDelta(self, min_delta):
        self.min_delta = min_delta
        
    def setPatience(self, patience):
        self.patience = patience
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        
    def shouldEarlyStop(self, val):
        self.should_early_stop = val
    
    def shouldPlot(self, val):
        self.should_plot_live_error = val
    
    def setReg(self, reg_param, type='l1'):
        self.l = l1 if type == 'l1' else l2
        self.reg_param = reg_param
    
    def setOptimizer(self,val):
        self.optimizer = val
    
    def runLearningCurve(self, steps = 10, should_save_fig = True):
        
        
        print ("\nTrainig curve is about to be produced....\nIt will take a while, plese be patient...\n")
        self.log.info("\nTrainig curve is about to be produced\n")
        
        l = len(self.X_train)
        
        cv_errors, train_errors = [], []
        
        for i in range(1,steps+1):
        
            model = 0
        
            # Split rows for Learning curves
            indexer = int((i/steps)*l)
            X_train = self.X_train[:indexer]
            Y_train = self.Y_train[:indexer]
             
            #creating the structure of the neural network
            model = Sequential()
            model.add(Dense(self.layers[0], input_dim = self.input_dim, activation = self.input_activation_func))
            for ind in range(1,len(self.layers)):
                model.add(Dense(self.layers[ind], activation = self.hidden_activation_func))
            model.add(Dense(1, activation = self.final_activation_func))
             
            # Compile model
            model.compile(loss=self.loss_func, optimizer=self.optimizer)
             
            # Creating Early Stopping function and other callbacks
            call_back_list = []
            early_stopping = EarlyStopping(monitor='loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto') 
            plot_losses = PlotLosses(i)
        
            if self.should_early_stop:
                call_back_list.append(early_stopping)
            if self.should_plot_live_error:
                call_back_list.append(plot_losses) 
             
            # Fit the model
            X_train, Y_train = X_train.values, Y_train.values
            model.fit(X_train, Y_train, validation_data=(self.X_cv, self.Y_cv), epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=2, callbacks=call_back_list)
            plot_losses.closePlot()
            
            # Evaluate the model
            train_scores = model.evaluate(X_train, Y_train, verbose=2)
            cv_scores = model.evaluate(self.X_cv, self.Y_cv, verbose=2)
            
            print ("Step", i, 'Trian_err:', train_scores, 'Cv_err:', cv_scores)
             
            # Add errors to list
            train_errors.append(train_scores)
            cv_errors.append(cv_scores)
             
            print ("---Step %d is done--\n---------------------\n" %(i))
        
        
        # Serialize model to JSON
        save_address = self.directory + "/" + self.name 
        model.save(save_address + ".h5")
        print ("---------------------\nModel is Saved")
        
        # Creating X values for plot
        x_axis = [i for i in range(1,steps+1)]
        
        # Plot
        plt.clf()
        plt.xlabel('Step')
        plt.ylabel('Error - MSE')
        plt.title(self.name+'Learning Curve')
        cv_err, = plt.plot(x_axis, cv_errors, label = 'CV-err')
        train_err, = plt.plot(x_axis, train_errors, label = 'Train-err')
        
        plt.legend(loc = 2,
                   handles=[cv_err, train_err],
                   fontsize = 'x-small')
        plt.grid(True)
        
        if should_save_fig:
            plt.savefig(self.directory + '/TC-' + self.name + '.png')
        if self.should_plot_live_error:
            plt.show()
    
    @timeit
    def runRegularizationParameterAnalysis(self, first_guess = 0.001, final_value = 3, increment = 2, should_save_fig = True):
        
        self.log.info('Regularization analysis is about to be conducted, first guess: %f, final value: %f, increment: %d'%(first_guess, final_value, increment))
        
        # Creating empty list for errors
        cv_errors, train_errors = [], []
        
        X_train, Y_train = self.X_train.values, self.Y_train.values
        
        labels = []
        reg = first_guess
        while reg < final_value:
            
            labels.append(f"{reg:.2E}")
            #creating the structure of the neural network
            model = Sequential()
             
            model.add(Dense(self.layers[0], input_dim = self.input_dim,
                            activation = self.input_activation_func,
                            kernel_regularizer=self.l(reg)))
            for ind in range(1,len(self.layers)):
                model.add(Dense(self.layers[ind], activation = self.hidden_activation_func,
                                kernel_regularizer=self.l(reg)))
            model.add(Dense(1, activation = self.final_activation_func,
                            kernel_regularizer=self.l(reg)))
             
            # Compile model
            model.compile(loss=self.loss_func, optimizer=self.optimizer)
            
            # Creating Early Stopping function and other callbacks
            call_back_list = []
            early_stopping = EarlyStopping(monitor='loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto') 
            plot_losses = PlotLosses(f"{reg:.2E}")
        
            if self.should_early_stop:
                call_back_list.append(early_stopping)
            if self.should_plot_live_error:
                call_back_list.append(plot_losses)
                
             
            # Fit the model
            model.fit(X_train, Y_train, validation_data=(self.X_cv, self.Y_cv), epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=2, callbacks=call_back_list)
            plot_losses = PlotLosses(reg)
             
            # evaluate the model
            train_scores = model.evaluate(X_train, Y_train, verbose=0)
            cv_scores = model.evaluate(self.X_cv, self.Y_cv, verbose=0)
            
            
            self.log.info(f"Regparam: {reg:.4f}, Train Error: {train_scores:.4f}, Cv_err: {cv_scores:.4f}")
            
             
            cv_errors.append(cv_scores)
            train_errors.append(train_scores)
            
            print (f"Train Error: {train_scores:.4f}, Cv_err: {cv_scores:.4f}")
            print (f"---Fitting with {reg:.2E} as regularization parameter is done---")
            
            reg = reg*increment
             
        
        
        x_axis = [(i+1) for i in range(len(cv_errors))]
        
        plt.clf()
        plt.xlabel('Regularization Paremeter')
        plt.ylabel(f'Error - {self.loss_func}')
        plt.title(self.name+'Regularization Analysis')
        cv_err, = plt.plot(x_axis, cv_errors, label = 'CV-err')
        train_err, = plt.plot(x_axis, train_errors, label = 'Train-err')
        
        plt.legend(loc = 2,
                   handles=[cv_err, train_err],
                   fontsize = 'x-small')
        
        plt.xticks(x_axis, labels)
        plt.grid(True)
        
        rprt_df = pd.DataFrame({'Reg': labels,
                                'CV Errors' : cv_errors,
                                'Train Errors' : train_errors})
        
        rprt_df.to_csv(self.directory+ '/RegAnalysis-' + self.name + '.csv' )
        
        
        if should_save_fig:
            plt.savefig(self.directory + '/RA-' + self.name + '.png')
        if self.should_plot_live_error:
            plt.show()
    
    @timeit
    def fitModel(self, drop = 1):
        
        #creating the structure of the neural network
        model = Sequential()
        model.add(Dense(self.layers[0],
                        input_dim = self.input_dim,
                        activation = self.input_activation_func,
                        kernel_regularizer=self.l(self.reg_param)))
        if drop != 1:
            model.add(Dropout(drop)) 
        
        
        for ind in range(1,len(self.layers)):
            model.add(Dense(self.layers[ind],
                            activation = self.hidden_activation_func,
                            kernel_regularizer=self.l(self.reg_param)))
            if drop!= 1:
                model.add(Dropout(drop))
        model.add(Dense(1, activation = self.final_activation_func, kernel_regularizer=self.l(self.reg_param)))
         
        # Compile model
        model.compile(loss=self.loss_func, optimizer=self.optimizer)
         
        # Creating Early Stopping function and other callbacks
        call_back_list = []
        early_stopping = EarlyStopping(monitor='loss', min_delta = self.min_delta, patience=self.patience, verbose=1, mode='auto') 
        plot_losses = PlotLosses(0)
        
        
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        self.log.info(short_model_summary)
    
        if self.should_early_stop:
            call_back_list.append(early_stopping)
        if self.should_plot_live_error:
            call_back_list.append(plot_losses) 
        
        # Fit the model
        X_train, Y_train = self.X_train.values, self.Y_train.values
        model.fit(X_train, Y_train, validation_data=(self.X_cv, self.Y_cv), epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=2, callbacks=call_back_list)
        plot_losses.closePlot(self.directory + "/Losses.png", should_save = True)
        
        # Evaluate the model
        train_scores = model.evaluate(X_train, Y_train, verbose=2)
        cv_scores = model.evaluate(self.X_cv, self.Y_cv, verbose=2)
        test_scores = model.evaluate(self.X_test, self.Y_test, verbose=2)
            
        print ()
        print ('Trian_err:', train_scores, 'Cv_err:', cv_scores, 'Test_err', test_scores)
        self.log.info('Trian_err: %0.5f, CV_err: %0.5f, Test_err: %0.5f' %(train_scores, cv_scores, test_scores))
        
        # Serialize model to JSON
        save_address = self.directory + "/" + self.name 
        model.save(save_address + ".h5")
        print ("---------------------\nModel is Saved")
        
    
    @timeit
    def loadModel(self):
        
        # load json and create model
        address = self.directory + "/" +  self.name
        self.model = load_model(address+ ".h5")
        self.model.compile(loss=self.loss_func, optimizer=self.optimizer)
        
    @timeit
    def getReport(self):
        
        y_pred_train = self.model.predict(self.X_train).reshape(1,-1)[0]
        y_pred_test = self.model.predict(self.X_test).reshape(1,-1)[0]
    
        
        self.evaluate_regression(self.Y_train, y_pred_train, inds = self.dates_train, label='DNN-OnTrain')
        self.evaluate_regression(self.Y_test, y_pred_test, self.dates_test, label='DNN-OnTest')
        self.shap_deep_regression(self.model, self.X_train, self.X_test, list(self.X.columns))
        


        