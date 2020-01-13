from DNNRegression import *

def exec():
    '''
    Followings should be filled by the practitioner according to the paper and readme.txt file
    '''
    
    file_name = "SoilWorld"
    project_name = 'SoilWorld'
    
    myRegressor = Regressor(file_name, name = project_name, should_shuffle=True, split_size = 0.4)
    
    myRegressor.setLayers([15, 10, 5])
    myRegressor.setLossFunction('MSE')
    myRegressor.setEpochs(50)
    
    myRegressor.setInputActivationFunction('sigmoid')
    myRegressor.setHiddenActivationFunction('relu')
    myRegressor.setFinalActivationFunction('linear')
    
    myRegressor.setOptimizer('Adam')
    
    myRegressor.shouldPlot(True)
    myRegressor.shouldEarlyStop(True)
    
    myRegressor.setBatchSize(128)
    myRegressor.setPatience(50)
    myRegressor.setMinDelta(1)
    myRegressor.setReg(0.000002, 'l1')
    
#     myRegressor.runLearningCurve(steps=10)
    myRegressor.runRegularizationParameterAnalysis(first_guess = 0.001, final_value = 0.02, increment=3)

    myRegressor.fitModel(drop = 1)
    myRegressor.loadModel()
    myRegressor.getReport()
    
    
if __name__ == "__main__":
    exec()