# Developing-a-Deep-Neural-Network-based-Framework-for-Complex-Correlations-in-Engineering-Metrics
///-----------------------------------///
/                        Vahid Asghari  /
/                                       /
/---------------------------------------/

Audience: Anyone with beginner level of python programming skills, and looking for a simple code for deep neural netwroks
to cunduct regression analysis

This code was originally used in the paper "Developing a Deep Neural Network based Framework for Complex Correlations in Engineering Metrics", by Vahid Asghari, Andy Y.F. Leung and Mark S.C. Hsu
If you are using this code, please cite this paper and following libraries (KERAS, scikit-learn, and tensorflow, SHAP)


For using this code, please follow these steps:

1- After preprocessing your data (such as feature scaling (except the output variable), removing outliers,...), save the dataset as a "csv" file
   SoilWorld is the dataset used in the paper and can be used as a sample.
   The first column of the file should be the index or date and the last column of the dataset must be the output variable

3- Open 'RunMe.py' using any IDE and Fill in the necessary settings:
	3-1: Layers should be in brackets like [12, 6] . This sets the number of the nodes in each hidden layer
	     Example: [12, 6] creates two hidden layers, first one with 12 nodes and second one with 6 nodes
	3-2: Loss functions: https://keras.io/losses/
	3-3: Activation functions: https://keras.io/activations/
	3-4: Opimizers: https://keras.io/optimizers/
	3-5: shouldPlot: Whether plot the errors during training
	3-6: shouldEarlyStop: Whether stops the training by reduction in the error or by reaching the epoch number
	3-7: setReg( regularization parameter, type of regilarization l1 or l2)
	
	Other settings are self explanatory and can be found in the paper

4- runLearningCurve(steps=10):
	conducts learning curve analysis in specified steps

5- runRegularizationParameterAnalysis(first_guess = 0.001, final_value = 0.02, increment=3):
	conducts analysis for finding regularization parameter

6- fitModel(drop = 1):
	train the DNN model
	drop is the dropout value, if it is set to 1, dropout is not considered

7- loadModel():
	loads the DNN model

8- getReport()
	prints the results of the model in the "Report.log", in the folder named by the project name determined by the user
	Draws actual vs predicted values, actual and predicted values versus sorted samples
	Draws feature importances based on SHAP values
