# Kaggle competition for Santander Customer Transaction Prediction
Can you identify who will make a transaction?:
Website: https://www.kaggle.com/c/santander-customer-transaction-prediction

The datasets of this competition are as clean as they can be:
* 200k rows in training set
* 200k rows in testing set
* 1XX features
* XX categorical
* XX numerical
* The distribution of the numerical features is normal, not much outliers
* 1 binary target
* Evaluation: ROC

MY APPROACH:
Build X pipelines:
* Pipeline 1: Logistic regression with regularization with all the features (*)
* Pipeline 2: Naive bayes with all the features (*)
* Pipeline 3: Random forest with all the features (*)
* Pipeline 4: XGboost with all the features (*)
* Pipeline 5: Neural Network with all the features (*)
* Pipeline 6: SVM (*)
* Feature selection: use results from Pipeline 1, Pipeline 3, and Pipeline 4, and (maybe) PCA to find the 'best' features
* Pipeline 7: Logistic regression with the best features
* Pipeline 8: Naive bayes with the best features
* Pipeline 9: Random forest with the best features
* Pipeline 10: XGBoost with the best features
* Pipeline 11: Neural Network with the best features
* Pipeline 12: SVM with the best features
* Ensamble all the previous methods using XGboost (i.e. train XGboost with the outputs of the pipelines and the best features)


