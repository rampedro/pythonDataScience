#Logistic Regression
#● Sigmoid Function
#● Regularization
#● Classifier Evaluation
##○ Probabilistic Evaluation
##○ Label Evaluation
##○ Ranking Evaluation
#● Class Imbalances
#● Plots: Confusion, ROC, AUC
#● Measures (Precision, Recall, F1, …)
#● Multi-class Classification


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# This dataset describes grains of rice as on 7 input features, with 2 class outputs (Cammeo or Osmancik)
rice = pd.read_csv("Rice_Osmancik_Cammeo_Dataset.csv")
rice.head()

# we have 7 feautres 


# Is it class-balanced or unbalanced?
# not it is not balanced 

rice.CLASS.value_counts()



# Split data, train logistic regression

X = rice.iloc[:,0:7].values
y = rice.CLASS.values

# random_state 0  picks the random selection consistent.

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.5, random_state = 0)





# Fit model, print coefficients
#penalty none means no regularization

ricelr = LogisticRegression(penalty="none").fit(Xtrain,ytrain)
print(f"Intercept b0: {ricelr.intercept_} \n Parameters/coeficient {ricelr.coef_}")

# first parametes is the b1 and second is b2



# Predict manually ( we are not using .predict instead calculate the y = Xb + itercept)
# we predict through sigmoid 
sigmoid = lambda x: 1/(1+np.exp(-x))
# we can use sigmoid(somevalue)

y_pred_manual = sigmoid(np.dot(Xtest,ricelr.coef_.T)+ricelr.intercept_)
print(y_pred_manual)

# each prediciton is for a single row. 



# Predict with sklearn. Note: probabilities of class 0 (first col), class 1 (2nd col)

ricelr.predict_proba(Xtest)
# first numebr is the 1-prediciton for class 1




# Get label predictions

# The threshhold by defult is 0.5     greater than 0.5 is class 1 . 
# class one here is the OSmancik


y_pred = ricelr.predict(Xtest)
print(y_pred)




## Evaluation of Classifiers


def compute_performance(yhat, y, classes):
    
    tp = sum(np.logical_and(yhat == classes[1], y == classes[1]))
    tn = sum(np.logical_and(yhat == classes[0], y == classes[0]))
    fp = sum(np.logical_and(yhat == classes[1], y == classes[0]))
    fn = sum(np.logical_and(yhat == classes[0], y == classes[1]))

    print(f"tp: {tp} tn: {tn} fp: {fp} fn: {fn}")
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    # "Of the ones I labeled +, how many are actually +?"
    precision = tp / (tp + fp)
    
    # "Of all the + in the data, how many do I correctly label?"
    recall = tp / (tp + fn)    
    sensitivity = recall
    
    # "Of all the - in the data, how many do I correctly label?"
    specificity = tn / (fp + tn)
        
    print("Accuracy:",round(acc,3),"Recall:",round(recall,3),"Precision:",round(precision,3),
          "Sensitivity:",round(sensitivity,3),"Specificity:",round(specificity,3))
    
    
    
print(compute_performance(y_pred,ytest,ricelr.classes_))





# Now let's experiment by adjusting the decision threshold

y_test_prob = ricelr.predict_proba(Xtest)
#print(y_test_prib)
#y_test_prob = y_test_prob[:,1] > 0.1  # by default is 0.5
yhat = ricelr.classes_[(y_test_prob[:,1] > 0.1 ).astype('int')]
print(yhat)

print(compute_performance(yhat,ytest,ricelr.classes_))

#print(y_test_prob)




# ROC using sklearns ROC curve. 
fpr, tpr, _ = roc_curve(ytest, y_test_prob[:,1], pos_label="Osmancik")
ax=sns.lineplot(fpr,tpr)

# AUROC

auc(fpt,tpr)

# it is a very good classifier ! it is close to 1 






##Multiclass Logistic Regression

# Read data
iris = pd.read_csv("iris.csv")
iris.head()


# Check out class distribution

print(iris.Species.value_counts())

# Create y and X. Not going to split these data for this demonstration.

X = iris.iloc[:,0:4].values
y = iris.Species.values



# Fit the data using Sklearn's Logistic Regression

irislr = LogisticRegression(penalty="none",solver="newton-cg").fit(X,y)
print(f"Intercept b0: {ricelr.intercept_} \n Parameters/coeficient {ricelr.coef_}")

# parameter values are way larger than the values 11 >> 5,1

#irislr2 = LogisticRegression(penalty="l2",solver="newton-cg").fit(X,y)
#print(f"Intercept b0: {ricelr2.intercept_} \n Parameters/coeficient {ricelr2.coef_}")



# Get the predictions

yhat = irislr.predict(X)
#print(yhat)

# Use score method to get accuracy of model
# Cannot use this to check the accuracy since the data are not numerical for the labels
# since we using multi class regresssion now.
#score = irislr.score(yhat, y)
#print(score)



#create a confusion matrix

conf = confusion_matrix(yhat,y)




# plot the confusion matrix

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(conf).plot()



