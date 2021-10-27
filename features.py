import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
pd.set_option('display.max_columns', 500)
%matplotlib inline


# First, read the csv into a df
df=pd.read_csv('L6.csv')

display(df)

plt.scatter(df.x,df.y)





# Build linear regression model with Pipeline, fit and predict
#include_bias=true - it will add a clumn with data 1 or 0
M1=Pipeline([
    ('poly1',PolynomialFeatures(degree=6,include_bias=False)),
    ('lin1',LinearRegression(fit_intercept=True))
    ])


# Define the x values and x_new values here

x=df.x.values.reshape(-1,1)
x_new=np.linspace(-1,1,len(x)).reshape(-1,1)
# Plot the points, also plot prediction

plt.scatter(x,df.y)
ypred=M1.fit(x,df.y).predict(x_new)
plt.plot(x_new,ypred)



# This seems wiggly. What if we use Ridge regression? 
# Side note: In practice, we don't shrink the bias term

M2=Pipeline([
    ('poly2',PolynomialFeatures(degree=6,include_bias=False)),
    ('scal2',StandardScaler()),
    ('ridge2',Ridge(alpha=5.0,fit_intercept=True))
])

plt.scatter(x,df.y)
ypred2=M2.fit(x,df.y).predict(x_new)
plt.plot(x_new,ypred2)


# Aside: what features did we construct (PolynomialFeatures)?
Xc=M1.named_steps['poly1'].fit_transform(x)
display(pd.DataFrame(Xc).head())

# We standardized our features for effective regularization

Xstd=StandardScaler().fit_transform(Xc)
display(pd.DataFrame(Xstd).head())

#took (each value - mean )/standar deviation - thats how we standardize the value

# How do the coefficients look?

print(f"M1 coeff:{M1.named_steps['lin1'].coef_}")
print(f"M2 coeff:{M2.named_steps['ridge2'].coef_} |intercept:{M2.named_steps['ridge2'].intercept_}")





# Ok, let's compare models with various hyperparameters using 5-fold CV
# We'll consider some range of lambda values and get our cv scores
# ("lambda" [regularization coefficient] is called "alpha" in sklearn.Ridge)

lam=np.exp(np.linspace(-4,4,20))
mse=np.zeros(len(lam))

for i in range(len(lam)):
    M2.set_params(ridge2=Ridge(alpha=lam[i]))
    cvsc=cross_val_score(M2,x,df.y,cv=5,scoring='neg_mean_squared_error')
    mse[i]=-cvsc.mean()
    print(f"lam:{lam[i]}|MSE{mse[i]}")
plt.plot(np.log(lam),mse)




# So low let's look at the crossvalidation error for the best setting of lambda 

M2.set_params(ridge2=Ridge(alpha=np.exp(1.5)))
print(-cross_val_score(M2,x,df.y,cv=5,scoring='neg_mean_squared_error').mean())



# Finally, let's consider the same dataset and perform Lasso Regularization

M4=Pipeline([
    ('poly4',PolynomialFeatures(degree=6,include_bias=False)),
    ('scal4',StandardScaler()),
    ('lasso4',Lasso(alpha=0.005,fit_intercept=True))
])

ypred4=M4.fit(x,df.y).predict(x_new)
plt.scatter(x,df.y)
plt.plot(x,ypred)
plt.plot(x,ypred2)
plt.plot(x,ypred4,"red")


# Let's check the coefficients. What do you notice compared to ridge regression? 

M4.named_steps['lasso4'].coef_

# Some coefficient are shrunk to ZERO !

