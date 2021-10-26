## build linear model



# Data: 1000 North Carolina births. Let's consider weeks (length of the pregnancy) and weight (birth weight in pounds).
bdf = pd.read_csv("ncb.csv")
#display(bdf)
print(bdf.head())


# Important trick: Let's exclude observations with missing week or weight data (NaN values). 
bdf = bdf.dropna(subset=["weeks","weight"])
print(bdf.head())


# What does the joint distribution look like?
sns.jointplot(data=bdf, x='weeks', y='weight')


# Let's build a linear model (w.r.t. b's.. consider a function form of b0x + b1x^2 + b2x^3) on this data and plot the fit 

# create sklearn LinearRegression() object
reg = sklearn.linear_model.LinearRegression()

# create design matrix (i.e. the X matrix)
X = np.c_[bdf.weeks, bdf.weeks**2, bdf.weeks**3]

# using linear regression, fit the data
reg.fit(X, bdf.weight)
print(reg.coef_)



# Ok, now that we have a parameter estimate we can plot a curve of best fit to the data (from our linear regression)

# generate many x points for a smooth curve fit
x_new = np.linspace(min(bdf.weeks), max(bdf.weeks),100)

# create design matrix (i.e. the X matrix)
X_new = np.c_[x_new, x_new**2, x_new**3]

# generate our curve of best fit predictions, given X_new
ypr = reg.predict(X_new)


# plot the dat and the curve of best fit
plt.scatter(bdf.weeks, bdf.weight)
plt.plot(x_new, ypr, color="red")





# Bootstrap the model parameters


# Let's bootstrap the model parameters b1,b2,b3 with 10000 bootstrap samples, given a sample S (dataframe)
def bootstrap_param(S):
    num_iterations = 1000
    sample_size = len(S)
    
    #initialize empty array of bootstrap parameter estimates
    boot_thetas = np.zeros((1000,3))
    
    for i in range(num_iterations):
        # create bootstrap sample
        S_star = S.sample(sample_size, replace=True)
        
        # create design matrix from bootstrap sample
        X = np.c_[S_star.weeks, S_star.weeks**2, S_star.weeks**3]
        reg.fit(X,S_star.weight)
        boot_thetas[i,:] = reg.coef_ 
    return boot_thetas
  
  
  
# We can call the function and see our bootstrapped sets of parameter estimates
boot_thetas = bootstrap_param(bdf)
print(boot_thetas)



# Bootstrap the model Prediction

# Finally, let's bootstrap the model prediction with 10000 bootstrap samples, given a sample S (dataframe) and x values
def bootstrap_pred(S):
    num_iterations = 1000
    sample_size = len(S)

    # Let's create our x data using linspace to get a smooth plot for each bootstrap-predicted set of y values
    x_new = np.linspace(min(S.weeks),max(S.weeks),100)
    X_new = np.c_[x_new,x_new**2,x_new**3]
    
    # Let's also initialize our ypred to an array of 0's
    ypred = np.zeros((1000,X_new.shape[0]))
    
    for i in range(num_iterations):
        #Here we want to store predictions for y from bootstrapped samples. Within the loop, let's get each sample S_star.
        S_star = S.sample(num_iterations,replace=True)
        
        #create design matrix from bootstrapped sample, fit to data
        X = np.c_[S_star.weeks, S_star.weeks**2, S_star.weeks**3]
        reg.fit(X,S_star.weight)
        
        #store predictions for y in ypred array
        ypred[i,:] = reg.predict(X_new)
        print(X_new.shape)
    return ypred
  
  
# Plot 20 of the bootstrapped predictions 
ypr_boot = bootstrap_pred(bdf)
#print(ypr_boot.shape)
for i in range(20):
    plt.plot(x_new, ypr_boot[i,:])
    
    
# Finally, let's compute our upper and lower bounds of our confidence interval for the predictions 
lower = np.quantile(ypr_boot - ypr, 0.025, axis=0) #axis=0 refers to rows
upper = np.quantile(ypr_boot - ypr, 0.975, axis=0)

#plot prediction
plt.plot(x_new, ypr)
#plot confidence interval (from bootstrap)
plt.plot(x_new, ypr - upper, 'r--')
plt.plot(x_new, ypr - lower, 'r--')


    
    
  

