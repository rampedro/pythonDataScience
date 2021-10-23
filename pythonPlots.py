import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as so
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("old_faithful.csv")
df.head()
print("numebr of samples {}".format(len(df)))


# Joint distribution plot with estimated density
# KDE kernal desity estimation ( type of plot, can also be histogram and etc.)
sns.jointplot(x="eruptions", y="waiting", data=df, kind="kde", fill=True,ratio=1)
sns.jointplot(x="eruptions", y="waiting", data=df, kind="hist", fill=True,ratio=1)


#Joint distribution rug plot
sns.displot(df, x="eruptions", y="waiting",kind="kde", rug=True)
sns.displot(df, x="eruptions", y="waiting",kind="hist", rug=True)
# if fill is false we get RUG PLOT # shows the datapoint on x axis


#marginal distribution plot for Y
sns.kdeplot(data=df,y="waiting")
#sns.histplot(data=df,y="waiting")
print(df.waiting.mean())
plt.axhline(df.waiting.mean(),color="red")


#conditional distribution plot for Y | X in [2,2.1]
fig, ax = plt.subplots()
#ax.set_xlim(0,0.08)
#ax.set_ylim(40,100)
df2 = df.loc[df["eruptions"].between(2,2.1,inclusive=True)]
sns.kdeplot(data=df2,y="waiting",color="black")
print("mean: ", df2.waiting.mean())
plt.axhline(df2.waiting.mean(),color="red")
plt.show()



#conditional distribution plot for Y | X in [2,2.1]
fig, ax = plt.subplots()
#ax.set_xlim(0,0.08)
#ax.set_ylim(40,100)
df2 = df.loc[df["eruptions"].between(2,2.1,inclusive=True)]
#df3 = df.loc[df.eruptions >=2]
#df4 = df3.loc[df3.eruptions <=2.1]
#df4 == df2 
sns.kdeplot(data=df2,y="waiting",color="black")
print("mean: ", df2.waiting.mean())
plt.axhline(df2.waiting.mean(),color="red")
plt.show()


#conditional distribution plot for Y | X in [4.4,4.5]
fig, ax = plt.subplots()
#ax.set_xlim(0,0.09)
#ax.set_ylim(40,100)
df3 = df.loc[df["eruptions"].between(4.4,4.5,inclusive=True)]
sns.kdeplot(data=df3,y="waiting",color="black")
print("mean: ", df3.waiting.mean())
plt.axhline(df3.waiting.mean(),color="orange")
