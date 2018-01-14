f# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 19:02:38 2017

@author: Fionn Delahunty
"""
###Imports####


import numpy as np
import matplotlib.pyplot as plt
#import sklearn as sk
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import pandas as pd
import matplotlib as mpl
import statsmodels.api as sm
#Wanted to try using this as well, you can view docs http://seaborn.pydata.org 
import seaborn as sns



#####Data cleaning and importing #####


#open text file 
text_file = open("problem-6-1.txt", "r")
text_file = text_file.readlines()

#Creates lists 
year =[]
age = []
height =[]
weight = []
bmi =[]
colors = ['red', 'greenyellow', 'blue']

#Extract the data 
for line in text_file:
    a,b,c,d =line.split()
    
    #Removes the two digits "or " 
    if len(b) > 3:
        b = "*"
    
    year.append(int(a))
    
    #I assume I could have thought of a better/cleaner way to do this
    try: 
        age.append(float(b))
    except ValueError:
        b = 0 
        age.append(float(b))
        float(b)
    
    try: 
        height.append(float(c))
    except ValueError:
        c = 0
        height.append(float(c))
        float(c)
    
    
    try: 
         weight.append(float(d))
    except ValueError:
        d = 0
        weight.append(float(d))
    
    #I assume I could have thought of a better/cleaner way to do this, also removes the BMI value zero 
    try:
          bmivalue=703*(float(d)/float(c)**2)
          if bmivalue==0:
              year = year[:-1]
          else:
              bmi.append(bmivalue)
          
    except ZeroDivisionError:
           year = year[:-1]
           
####Functions#####
          
#Function taken from Alexander Schliep work with premission 
def compute_linear_reg(data):
    """Plot a linear fit to 2D data including lines indicating
       residuals. Data is an array of tuples.

       Return slope, intercept of the linear fit.
    """
    lr = LinearRegression()
    # linear regression wants column vectors.
    # Not necessary in older version of scikit?
    lrX = data[:,0].reshape(-1, 1)
    lrY = data[:,1].reshape(-1, 1)
    lr.fit(lrX, lrY)
    b0 = lr.intercept_[0]
    b1 = ((lr.coef_)[0])[0]
    return b1, b0

#Function taken from Alexander Schliep work with premission
def plot_linear_fit(data, color, slope, intercept, title, xlabel, ylabel):
    """Plot a linear fit to 2D data including lines indicating
       residuals. Data is an array of tuples, color a color name, slope and
       intercept floats parameters for the linear function.
    """
    fit = lambda x: slope*x + intercept
    # For legend
    if intercept > 0:
        fitString = "y = %2.2f * x + %2.2f" % (slope, intercept)
    else:
        fitString = "y = %2.2f * x - %2.2f" % (slope, abs(intercept))
        
    # Draw linear fit
    # compute x range
    x_min = min(data[:,0])
    x_max = max(data[:,0])
    # Extend so that residuals do not dangle in mid air
    x_min -= abs(x_min)*0.01
    x_max += abs(x_max)*0.01
    plt.plot([x_min, x_max],[fit(x_min),fit(x_max)], c='black', label=fitString)

    # Plot residuals
    for (x,y) in data:
        plt.plot([x,x],[fit(x),y], c='grey', linewidth=0.5)
    # Plot data
    plt.scatter(data[:,0], data[:,1], c=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()  
   
#Function taken from Alexander Schliep work with premission
def residuals_linear_fit(data, slope, intercept):
    """Return numpy array with residuals. Data is an array of tuples"""
    fit = lambda x: slope*x + intercept
    residuals = []
    for (x,y) in data:
        residuals.append(y - fit(x))
    return residuals
    #return data[:,1] - ((data[:,0] * slope) + intercept)
    
#Function taken from Alexander Schliep work with premission    
def boxplots(data_sets, data_names, title, xlabel, ylabel):
    """Plot multiple boxes in one figure. data_sets is an list of 1D arrays,
       data_names a list of identifiers to use as labels.""" 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.boxplot(data_sets)
    #plt.xticks(range(1,len(data_sets)+1),data_names)
    plt.show()
    
#Function taken from Alexander Schliep work with premission    
def qqplots(data_sets, data_names, title, xlabel, ylabel):
    """Plot multiple qq-plots in one figure. data_sets is an list of 1D arrays,
       data_names a list of identifiers to use as labels.""" 
    n = len(data_sets)
    for i in range(n):
        plt.subplot(1, n, i+1, adjustable='box-forced')
        #plt.axis('equal')
        plt.title(data_names[i])
        stats.probplot(data_sets[i], dist='norm', plot=plt)
    plt.show()

def histogram(data_set, title, xlabel, ylabel):
    np.histogram(data_set)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
        
### Data analysis ###

def EDA_residuals(data1,data2):
    #Create numpy array with bmi and year  
    datapoints = np.column_stack((data1,data2))
    
    
    #Calacuate slope and intercept
    b1, b0 = compute_linear_reg(datapoints)

    #Cal residuals 
    datapoints = residuals_linear_fit(datapoints,b1,b0)
    
    plt.hist(datapoints)
    

    #Plot boxplot of BMI
    boxplots(datapoints,"bmi","Residuals","BMI","BMI")


    np.histogram(datapoints,year)
    plt.show()
    

    #Using Seaborn libary 
    #Create dataframe 
    create_dataframe =  {'Year' : data1, 'BMI' : data2}
    Panda_Dataframe= pd.DataFrame(create_dataframe)

    #Create stater plot + histrogram 
    sns.set(color_codes=True)
    sns.residplot(x="Year", y="BMI", data=Panda_Dataframe);

    
    plt.show()
    
    def qqplot(x, y, **kwargs):
        _, xr = stats.probplot(x, fit=False)
        _, yr = stats.probplot(y, fit=False)
        plt.scatter(xr, yr, **kwargs)
    


    stats.probplot(bmi, dist="norm", plot=plt)
    plt.show() 

    
    
def EDA_normal(data1,data2):
    #Create numpy array with bmi and year  
    datapoints = np.column_stack((data2,data1))

    #Calacuate slope and intercept
    b1, b0 = compute_linear_reg(datapoints)



    #Plot linear fit 
    plot_linear_fit(datapoints, colors[0], b1, b0, 'Regression plot of BMI verse Years ',
              'Year', 'BMI')
    plt.show()

    #Plot boxplot of BMI
    boxplots(bmi,"bmi","BMI,Year","BMI","BMI")

    plt.hist(bmi)
    
    #print results 
    results = sm.OLS(year[5:],bmi[5:]).fit()
    print(results.summary())

    

    #Using Seaborn libary 
    #Create dataframe 
    create_dataframe =  {'Year' : data2, 'BMI' : data1}
    Panda_Dataframe= pd.DataFrame(create_dataframe)

    #Create stater plot + histrogram 
    sns.set(color_codes=True)
    sns.lmplot(x="Year", y="BMI", data=Panda_Dataframe,fit_reg=False );

    sns.jointplot(x="Year", y="BMI", data=Panda_Dataframe);
    plt.show()



    def qqplot(x, y, **kwargs):
        _, xr = stats.probplot(x, fit=False)
        _, yr = stats.probplot(y, fit=False)
        plt.scatter(xr, yr, **kwargs)
    


    stats.probplot(bmi, dist="norm", plot=plt)
    plt.show() 

    sns.set(color_codes=True)
    sns.residplot(x="Year", y="BMI", data=Panda_Dataframe, );
    plt.show()
    
EDA_normal(bmi,year)
print "Residuals are now shown......"

EDA_residuals(year,bmi)




#






