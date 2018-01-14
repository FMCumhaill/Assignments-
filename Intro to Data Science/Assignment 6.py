# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:51:41 2017

@author: Fionn Delahunty
"""

import numpy as np
import matplotlib.pyplot as plt
#import sklearn as sk
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import pandas as pd
import matplotlib as mpl
#Wanted to try using this as well, you can view docs http://seaborn.pydata.org 
import seaborn as sns

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
        age.append(float(d))
    
    #I assume I could have thought of a better/cleaner way to do this, also removes the BMI value zero 
    try:
          bmivalue=702*float(d)/float(c)**2
          if bmivalue==0:
              year = year[:-1]
          else:
              bmi.append(bmivalue)
          
    except ZeroDivisionError:
           year = year[:-1]
           
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
    
  
reg = np.zeros(shape=(5,2))    

reg = np.column_stack((year,bmi))

b1, b0 = compute_linear_reg(reg)
    
reg_res = residuals_linear_fit(reg,b1,b0)

reg2 = np.zeros(shape=(5,2))    

reg2 = np.column_stack((year,reg_res))
b1, b0 = compute_linear_reg(reg2)


plot_linear_fit(reg2, colors[0], b1, b0, 'Residuals plot of BMI/Years verse Years ',
              'Years', 'BMI')

#
 #Attempt using Seaborn functions 
 

create_dataframe =  {'Year' : year, 'BMI' : bmi}
Panda_Dataframe= pd.DataFrame(create_dataframe)


sns.set(color_codes=True)

sns.residplot(x="Year", y="BMI", data=Panda_Dataframe, );


