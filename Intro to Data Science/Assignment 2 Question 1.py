# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 21:34:59 2017

@author: Fionn Delahunty
"""

#Opens dataset 
f = open('country-gdp-le.csv',"r")

#Define lists
county = []
gdp = []
le = []
accum_var = 0 

#For each line in the dataset 
for line in f:
    #Slipt the dataset by comma and assign county, gdp & le to three varibles 
    var_c,var_gdp,var_le=line.split(",")
    
    #Remove /n from the le data 
    var_le_2 = var_le.replace("\n","")
    
   #Convert GDP to float and deal with missing value  
    try:
        var_gdp = float(var_gdp)
    except ValueError:
        var_gdp = 0
    #Convert GDP to float and deal with missing value 
    try:
        var_le_2 = float(var_le_2)
    except ValueError: 
        var_le_2 = 0

    
    #Add each value to the list    
    county.append(var_c)
    gdp.append(var_gdp)
    le.append(var_le_2)
    
    #Calculate the ratio and deal with 0 
    try:
       rat = var_le_2/var_gdp
    except ZeroDivisionError:
        rat = 0 
    
    #Store the largest ratio 
    if rat > accum_var: 
        accum_var = rat
        name = var_c
    
        
    
        
    
    

#print county
#print gdp
#print le
print (str(name) + " has the highest life expectancy to GDP per capita ratio of:" + str(accum_var) )





f.close()
