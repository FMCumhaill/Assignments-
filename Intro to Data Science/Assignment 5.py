# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 16:08:47 2017

@author: Fionn Delahunty
"""

import numpy as np




h =300000
k = 4

jin = 0
for i in xrange(h):
    #M value 
    m = 1000
    
    #Range of data 
    tmp = [i*np.ones(m,int) for i in range(1,4)]
    c = np.concatenate(tmp)
    
    
    hit = np.random.choice(c, size=k, replace=False)
    #Number of large K 
    values =[1,2,3]
    cup = np.in1d(values,hit)
    if np.all(cup) == True:
        jin = jin + 1 
    
if jin >= h/2:
      print "good"
      

print "h", h 
print "jin", jin

    
    