# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 20:37:08 2017

@author: Fionn Delahunty
"""

import random

def roll(k):
    # Roll a fair die k times
    faces = (1,2,3,4,5,6)
    result = [random.choice(faces) for x in xrange(k)]
    return tuple(result) # Tuple needed as key in dictionary
k = 3 # 3 rolls of a fair die
f = {} # Dictionary to count frequency of each roll
r = 1000 # Nr of repetitions
theList = [] 
loopcounter  = 0 #List to count number of loops
loopList = [] #Store repeated values 
firstRollList =[] #Store the first roll of k


for i in xrange(r):
    #Count the number of loops
    loopcounter = loopcounter + 1 
    theRoll = roll(k)
    
    #on the first loop, assign the first roll to new var 
    if loopcounter == 1:
        firstRoll = theRoll
    
    #If var = the same as first roll append to a list 
    if theRoll == firstRoll:
        firstRollList.append(loopcounter)
    
    #if var = in the the list. Count or else add to list     
    if theRoll in theList:
        loopList.append(loopcounter)
    else: 
        theList.append(theRoll)
    
    if theRoll in f:
        f[theRoll] += 1
    else:
        f[theRoll] = 1

firstRollList.remove(1)

print "The K value was:", k            
print "Observed %d different outcomes in %d repetitions" % (len(f),r)
print "Min frequency %f, max %f" % (min(f.values()), max(f.values()))
print "It took", min(firstRollList) ,  " number of loops until the first roll was repeated" 
print "The first repeated outcomme was seen on loop:", min(loopList)
