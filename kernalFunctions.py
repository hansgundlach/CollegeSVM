# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 13:43:36 2016

@author: EliFo
"""

import numpy as np

#assortment of kernal functions
#to be addd to

#norm of a vector function 
def norm(X):
    totalSum = 0
    for n in X:
        totalSum += n**2
    return np.sqrt(totalSum)

#Linear Kernel is just a normal dot product so we can use 
#numpy.dot(x, y) 

#Polynomial Kernel 
#follows the form K(u,v) = (u * v + b)**2
#u and v are the two vectors and b is a constant
def polynomialK(u,v,b):
    return (np.dot(u,v)+b)**2    
    
#Guassian Kernal Funciton 
def gaussianK(v1, v2, sigma):
    return np.exp(-norm(v1-v2, 2)**2/(2.*sigma**2))
    
#computes the gramMatrix given a set of all points included in the data
def compute_gramMatrix(allPoints, kernal):
    b = 5 #constant for the polynomialKernal Funciton
    gramMatrix =  []
    row = []
    for n in allPoints:
        for v in allPoints:
            row.append(kernal(n,v,b))
        gramMatrix.append(row)
    return gramMatrix
        
        