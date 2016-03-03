# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 08:28:01 2016

@author: HansG17
"""

#SVM college prediction
#  Eli Fonesca and Hans Gundlach

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cvxopt
#http://www.tristanfletcher.co.uk/SVM%20Explained.pdf

class SVM:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    
    #def findParameters(self):
        
    
    
    def printer(self):
        print(self.X[0])
    
    
    """plot point"""
    def Graph(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = self.X[0]
        ys = self.X[1]
        zs = self.X[2]
        
        colors = self.y
        ax.scatter(xs,ys,zs,c=colors,marker=m)
        ax.set_xlabel("SAT score")
        ax.set_ylabel("GPA")
        ax.set_zlabel("AP scores")
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = R**2
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
        plt.show()
        
        
        
a = [[1,1,1,1,-1,-3,-5],[1,2,2,1,-6,-5,-2],[1,3,3,1,.5,3,50]]
d = SVM(a,[.1,.2,.3,.4,.5,.6,.7])

#print(colors)
d.Graph() 

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
#and a kernal funciton to apply to all combinations of point vectors
def compute_gramMatrix(allPoints, kernal):
    b = 5 #constant for the polynomialKernal Funciton
    gramMatrix =  []
    row = []
    for n in allPoints:
        for v in allPoints:
            row.append(kernal(u,v,b))
        gramMatrix.append(row)
    return gramMatrix