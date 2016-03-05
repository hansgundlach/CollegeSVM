# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 08:28:01 2016

@author: HansG17
"""
#http://prajitr.github.io/quick-haskell-syntax/
#SVM college prediction
#  Eli Fonesca and Hans Gundlach

import numpy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cvxopt
from matplotlib import cm
from cvxopt import matrix, solvers
from itertools import izip
#http://www.tristanfletcher.co.uk/SVM%20Explained.pdf

class SVM:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def findParameters(self):
        
        
        # min 1/2 x^T P x + q^T x
        #Ax = b
        #y's are answer vectors 
        #put in cvxopt 
       # 
        P = cvxopt.matrix(np.outer(self.y,self.y)* self.gramMatrix())
        q = cvxopt.matrix((numpy.ones(len(self.y))).T)
        #G = 
        #h = 
        limits = np.asarray(self.y)
        A = cvxopt.matrix(limits.T)
        #genrates matrix of zzeros
        b = cvxopt.matrix(numpy.zeros(len(self.y)))
        # actually comp
        param = cvxopt.solvers.qp(P,q,A,b);
        return np.ravel(param)
        
    """def norm(X):
    totalSum = 0
    for n in X:
        totalSum += n**2
    return np.sqrt(totalSum)"""
    
    def WB_calculator(self):
        yi = self.y
        important = findParameters()
        firstsum = [0 for x in range(0,len(self.y))]
        for point in important:
            firstsum = map(sum, izip(firstsum,point*important[n]*yi[n]))
            
        #this part calculates bias
            #this is a very naive implementation of bias
            #xstuff is the x_coordinate vector we find this by transpose
            b = 0
        for i in range(0,len(important)):
            b = b+ (yi[i]- np.dot(firstsum,xstuff))
            
        realB = b/len(important)
        answer = (firstsum , realB)
            
            
            
            
    
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
    """def grammatrix(self, allPoints):
        b = 5 #constant for the polynomialKernal Funciton
        gramMatrix =  []
        row = []
        for n in allPoints:
            for v in allPoints:
                row.append(polynomialK(n,v,b))
                gramMatrix.append(row)
                #conver to numpy array for matrix multiplication
        finalAnser = np.asArray(gramMatrix)
        return finalAnser"""
    
    def gramMatrix(self): 
        gramMatrix = []
        data = np.asarray(self.X)
        dataTran = data.T
        print(dataTran)
        for x in dataTran:
            row = []
            #print(row)
            for y in dataTran:
               
                row.append(np.dot(x,y))
                
            gramMatrix.append(row)
            #print(row)
        return gramMatrix
                
                
        
        
       # print(self.X[0])

    
    
    """plot point"""
    def Graph(self):
        #here we actaually graph the functionb 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = self.X[0]
        ys = self.X[1]
        zs = self.X[2]
        
        colors = self.y
        ax.scatter(xs,ys,zs,c=colors)
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
check = np.asarray(a)

print(check.T)
d = SVM(a,[.1,.2,.3,.4,.5,.6,.7])
#print(d.gramMatrix())
print((np.outer(d.y,d.y)*d.gramMatrix())[6])
print((numpy.ones(len(d.y))).T) 
#print(d.findParameters())
print(map(sum, izip([0,1],[2,3])))


#print(colors)
d.Graph() 