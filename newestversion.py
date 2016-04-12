# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 08:28:01 2016

@author: HansG17
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from matplotlib import cm

#http://www.tristanfletcher.co.uk/SVM%20Explained.pdf

class SVM:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def findParameters(self,X,y):
        # min 1/2 x^T P x + q^T x
        #Ax = b
        #y's are if the student is accepted -1 for being not accpeted 1 for being accpeted 
        #put in cvxopt 
        n_samples, n_features = X.shape
        K = self.gramMatrix(X)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        Gtry = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        htry = cvxopt.matrix(np.zeros(n_samples))
        

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        param = cvxopt.solvers.qp(P, q, Gtry, htry, A, b)
        array = param['x']
        return array
    
    
    def WB_calculator(self,X,y):
        #calculates w vector
        yi = self.y
        X = np.asarray(X)
        y = np.asarray(y)
        important = self.findParameters(X,y)
        print("these are parameters")
        print(important)
        firstsum = [0 for x in range(0,len(y))]
        for point in range(0,len(important)):
            liste = X[point]*important[point]*yi[point]
            firstsum = [x + y for x, y in zip(firstsum,liste)]
            

            
        #this part calculates bias
            #this is a very naive implementation of bias
            #xstuff is the x_coordinate vector we find this by transpose
            bunsen = 0
        for i in range(0,len(important)):
            bunsen = bunsen+ (yi[i]- np.dot(firstsum,X[i]))
            
        avgB = bunsen/len(important)
        answer = (firstsum , avgB)
        print("w vector")
        print(firstsum)
        return answer
            
            
     # kernal is a type of dot product#innerproduct      
    def polynomialK(self,u,v,b):
        return (np.dot(u,v)+b)**2    
    

    def gaussianK(self,v1, v2, sigma):
        return np.exp(-norm(v1-v2, 2)**2/(2.*sigma**2))
    
#computes the gramMatrix given a set of all points included in the data
#this is basicly a matrix of dot prodducts
    
    def gramMatrix(self,X): 
        gramMatrix = []
        data = np.asarray(self.X)
        dataTran = data
        #print(dataTran)
        for x in dataTran:
            row = []
            #print(row)
            for y in dataTran:
               
                row.append(np.dot(x,y))
                
            gramMatrix.append(row)
            #print(row)
        return gramMatrix
    def determineAcceptance(self,point,X,y):
        # I'm not sure if this is the proper bounding lets checl
        cutoff = self.WB_calculator(X,y)
        if(np.dot(cutoff[0],point)+cutoff[1] >0):
            print("You got in")
        elif(np.dot(cutoff[0],point)+cutoff[1]<0):
            print("Study")
                
    # plots  plane and points
    def Graph(self,X,y):
        important_stuff = self.WB_calculator(X,y)
        weights = important_stuff[0] 
        c = important_stuff[1]
        #here we actaually graph the functionb 
        graphable = X.T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = graphable[0]
        ys = graphable[1]
        zs = graphable[2]
        
        colors = self.y
        ax.scatter(xs,ys,zs,c=colors)
        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.set_zlabel("C")
        #this changes orientation and look of surface
        ax.view_init(azim = 180+160,elev = 0)
        X = np.arange(-1.2, 1.2, 0.25)
        Y = np.arange(-1.2, 1.2, 0.25)
        X, Y = np.meshgrid(X, Y)
        
        Z = ((-weights[0]*X + -weights[1]*Y - c)/(weights[2]))
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
        plt.show()
        
#list of points to test
a = [[.1,.1,.1],[.2,.2,.2],[.15,.15,.15],[.9,.9,.9],[.95,.95,.95]]
check = np.asarray(a)
#acceptance list
b = [-1.0,-1,-1,1,1]
bigger =np.asarray(b)
d = SVM(a,b)
print(d.gramMatrix(check)[0])
print("parameters ya")
print(d.findParameters(check,bigger))
print(d.WB_calculator(check,bigger))
d.Graph(check,bigger)
#determines acceptance for prospective applicant
print(d.determineAcceptance([.01,.01,.01],check,bigger))
