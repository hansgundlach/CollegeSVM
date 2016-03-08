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
        
    def findParameters(self,X,y):
        
        # min 1/2 x^T P x + q^T x
        #Ax = b
        #y's are answer vectors 
        #put in cvxopt 
       # 
       
        """P = cvxopt.matrix(np.outer(self.y,self.y)* self.gramMatrix())
        q = cvxopt.matrix((numpy.ones(len(self.y))).T)
        #G = 
        #h = 
        limits = np.asarray(self.y)
        A = cvxopt.matrix(limits.T)
       
        
        #genrates matrix of zzeros
        b = cvxopt.matrix(numpy.zeros(len(self.y)))
        # actually comp
        param = cvxopt.solvers.qp(P,q,G,h,A,b);"""
        n_samples, n_features = X.shape
        K = self.gramMatrix(X)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i \leq 0
        # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * 1)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        param = cvxopt.solvers.qp(P, q, G, h, A, b)
        array = param['x']
        return array
        
    """def norm(X):
        totalSum = 0
               # for n in X:
                #totalSum += n**2
    return np.sqrt(totalSum)"""
    
    def WB_calculator(self,X,y):
        #calculates w vector
        yi = self.y
        X = np.asarray(X)
        y = np.asarray(y)
        important = self.findParameters(X,y)
        print("this is important")
        print(important)
        firstsum = [0 for x in range(0,len(y))]
        for point in range(0,len(important)):
            liste = X[point]*important[point]*yi[point]
            firstsum = [x + y for x, y in zip(firstsum,liste)]
            

            
        #this part calculates bias
            #this is a very naive implementation of bias
            #xstuff is the x_coordinate vector we find this by transpose
            b = 0
        for i in range(0,len(important)):
            b = b+ (yi[i]- np.dot(firstsum,X[i]))
            
        avgB = b/len(important)
        answer = (firstsum , avgB)
        print("firstsum")
        print(firstsum)
        return answer
            
            
            
            
    
    #Linear Kernel is just a normal dot product so we can use 
#numpy.dot(x, y) 

#Polynomial Kernel 
#follows the form K(u,v) = (u * v + b)**2
#u and v are the two vectors and b is a constant
    def polynomialK(self,u,v,b):
        return (np.dot(u,v)+b)**2    
    
#Guassian Kernal Funciton 
    def gaussianK(self,v1, v2, sigma):
        return np.exp(-norm(v1-v2, 2)**2/(2.*sigma**2))
    
#computes the gramMatrix given a set of all points included in the data
    
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
        Harvard = self.WB_calculator(X,y)
        if(np.dot(Harvard[0],point)+Harvard[1] >0):
            print("You got in")
        elif(np.dot(Harvard[0],point)+Harvard[1]<0):
            print("Study")
                
    # plots acceptance cutoff
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
        ax.set_xlabel("SAT score")
        ax.set_ylabel("GPA")
        ax.set_zlabel("AP scores")
        #this changes orientation and look of surface
        ax.view_init(azim = 180+40,elev = 22)
        X = np.arange(-2, 2, 0.25)
        Y = np.arange(-2, 2, 0.25)
        X, Y = np.meshgrid(X, Y)
        
        Z = ((weights[0]*X + weights[1]*Y - c)/(weights[2]))
        #R = np.sqrt(X**2 + Y**2)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
        plt.show()
        
        
#lets make this a list of points       
#a = [[1,1,1,1,-1,-3,-5],[1,2,2,1,-6,-5,-2],[1,3,3,1,.5,3,50]]
a = [[-.1,-.1,-.1],[-.2,-.2,-.2],[.15,.15,.15],[.9,.9,.9],[.95,.95,.95]]
check = np.asarray(a)
b = [.01,.01,.01,1,1]
bigger =np.asarray(b)
d = SVM(a,b)
print(d.gramMatrix(check)[0])
#print((np.outer(d.y,d.y)*d.gramMatrix())[6])
#print((numpy.ones(len(d.y))).T) 
print("parameters ya")
print(d.findParameters(check,bigger))
print(d.WB_calculator(check,bigger))
d.Graph(check,bigger) 
d.determineAcceptance([.01,.01,.01],check,bigger)
#print(map(sum, izip([0,1],[2,3])))


#print(colors)
#d.Graph(check.T,bigger) 