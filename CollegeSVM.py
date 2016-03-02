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