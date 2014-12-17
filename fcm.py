#!/usr/bin/env python
import numpy as np
import sys
import random
import math
from operator import itemgetter, attrgetter

class fuzzy_cmeans:
    def __init__(self, centroids_number, centroids_size, q, tol):
        self.centroids_number = centroids_number
        self.centroids_size = centroids_size
        self.q = q
        self.tol = tol
        self.centroids = np.array(self.setupCentroids(), ndmin = 2)
      
    def loadFeatures(self, filename):
        features = []
        f = open(filename, 'r')
        lines = f.readlines() 
        f.close()
        for line in lines:
            s = line.split(',')
            f = []
            for v in s:
                f.append(float(v))
            features.append(f)
        self.features = np.array(features, ndmin = 2)
        self.features_number = len(features)
        print "FN", self.features_number
        Ui = []
        for i in range(0, self.centroids_number):
            Uj = []
            for j in range(0, self.features_number):
                Uj.append(0.0)
            Ui.append(Uj)
        self.U = np.array(Ui, ndmin = 2)
        self.U_tick = np.array(Ui, ndmin = 2)
 
    def setupCentroids(self):
        q = []
        for i in range(0, self.centroids_number):
            c = []
            for j in range(0, self.centroids_size):
                r = random.random()
                c.append(r)
            q.append(c)
        return q

    def getK(self, j):
        # step 2
        ksum = 0
        inv_q = float(1) / (self.q - 1)
        for k in range(0, self.centroids_number):
            a = self.features[j]
            b = self.centroids[k]
            dist = float(1) / np.linalg.norm(a - b)
            dist = math.pow(dist, inv_q)
            ksum += dist
        return ksum

    def calcError(self, i, j):
        # step 2
        a = self.U[i][j]
        b = self.U_tick[i][j] 
        dist = np.linalg.norm(a - b)
        return dist

    def computeMembership(self):
        # step 2
        inv_q = float(1) / (self.q - 1) 
        max_ij = None
        for i in range(0, self.centroids_number):
            for j in range(0, self.features_number):
                a = self.features[j]
                b = self.centroids[i]
                #print "a b", a, b
                dist = float(1) / np.linalg.norm(a - b)
                dist = math.pow(dist, inv_q)
                ksum = self.getK(j)
                #print "dist ksim", dist, ksum
                self.U[i][j] = dist / ksum

                e = self.calcError(i, j)
                if max_ij == None:
                    max_ij = e
                elif e < max_ij:
                    max_ij = e

                self.U_tick[i][j] = self.U[i][j]
                #print "u_ij", self.U[i][j]
        return max_ij 

    def computeTop(self, i):
        # step 3
        c = []
        for j in range(0, self.centroids_size):
            c.append(0.0)
        centroid_top = np.array(c, ndmin = 2)
        for j in range(0, self.features_number):
             u = np.array(math.pow(self.U[i][j], self.q))
             centroid_top += u * self.features[j]
        #print "c", centroid_top 
        return centroid_top

    def computeBottom(self, i):
        # step 3
        c = []
        for j in range(0, self.centroids_size):
            c.append(0.0)
        centroid_bottom = np.array(c, ndmin = 2)
        for j in range(0, self.features_number):
            centroid_bottom += np.array(math.pow(self.U[i][j], self.q))
        #print "c`", centroid_bottom
        return centroid_bottom
             

    def computeCentroids(self):
        # step 3
        for i in range(0, self.centroids_number):
            a = self.computeTop(i) 
            b = self.computeBottom(i)
            #print "a/b", a / b
            self.centroids[i] = a / b

    def train(self):
        e = self.computeMembership()
        self.computeCentroids()
        print "e", e, "tol", self.tol
        while e > self.tol:
            e = self.computeMembership()
            print "e", e, "tol", self.tol
            self.computeCentroids()
        
    def test(self, F):
        res = []
        tf = np.array(F, ndmin = 2)
        for i in range(0, self.centroids_number):
            a = self.centroids[i]
            b = tf
            dist = np.linalg.norm(a - b)
            res.append( (i, dist) )
        res = sorted(res, key=itemgetter(1))
        return res
             

    def dumpCentroids(self):
        print "dumpCentroids"
        for i in self.centroids:
            print i


    def dumpFeatures(self):
        print "dumpFeatures"
        for i in self.features:
            print i

if __name__ == "__main__":
    c = fuzzy_cmeans(4, 4, 1.5, 0.005)
    c.dumpCentroids()
    c.loadFeatures("cmeans.data2")
    c.dumpFeatures()
    c.train()
    c.dumpCentroids()

    fp = open("cmeans.data", 'r')
    l = fp.readlines()
    fp.close()
    for p in l:
        ps = p.split(",")
        f = []
        for v in ps:
            f.append( float(v) )
           
        res = c.test(f)
        print "f = ", f
        print "res = ", res
