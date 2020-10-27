# -*- coding: utf-8 -*-
# File: differentialGeometry.py
#
# Computes various quantities of differential geometric from a
# given coordinate system and metric in that coordinate system.
# Checks that the covariant derivatives of the covariant (g_) and
# contravariant (g) metrics vanishes.
# Checks that the covariant divergens of the Einstein tensor vanishes.

from __future__ import division
import numpy
import sympy

def ginv(g_):
    '''Compute contravariant metric g from covariant metric g_'''
    gM    = sympy.Matrix(g_)
    g = gM.inv()
    return numpy.array(g)

def determinantg(g_):
    '''Compute determinant of covariant metric g_'''
    gM = sympy.Matrix(g_)
    gdet = gM.det()
    return gdet

def dg_(g_,coords):
    '''Compute first derivatives g_x of covariant metric g_'''
    d = coords.size
    g_x = numpy.zeros((d,d,d),dtype=object)
    for k in range(d):
        for m in range(d):
            for n in range(d):
                g_x[k,m,n] = sympy.diff(g_[k,m],coords[n])
    return g_x

def d2g(g_x,coords):
    '''Compute second derivatives g_xy of covariant metric g_'''
    d = coords.size
    g_xy = numpy.zeros((d,d,d,d),dtype=object)
    for j in range(d):
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    g_xy[j,k,m,n] = sympy.diff(g_x[j,k,m],coords[n])
    return g_xy

def christoffel(g,g_x):
    '''Compute Christoffel symbols (metric connection coefficients)'''
    d = g[:,0].size
    c9l = numpy.zeros((d,d,d),dtype=object)
    for j in range(d):
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    c9l[j,k,m] += g[j,n]*(g_x[n,k,m]+g_x[n,m,k]-g_x[k,m,n])
    return sympy.Rational(1,2)*c9l

def dconnection(c8n,coords):
    '''Compute derivatives of connection coefficients'''
    d = c8n[:,0,0].size
    c8nx = numpy.zeros((d,d,d,d),dtype=object)
    for j in range(d):
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    c8nx[j,k,m,n] = sympy.diff(c8n[j,k,m],coords[n])
    return c8nx

def riemann(c8n,c8nx):
    '''Compute Riemann tensor from connection coefficients and derivatives'''
    d = c8n[:,0,0].size
    r5n = numpy.zeros((d,d,d,d),dtype=object)
    for j in range(d):
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    r5n[j,k,m,n] += c8nx[j,n,k,m]
                    r5n[j,k,m,n] -= c8nx[j,m,k,n]
                    for q in range(d):
                        r5n[j,k,m,n] += c8n[j,m,q]*c8n[q,n,k]
                        r5n[j,k,m,n] -= c8n[j,n,q]*c8n[q,m,k]
                    r5n[j,k,m,n] = sympy.simplify(r5n[j,k,m,n])
    return r5n

def riemann_(g_, r5n):
    '''Return the Riemann tensor with all indices in lower position'''
    d = g_[:,0].size
    r5n_= numpy.zeros((d,d,d,d),dtype=object)
    for j in range(d):
        for k in range(d):
            for m in range(d):
                for n in range(d):
                    for q in range(d):
                        r5n_[j,k,m,n] += g_[j,q]*r5n[q,k,m,n]
                    r5n_[j,k,m,n] = sympy.simplify(r5n_[j,k,m,n])
    return r5n_

def ricci_(r5n):
    '''Compute covariant components of Ricci tensor from Riemann tensor'''
    d = r5n[:,0,0,0].size
    r4i_ = numpy.zeros((d,d),dtype=object)
    for m in range(d):
        for n in range(d):
            for k in range(d):
                r4i_[m,n] -= r5n[k,m,k,n]
    return r4i_

def scalarcurvature(g, r4i_):
    '''Compute scalar curvature from Ricci tensor and contravariant metric'''
    d = r4i_[:,0].size
    r = 0
    for m in range(d):
        for n in range(d):
            r += g[m,n]*r4i_[n,m]
    return sympy.simplify(r)

def einstein_(g_,r,r4i_):
    '''Compute covariant components of the Einstein tensor'''
    d = g_[:,0].size
    e6n_ = numpy.zeros((d,d),dtype=object)
    for m in range(d):
        for n in range(d):
            e6n_[m,n] = sympy.simplify(r4i_[m,n]-sympy.Rational(1,2)*r*g_[m,n])
    return e6n_

def einstein(g,e6n_):
    '''Compute contravariant components of the Einstein tensor'''
    d = g[:,0].size
    e6n = numpy.zeros((d,d),dtype=object)
    for m in range(d):
        for n in range(d):
            for j in range(d):
                for k in range(d):
                    e6n[m,n] += g[m,j]*g[n,k]*e6n_[j,k]
    return e6n

def checkDg_(g_,c9l,coords):
    '''Is the covariant derivative of the covariant metric zero?'''
    d = coords.size
    g_X = numpy.zeros((d,d,d),dtype=object)
    g_X_is_zero = True
    for k in range(d):
        for m in range(d):
            for n in range(d):
                g_X[k,m,n] += sympy.diff(g_[k,m], coords[n])
                for q in range(d):
                    g_X[k,m,n] -= c9l[q,n,k]*g_[q,m]
                    g_X[k,m,n] -= c9l[q,n,m]*g_[k,q]
                g_X[k,m,n] = sympy.simplify(g_X[k,m,n])
                if (g_X[k,m,n] != 0):
                    print('g_({},{};{}) = {}'.format(k,m,n,g_X[k,m,n]))
                    #print "g_{%d,%d;%d} = %s" % (k,m,n,g_X[k,m,n])
                    g_X_is_zero = False
    return g_X_is_zero

def checkDg(g,c9l,coords):
    '''Is the covariant derivatives of the contravariant metric g zero?'''
    d = coords.size
    gX = numpy.zeros((d,d,d),dtype=object)
    gX_is_zero = True
    for k in range(d):
        for m in range(d):
            for n in range(d):
                gX[k,m,n] += sympy.diff(g[k,m], coords[n])
                for q in range(d):
                    gX[k,m,n] += c9l[k,n,q]*g[q,m]
                    gX[k,m,n] += c9l[m,n,q]*g[k,q]
                gX[k,m,n] = sympy.simplify(gX[k,m,n])
                if (gX[k,m,n] != 0):
                    print('g^({},{}_;{}) = {}'.format(k,m,n,gX[k,m,n]))
                    #print "g^{%d,%d}_;%d = %s" % (k,m,n,gX[k,m,n])
                    gX_is_zero = False
    return gX_is_zero

def checkDivEinstein(e6n,c9l,coords):
    '''Is the Einstein tensor covariantly conserved?'''
    d = coords.size
    divE6n = numpy.zeros(d,dtype=object)
    divE6n_is_zero = True
    for m in range(d):
        for n in range(d):
            divE6n[m] += sympy.diff(e6n[m,n], coords[n])
            for q in range(d):
                divE6n[m] += c9l[m,n,q]*e6n[q,n]
                divE6n[m] += c9l[n,n,q]*e6n[m,q]
        divE6n[m] = sympy.simplify(divE6n[m])
        if (divE6n[m] != 0):
            print('G^({} n)_;n = {}'.format(m, divE6n[m]))
            #print "G^{%d n}_;n = %s" % (m, divE6n[m])
            divE6n_is_zero = False
    return divE6n_is_zero

def computeGeometry(g_, coords):
    '''Compute some geometric properties of the space defined by the
    covariant metric 'g_' in the coordinates 'coords'.'''
    g   = ginv(g_)
    gdet = determinantg(g_)
    g_x   = dg_(g_,coords)
    c9l  = christoffel(g,g_x)
    stat1 = checkDg_(g_,c9l,coords)
    print ("Covariant derivatives of covariant metric vanishes: {}".format(stat1))
    stat2 = checkDg(g,c9l,coords)
    print ("Covariant derivatives of contravariant metric vanishes: {}".format(stat2))
    c9lx = dconnection(c9l,coords)
    r5n  = riemann(c9l,c9lx)
    r5n_ = riemann_(g_,r5n)
    r4i_ = ricci_(r5n)
    r    = scalarcurvature(g,r4i_)
    e6n_ = einstein_(g_,r,r4i_)
    e6n = einstein(g,e6n_)
    stat3 = checkDivEinstein(e6n,c9l,coords)
    print ("Covariant divergence of the Einstein tensor vanishes: {}".format(stat3))
    if (stat1 and stat2 and stat3):
        return numpy.array([coords,g_,g,gdet,c9l,r5n,r5n_,r4i_,r,e6n_,e6n],
            dtype=object)

def linear2matrixIdx(m2,d):
    '''Convert linear index of antisymmetric matrices to matrix indices'''
    if (m2 >= d*(d-1)//2):
        return 0,0
    ma = 0; cumsum = d-2
    while (cumsum < m2):
        ma += 1; cumsum += d-1-ma
    mb = m2+d-cumsum-1
    return ma,mb

def printGeometry(geometry):
    '''Print some nonzero quantities of 'geometry'.'''
    # geometry = [coords,g_,g,gdet,c9l,r5n,r5n_,r4i_,r,e6n_,e6n]
    d    = geometry[0].size
    c9l  = geometry[4]
    r5n_ = geometry[6]
    r    = geometry[8]
    e6n  = geometry[10]
    d2   = d*(d-1)//2
    print ("\nScalarCurvature = {}".format(r))
    print ("\nNonzero components of Christoffel symbols:")
    for k in range(d):
        for m in range(d):
            for n in range(m,d):
                c9l_kmn = c9l[k,m,n]
                if (c9l_kmn != 0):
                    print('C^{}_{}{} = {}'.format(k,m,n,c9l_kmn))
                    #print "C^%d_%d%d = %s" % (k,m,n,c9l_kmn)
    print ("\nNonzero components of Einstein tensor:")
    for m in range(d):
        for n in range(m,d):
            e6n_mn = e6n[m,n]
            if (e6n_mn != 0):
                print('G^{}{} = {}'.format(m,n,e6n_mn))
                #print "G^%d%d = %s" % (m,n,e6n_mn)
    print ("\nNonzero components of Riemann tensor:")
    for m2 in range(d2):
        ma,mb = linear2matrixIdx(m2,d)
        for n2 in range(m2,d2):
            na,nb = linear2matrixIdx(n2,d)
            r5n_mn = r5n_[ma,mb,na,nb]
            if (r5n_mn != 0):
                print('R_{}{}{} = {}'.format(ma,mb,na,nb,r5n_mn))
                #print "R_%d%d%d%d = %s" % (ma,mb,na,nb,r5n_mn)