# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:19:54 2020

@author: Torstein
"""

#! /usr/bin/env python
# File: computeGeometryExample.py
#
# Compute some differential geometric quantities for a selection of spacetimes

from __future__ import division
import sys
import numpy
import sympy
from differentialGeometry import computeGeometry, printGeometry

def computeFlatRobertsonWalkerCartesian():
    '''Compute some geometric properties of the flat
    Robertson-Walker geometry in Cartesian coordinates'''
    t = sympy.Symbol('t')
    a = sympy.Function('a')(t)
    x,y,z,c= sympy.symbols(['x','y','z','c'])
    coords = numpy.array([t,x,y,z])
    d = coords.size
    g_ = numpy.zeros((d,d),dtype=object)
    g_[0,0] = c**2
    g_[1,1] = -a**2
    g_[2,2] = -a**2
    g_[3,3] = -a**2
    geometry = computeGeometry(g_,coords)
    # geometry = [coords,g_,g,gdet,c9l,r5n,r5n_,r4i_,r,e6n_,e6n]
    if (geometry != None):
        printGeometry(geometry)

def main(argv):
    computeFlatRobertsonWalkerCartesian()

if __name__ == "__main__":
    main(sys.argv[1:])