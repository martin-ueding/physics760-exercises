#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

import argparse

import numpy as np
import matplotlib.pyplot as pl

def main():
    options = _parse_args()

    average = np.genfromtxt('averages.csv')
    rms = np.genfromtxt('rms.csv')

    pl.loglog(average[:, 0], average[:, 1], marker='.', linestyle='none', label='Average')
    pl.loglog(rms[:, 0], rms[:, 1], marker='.', linestyle='none', label='RMS')
    pl.grid(True)
    pl.legend(loc='best')
    pl.savefig('plot.pdf')

def _parse_args():
    '''
    Parses the command line arguments.

    :return: Namespace with arguments.
    :rtype: Namespace
    '''
    parser = argparse.ArgumentParser(description='')
    options = parser.parse_args()

    return options

if __name__ == '__main__':
    main()
