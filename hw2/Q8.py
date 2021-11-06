#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 09:57:19 2021

@author: md703
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

def f(n):
    delta = 4*(2*n+1) * np.exp(-n/800)
    return delta

N = np.linspace(10000, 14000, num=5)
delta = f(N)
plt.plot(N, delta, "-o")
plt.show()