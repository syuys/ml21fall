#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 19:11:09 2021

@author: md703
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import matplotlib.pyplot as plt
plt.close("all")
import numpy as np
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300


# %%
def getX2(f, x1):
    x2 = -f[0]/f[1] * x1
    return x2


def getY(w, x):
    y = np.matmul(w, x)
    return y


# %% Generate points
start = -10
end = 10
num = 20
x1s = np.random.uniform(start+0.5, end-0.5, num)
x2s = np.random.uniform(start+0.5, end-0.5, num)
xSet = np.concatenate((x1s[None, :], x2s[None, :]), axis=0)

# %% set f vector and label the points
f = np.array([5, -2])
ySet = np.sign(getY(f, xSet))

plt.axes().set_aspect("equal")
# plot points
plt.plot(x1s[np.where(ySet<0)], x2s[np.where(ySet<0)], "x", color="black")
plt.plot(x1s[np.where(ySet>0)], x2s[np.where(ySet>0)], "o", color="red")
# plot f
lineX1 = np.linspace(start, end, 100)
lineX2 = getX2(f, lineX1)
plt.plot(lineX1, lineX2, color="blue")
plt.xlim(start, end)
plt.ylim(start, end)
plt.show()

# %% find g iteratively
g = np.array([np.random.uniform(0, 1), np.random.uniform(-1, 1)])

for _ in range(100):
    yprimeSet = np.sign(getY(g, xSet))
    plt.axes().set_aspect("equal")
    # plot original points
    plt.plot(x1s[np.where(ySet<0)], x2s[np.where(ySet<0)], "x", color="black")
    plt.plot(x1s[np.where(ySet>0)], x2s[np.where(ySet>0)], "o", color="red")
    # make first coefficient of g be positive
    if g[0] < 0:
        g = -g
    # plot g
    lineX1 = np.linspace(start, end, 100)
    lineX2 = getX2(g, lineX1)
    plt.plot(lineX1, lineX2, "b--")
    # plot error points
    plt.scatter(x1s[np.where(ySet!=yprimeSet)], x2s[np.where(ySet!=yprimeSet)], s=200, color="none", edgecolor="blue")
    plt.xlim(start, end)
    plt.ylim(start, end)
    plt.show()
    
    # choose the nearst error point
    if np.where(ySet!=yprimeSet)[0].size == 0:
        print("final g:", g)
        break
    x1sError = x1s[np.where(ySet!=yprimeSet)]
    x2sError = x2s[np.where(ySet!=yprimeSet)]
    ySetError = ySet[np.where(ySet!=yprimeSet)]
    xErrorSet = np.concatenate((x1sError[None, :], x2sError[None, :]), axis=0)
    distanceSet = abs(getY(g, xErrorSet)) / np.sqrt((g**2).sum())
    x1sErrorChosen = x1sError[np.argmin(distanceSet)]
    x2sErrorChosen = x2sError[np.argmin(distanceSet)]
    ySetErrorChosen = ySetError[np.argmin(distanceSet)]
    
    # update g
    g = g + np.sign(ySetErrorChosen) * np.array([x1sErrorChosen, x2sErrorChosen])
    
