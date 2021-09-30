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
x1s = np.random.uniform(start, end, num)
x2s = np.random.uniform(start, end, num)
xSet = np.concatenate((x1s[None, :], x2s[None, :]), axis=0)

# %% set f vector and label the points
f = np.array([2, -2])
ySet = np.sign(getY(f, xSet))

plt.axes().set_aspect("equal")
# plot points
plt.plot(x1s[np.where(ySet<0)], x2s[np.where(ySet<0)], "x", color="black")
plt.plot(x1s[np.where(ySet>0)], x2s[np.where(ySet>0)], "o", color="red")
# plot f
lineX1 = np.linspace(start*1.5, end*1.5, 100)
lineX2 = getX2(f, lineX1)
plt.plot(lineX1, lineX2, color="blue")
plt.xlim(start*1.5, end*1.5)
plt.ylim(start*1.5, end*1.5)
plt.show()

# %% find g iteratively
g = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
print("initial g:", g)

for _ in range(100):
    # add new g to the plot
    if _ > 0:
        plt.arrow(0, 0, g[0], g[1], head_width=0.5, width=0.1, linestyle="--")
        plt.show()
    
    # calculate the guessed yprime
    yprimeSet = np.sign(getY(g, xSet))
    
    # catch the nearst error point
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
    
    # plot
    plt.axes().set_aspect("equal")
    # plot original points
    plt.plot(x1s[np.where(ySet<0)], x2s[np.where(ySet<0)], "x", color="black")
    plt.plot(x1s[np.where(ySet>0)], x2s[np.where(ySet>0)], "o", color="red")
    # make first coefficient of g be positive
    # if g[0] < 0:
    #     g = -g
    # mark error points
    plt.scatter(x1s[np.where(ySet!=yprimeSet)], x2s[np.where(ySet!=yprimeSet)], s=200, color="none", edgecolor="blue")
    # highlight the nearest error point
    plt.scatter(x1sErrorChosen, x2sErrorChosen, s=200, color="none", edgecolor="blue", linewidths=3)
    # mark g and normal vector
    lineX1 = np.linspace(start*1.5, end*1.5, 100)
    lineX2 = getX2(g, lineX1)
    plt.plot(lineX1, lineX2, "b--")
    plt.arrow(0, 0, g[0], g[1], head_width=0.5, width=0.1)
    plt.xlim(start*1.5, end*1.5)
    plt.ylim(start*1.5, end*1.5)
    plt.title("iter. {}, g: {}".format(_, np.round(g, 2)))
    
    # update g
    g = g + np.sign(ySetErrorChosen) * np.array([x1sErrorChosen, x2sErrorChosen])
    
# final plot
plt.axes().set_aspect("equal")
# plot original points
plt.plot(x1s[np.where(ySet<0)], x2s[np.where(ySet<0)], "x", color="black")
plt.plot(x1s[np.where(ySet>0)], x2s[np.where(ySet>0)], "o", color="red")
# mark g and normal vector
lineX1 = np.linspace(start*1.5, end*1.5, 100)
lineX2 = getX2(g, lineX1)
plt.arrow(0, 0, g[0], g[1], head_width=0.5, width=0.1)
plt.plot(lineX1, lineX2, "b--")
plt.xlim(start*1.5, end*1.5)
plt.ylim(start*1.5, end*1.5)
plt.title("iter. {} --- Final.".format(_))
plt.show()