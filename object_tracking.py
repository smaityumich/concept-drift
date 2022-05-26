#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import pairwise_distances
import json
import pandas as pd
from os import listdir, path





import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def true_parameter(t, a = 1, omega = 1):
    theta_1 = (np.exp(-a * t) * np.sin(t * 2 * np.pi * omega)).reshape((-1, 1))
    theta_2 = (np.exp(-a * t) * np.cos(t * 2 * np.pi * omega)).reshape((-1, 1))
    theta = np.concatenate([theta_1, theta_2], axis = 1)
    return theta





def observations(X, theta, tau = 0.1):
    y_true = pairwise_distances(theta, X)
    eps = np.random.normal(scale = tau, size = y_true.shape)
    y = y_true + eps
    return y


def gradient_estimate(X, theta_prev, y_part_gd, alpha):
    
    theta_prev = theta_prev.reshape((1, -1))
    pairwise = theta_prev - X
    
    y_mean = (alpha.T @ y_part_gd).T
    multiplier = np.linalg.norm(pairwise, axis = 1) ** 2 - y_mean.reshape((-1, ))
    multiplier = multiplier.reshape((-1, 1))
    
    gradient = multiplier * pairwise
    gradient = 2 * gradient.mean(axis = 0)
    return gradient


def Hessian(X, theta_prev, y_part_gd, alpha):
    
    n, d = X.shape
    
    theta_prev = theta_prev.reshape((1, -1))
    pairwise = theta_prev - X
    
    y_mean = (alpha.T @ y_part_gd).T
    multiplier = np.linalg.norm(pairwise, axis = 1) ** 2 - y_mean
    multiplier = multiplier.reshape((-1, 1))
    
    H = 2 * np.eye(d) * multiplier.mean()
    H += 4 * pairwise.T @ pairwise / n
    
    return H

def time_derivative(X, theta_prev, y_part_td, beta):
    
    y_mean = (beta.T @ y_part_td).T
    
    theta_prev = theta_prev.reshape((1, -1))
    pairwise = theta_prev - X
    
    td = (-2) * pairwise.T @ y_mean/y_mean.shape[0]
    return  td



def gradient_descend(X, y, h, eta, theta_start = np.zeros(shape = (2, 1))):
    T, n = y.shape
    d = X.shape[1]
    
    m = int(h ** (-4/5))
    alpha = np.arange(0, m)
    alpha = 2 * (2 * m - 1 - 3 * alpha)/(m * (m + 1))
    alpha = alpha.reshape((-1, 1))
    
    theta_estimates = np.zeros(shape = (T, d))
    
    ## update step
    for i in range(m, T):
        data_start = i - m
        y_part_gd = y[data_start:i, :]
        theta_prev = theta_estimates[i-1, :]
        gradient = gradient_estimate(X, theta_prev, y_part_gd, alpha)
        
        
        theta_next = theta_prev - eta * gradient
        theta_estimates[i, :] = theta_next.reshape((-1, ))
        
    return theta_estimates, m


def prediction_correction(X, y, h, eta, theta_start = np.zeros(shape = (2, 1))):
    T, n = y.shape
    d = X.shape[1]
    
    m = int(h ** (-4/5))
    alpha = np.arange(0, m)
    alpha = 2 * (2 * m - 1 - 3 * alpha)/(m * (m + 1))
    alpha = np.flip(alpha)
    alpha = alpha.reshape((-1, 1))
    
    
    p = int(h ** (-3/4))
    beta = np.arange(0, p)
    beta = 6 * (p - 1 - 2 * beta)/(h * p * (p**2 - 1))
    beta = np.flip(beta)
    beta = beta.reshape((-1, 1))
    
    theta_estimates = np.zeros(shape = (T, d))
    
    ## update step
    for i in range(m, T):
        data_start = i - m
        y_part_gd = y[data_start:i, :]
        theta_prev = theta_estimates[i-1, :]
        gradient = gradient_estimate(X, theta_prev, y_part_gd, alpha)
        hessian = Hessian(X, theta_prev, y_part_gd, alpha)
        
        data_start_td = i - p
        y_part_td = y[data_start_td:i, :]
        td = time_derivative(X, theta_prev, y_part_td, beta)
        
        theta_next = (theta_prev - eta * gradient).reshape((-1, 1)) - h * np.linalg.inv(hessian) @ td
        theta_estimates[i, :] = theta_next.reshape((-1, ))
        
    return theta_estimates, m, p





if __name__ == '__main__':
    fig, axs = plt.subplots(1, 2, figsize = (12, 5))
    
    
    labelsize = 20
    ticksize = 15
    lw = 2
    
    
    listdict = []
    path = '/path/to/folder/'
    dirname = path + 'summary/'
    for file in listdir(dirname):
        if path.isfile(dirname + file):
            with open(dirname + file, 'r') as f:
                try:
                    listdict.append((json.load(f)))
                except:
                    continue
            
    df = pd.DataFrame(listdict)
    df = df.loc[df['expt'] == 'object-tracking']
    
    display_cols = ['gd', 'pc',]
    agg_dict = dict()
    for key in display_cols:
        agg_dict[key] = ['mean', 'std']
    
    result = df.groupby(['h'], as_index=False).agg(agg_dict)
    result = result.sort_values('h', ascending=False)
    
    
    ax = axs[1]
    
    
    
    x = result['h']
    
    mean, std = result[('gd', 'mean')], result[('gd', 'std')]
    ax.errorbar(x, mean, std, color = 'k', linestyle = '--', marker = 'x', lw = 2)
    
    mean, std = result[('pc', 'mean')], result[('pc', 'std')]
    ax.errorbar(x, mean, std, color = 'r', linestyle = '-', marker = 'o', lw = 2)
    ax.set_xlabel(r'$h$', fontsize = labelsize)
    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(plt.LogLocator(2))
    ax.yaxis.set_minor_locator(plt.LogLocator(2))
    
    
    yticks = [1, 0.5, ]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=ticksize)
    ax.tick_params('both', labelsize=ticksize)
    ax.set_ylabel(r'$\|\hat\theta(t) - \theta^\star(t)\|_2$', fontsize=labelsize)
    ax.grid()
    
    lines = [
        Line2D([0], [0], color = 'k', linestyle='--', marker = 'x', lw = 2), 
        Line2D([0], [0], color = 'r', linestyle='-', marker = 'o', lw = 2),
        ]
    labels = ['GD', 'PC']
    ax.legend(lines, labels, fontsize = labelsize)
    
    
    ax = axs[0]
    
    nums = [300, 3000, 30000, ]
    ltys = ['-', '--', ':',]
    
    
    lines = [
        Line2D([0], [0], color = 'k', linestyle='-', lw = 2), 
        Line2D([0], [0], color = 'r', linestyle='-', lw = 2),
        ]
    labels = ['GD', 'PC']
    hs = [r'10^{-2}', r'10^{-3}', r'10^{-4}']
    
    
    
    
    
    for num, lty, hstr in zip(nums, ltys, hs):
        
        t = np.linspace(0, 3, num = num)
        theta_star =  true_parameter(t, a = 0, omega = 1)
        
        
        xgrid = np.arange(-1, 1.1, step = 0.2)
        ygrid = np.arange(-1, 1.1, step = 0.2)
        XG, YG = np.meshgrid(xgrid, ygrid)
        XG = XG.reshape((-1, ))
        YG = YG.reshape((-1, ))
    
        X = np.vstack((XG, YG))
        X = X.T
        
        
        y = observations(X, theta_star, tau = 0.5)
        h = t[1] - t[0]
        eta = h ** (0.3)
        theta_gd, m1 = gradient_descend(X, y, h = h, eta  = eta)
        
        eta = h ** (4/5)
        theta_pc, m2, p = prediction_correction(X, y, h = h, eta  = eta)
        
        
        no_of_steps = m1
        err_gd = np.linalg.norm(theta_star[no_of_steps:] - theta_gd[no_of_steps:], axis = -1)
        err_pc = np.linalg.norm(theta_star[no_of_steps:] - theta_pc[no_of_steps:], axis = -1)
        ax.plot(t[no_of_steps:], err_gd, color = 'k', linestyle = lty, lw = 2)
        ax.plot(t[no_of_steps:], err_pc, color = 'r', linestyle = lty, lw = 2)
        lines.append(Line2D([0], [0], linestyle = lty, lw = 2, c = 'blue'))
        labels.append(r'$h = ' + hstr + r'$')
        
        
        
    ax.set_ylabel(r'$\|\hat\theta(t) - \theta^\star(t)\|_2$', fontsize=labelsize)
    ax.set_xlabel(r'$t$', fontsize=labelsize)
    ax.legend(lines, labels, fontsize=labelsize)
    ax.tick_params('both', labelsize=ticksize)
    ax.grid()
    plt.savefig(path + 'object-tracking.pdf', bbox_inches='tight')
    
    
    
