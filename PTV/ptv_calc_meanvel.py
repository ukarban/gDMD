# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 13:09:40 2025

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import trapezoid as trapz
from scipy.interpolate import PchipInterpolator as pchip
from scipy.optimize import minimize

#%%
find_borders = False
calc_first_time = False
    
t_list = range(1198)
f_root = 'field_data_denoised/'


nnbors = 5
dt = 1

# Find domain borders
if find_borders:
    xmin,ymin = 1e5,1e5
    xmax,ymax = -1e5,-1e5
    nptmax = 0
    for it in t_list:
        fname = f'{f_root}{it:04d}.npy'
        seeds = np.load(fname)
        xmin = min(xmin,np.min(seeds[:,:,0]))
        ymin = min(ymin,np.min(seeds[:,:,1]))
        xmax = max(xmax,np.max(seeds[:,:,0]))
        ymax = max(ymax,np.max(seeds[:,:,1]))
        nptmax = max(nptmax,seeds.shape[0])
        if np.mod(it,10) == 0:
            print(f'{it}/{t_list.stop} has been loaded')
            
    np.savez('borders',x=[xmin,xmax],y=[ymin,ymax],npt=nptmax)
mean_to_load = 'meanvel_linspace.npz'
# mean_to_load = 'meanvel_chebspace.npz'
# mean_to_load = 'meanvel_customspace.npz'
nx,ny = 180,240//4
dist_power = 1
if calc_first_time:
    borders = np.load('borders.npz')
    xvec = np.linspace(borders['x'].min(), borders['x'].max(),nx)
    yvec = np.linspace(borders['y'].min(), borders['y'].max(),ny)
else:
    borders = np.load('borders.npz')
    meanvel = np.load(mean_to_load)
    xvec = np.linspace(borders['x'].min(), borders['x'].max(),nx)
    yvec = np.linspace(borders['y'].min(), borders['y'].max(),ny)
    y_lip = meanvel['ylip']
    y_axis = meanvel['yaxis']
    y_lip2 = 2*y_axis - y_lip
    dylip = (y_axis-y_lip)/2.
    pchip_pts = [y_axis, y_lip2-dylip, y_lip2, y_lip2+dylip, 2*y_axis, yvec.max()]
    pchip_dy = [4, 4, 1, 4, 4, 6]
    # Target dy distribution as a function of y (position)
    def target_dy(y):
        return pchip(pchip_pts, pchip_dy)(y)
    def get_y(scale):
        y = [y_axis]
        for ipt in range(ny):
            dy = scale*target_dy(y[-1])
            y.append(y[-1]+dy)
        return np.vstack(y)
    def loss_fun(scale):
        y = [y_axis]
        for ipt in range(ny):
            dy = scale*target_dy(y[-1])
            y.append(y[-1]+dy)
        return np.abs(y[-1]-yvec.max())
    result = minimize(loss_fun, 1, method='SLSQP')
    scale = result.x
    yvec_half = get_y(scale).flatten()
    yvec_halfR = y_axis - (yvec_half[1:]-y_axis)[::-1]
    yvec = np.append(np.append(0.,yvec_halfR[yvec_halfR>0.]),yvec_half)
    

XX,YY = np.meshgrid(xvec,yvec,indexing='xy')


xy_cart = np.array([XX.flatten(),YY.flatten()]).transpose()
uv_cart = np.zeros(xy_cart.shape)

knn = NearestNeighbors(n_neighbors=nnbors)
umax = 0
for it in t_list:
    if np.mod(it,10) == 0:
        print(f'{it}/{t_list.stop} is loading')
    fname = f'{f_root}{it:04d}.npy'
    seeds = np.unique(np.load(fname),axis=0)
    uv_it = np.squeeze(np.diff(seeds,axis=1))/dt
    xy_it = np.squeeze(np.average(seeds,axis=1))
    knn.fit(xy_it)
    dist_mat, neighbours_mat = knn.kneighbors(xy_cart)
    dist_mat[dist_mat<1e-5] = 1e-5
    uave = np.average(uv_it[neighbours_mat,0],axis=1,weights=1/dist_mat)
    vave = np.average(uv_it[neighbours_mat,1],axis=1,weights=1/dist_mat)
    uv_cart += np.array([uave,vave]).transpose()/t_list.stop
    umax = max(umax,uv_it.max())
    
Umat = uv_cart[:,0].reshape(XX.shape)
Vmat = uv_cart[:,1].reshape(XX.shape)

# Find the axis and lip lines

Uthold = 2
lgca = Umat > Uthold
y_axis = np.average(YY[lgca])
lgcl = (YY < y_axis) & (Umat < Umat.max()*2/3) & (Umat > Umat.max()*1/3)
y_axis, y_lip = 0, 0
for ii in range(xvec.shape[0]):
    Uint = cumtrapz(Umat[lgca[:,ii],ii],yvec[lgca[:,ii]])
    yint = (yvec[lgca[:,ii]][1:] + yvec[lgca[:,ii]][0:-1])/2
    y_axis += np.interp(Uint[-1]/2,Uint,yint)/xvec.shape[0]
    y_lip += np.interp(Umat[:,ii].max()/2,Umat[:,ii][lgcl[:,ii]],yvec[lgcl[:,ii]])/xvec.shape[0]

test = False
if calc_first_time:
    np.savez('meanvel_linspace',X=XX,Y=YY,U=Umat,V=Vmat,yaxis=y_axis,ylip=y_lip)
else:
    np.savez('meanvel_chebspace',X=XX,Y=YY,U=Umat,V=Vmat,yaxis=y_axis,ylip=y_lip)

