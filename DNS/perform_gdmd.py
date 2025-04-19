# A routine to perform gap-based DMD (gDMD) using the DNS stored after running
# 'cylinder2d_gdmd.py'.


import numpy as np
import ray
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from dmd_dict import dmd_pair
#%%

@ray.remote
def seed_dens_par(i,s1_ref,s2_ref,xlocal_ref,nnbors,pwr):
    # FOR LINUX
    # s1 = ray.get(s1_ref)  # shape (N, 2, T)
    # s2 = ray.get(s2_ref)
    # xlocal = ray.get(xlocal_ref)
    # FOR WINDOWS OR WSL
    s1 = ray.get(s1_ref[0])  # shape (N, 2, T)
    s2 = ray.get(s2_ref[0])
    xlocal = ray.get(xlocal_ref[0])

    dens1_slice = np.zeros((xlocal.shape[0], 2))
    dens2_slice = np.zeros((xlocal.shape[0], 2))
    def cart2pol(x, y, x_offset, y_offset):
        xnew = x - x_offset
        ynew = y - y_offset
        rho = np.sqrt(xnew**2 + ynew**2)
        phi = np.arctan2(ynew, xnew)
        return(rho, phi)
    
    seeds_to_store1 = s1[:,:,i]
    seeds_to_store2 = s2[:,:,i]
    knn = NearestNeighbors(n_neighbors=nnbors)
    # for image 1
    knn.fit(seeds_to_store1)
    _, neighbours_mat = knn.kneighbors(seeds_to_store1)
    neighbours_mat = neighbours_mat[:,1:]
    x_dist =  (seeds_to_store1[neighbours_mat,0] - np.tile(seeds_to_store1[:,0],(nnbors-1,1)).T)
    y_dist =  (seeds_to_store1[neighbours_mat,1] - np.tile(seeds_to_store1[:,1],(nnbors-1,1)).T)
    rad, th = cart2pol(x_dist,y_dist,0,0)
    densx = np.sum(np.abs(np.cos(th)**pwr)*abs(x_dist),1)/np.sum(np.abs(np.cos(th)**pwr),1)
    densy = np.sum(np.abs(np.sin(th)**pwr)*abs(y_dist),1)/np.sum(np.abs(np.sin(th)**pwr),1)
    distanceloc_mat, neighboursloc_mat = knn.kneighbors(xlocal)
    distanceloc_mat[distanceloc_mat<1e-6] = 1e-6
    dens1_slice[:,0] = np.mean(densx[neighboursloc_mat],1)
    dens1_slice[:,1] = np.mean(densy[neighboursloc_mat],1)

    # for image 2
    x_dist =  (seeds_to_store2[neighbours_mat,0] - np.tile(seeds_to_store2[:,0],(nnbors-1,1)).T)
    y_dist =  (seeds_to_store2[neighbours_mat,1] - np.tile(seeds_to_store2[:,1],(nnbors-1,1)).T)
    rad, th = cart2pol(x_dist,y_dist,0,0)
    densx = np.sum(np.abs(np.cos(th)**pwr)*abs(x_dist),1)/np.sum(np.abs(np.cos(th)**pwr),1)
    densy = np.sum(np.abs(np.sin(th)**pwr)*abs(y_dist),1)/np.sum(np.abs(np.sin(th)**pwr),1)
    knn.fit(seeds_to_store2)
    distanceloc_mat, neighboursloc_mat = knn.kneighbors(xlocal)
    dens2_slice[:,0] = np.mean(densx[neighboursloc_mat],1)
    dens2_slice[:,1] = np.mean(densy[neighboursloc_mat],1)
    return dens1_slice, dens2_slice


#%% Define parameters
# DNS parameters (copied from cylinder2d_gdmd.py)
L = 2.2
H = 0.41
c_x = c_y = 0.2
r = 0.05
gdim = 2
resmindiv = 3
res_min = r / resmindiv
t = 0
T = 310                     # Final time
dt = 1 / 1600               # Time step size
Tsteady = T/100             # Time to reach steady state
save_freq = 50              # Saving frequency
num_steps = int(T / dt)
num_stored = int(num_steps/save_freq)+1
mu_value = 0.001            # viscosity
rho_value = 1               # density
U_value = 1                 # inlet velocity
random_dist = True          # flag for random seed distribution -> True: normal, False: uniform
stdHdiv = 3                 # Standard deviation of the normal distribution along y used for generating seeds
std_seed = H/stdHdiv
newseed_rate = 1            # New seed feeding frequency
newseed_num = 8             # Number of seeds generated each time
newseed_bw = L/100          # Length of the inlet zone where new seeds are added
nnbors = 15                 # Number of neighboring points 
# DMD parameters
nModes = 6                  # Number of modes to be computed
xlim_dmd = [0.3, 1.5]       # Limits for DMD window
ylim_dmd = [0.1, 0.3]
pwr = 5                     # Exponent used in weighted averaging of directional gap
nproc = 36
# Load the data
fname4npsave = str('final_np_rmin%02d_T%03d_stdH%02d_nsnum%02d.npz' % (resmindiv,T,stdHdiv,newseed_num))
# fname4npsave = str('np_s%03d_rmin%02d_T%03d_stdH%02d_nsnum%02d.npz' % (seed_factor,resmindiv,T,stdHdiv,newseed_num))
# fname4npsave = str('np_Re50_s%03d_rmin%02d_T%03d_stdH%02d_nsnum%02d.npz' % (seed_factor,resmindiv,T,stdHdiv,newseed_num))
print('Loading the data...')
data = np.load(fname4npsave)
u_to_store1,u_to_store2 = data['vel1'], data['vel2']
xlocal = data['xglobal']
E, E2 = data['E'], data['E2']
s1,s2 = data['seeds1'], data['seeds2']
print('Finished loading')
npts,num_stored = u_to_store1.shape[0],u_to_store1.shape[2]
idxMat = np.tile(np.arange(nnbors).reshape(1,nnbors),(len(xlocal),1))

#%% main function  
if __name__ == '__main__':   
    if not ray.is_initialized():
        ray.init(num_cpus=nproc,ignore_reinit_error=True)
    # FOR LINUX
    # s1_ref = ray.put(s1)
    # s2_ref = ray.put(s2)
    # xlocal_ref = ray.put(xlocal)
    # FOR WINDOWS OR WSL
    s1_ref = ray.put([ray.put(s1)])
    s2_ref = ray.put([ray.put(s2)])
    xlocal_ref = ray.put([ray.put(xlocal)])
    # Calculating the directional gap of seed particles   
    futures = [
        seed_dens_par.remote(i, s1_ref, s2_ref, xlocal_ref, nnbors, pwr)
        for i in range(num_stored)
    ]
    # results = ray.get(futures)
    # Track progress as tasks finish
    results = []
    with tqdm(total=num_stored) as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            result = ray.get(done[0])
            results.append(result)
            pbar.update(1)
    # Stack results into final arrays
    density_local1 = np.stack([r[0] for r in results], axis=2)
    density_local2 = np.stack([r[1] for r in results], axis=2)
    ray.shutdown()
    
    #%% Plotting seeds tracing the flow
    # from drawnow import drawnow
    # def plot_scatter2():
    #     plt.scatter(xx,yy,s=1)
    #     # plt.xlim(0,L)
    #     # plt.ylim(0,H)
    #     plt.axis('equal')
    #     plt.xlim(0.0,1.5)
    #     plt.ylim(0.1,0.3)
    # plt.ion();
    # plt.figure(2);
    # for ii in range(100):
    #     offset = 200
    #     xx,yy = s1[:,0,ii+offset],s1[:,1,ii+offset]
    #     drawnow(plot_scatter2, show_once=True)
    #     plt.pause(1e-3)
    
    #%% Perform DMD
    def cart2pol(x, y, x_offset, y_offset):
        xnew = x - x_offset
        ynew = y - y_offset
        rho = np.sqrt(xnew**2 + ynew**2)
        phi = np.arctan2(ynew, xnew)
        return(rho, phi)
    print('\n')
    print('Performing DMD...\n')
    # Limit the area to apply DMD
    xlim_dmd = [0.3, 1.5]
    ylim_dmd = [0.1, 0.3]
    rp,thp = cart2pol(xlocal[:,0],xlocal[:,1],c_x,c_y)
    lgcx = np.logical_and(xlocal[:,0]>xlim_dmd[0],xlocal[:,0]<xlim_dmd[1])
    lgcy = np.logical_and(xlocal[:,1]>ylim_dmd[0],xlocal[:,1]<ylim_dmd[1])
    lgcr = rp > r+0.0005
    lgcxyr = np.logical_and(np.logical_and(lgcx,lgcy),lgcr)
    lgcd = np.tile(~lgcxyr.reshape(npts,1,1),(1,2,num_stored))
    lgcg = np.tile(~lgcxyr.reshape(npts,1),(1,2))
    np_dmd = np.sum(lgcxyr)
    u1dmd = np.ma.masked_where(lgcd, u_to_store1).compressed().reshape(np_dmd*2, num_stored)
    u2dmd = np.ma.masked_where(lgcd, u_to_store2).compressed().reshape(np_dmd*2, num_stored)
    d1dmd = np.ma.masked_where(lgcd, density_local1).compressed().reshape(np_dmd*2, num_stored)
    d2dmd = np.ma.masked_where(lgcd, density_local2).compressed().reshape(np_dmd*2, num_stored)
    xdmd = np.ma.masked_where(lgcg, xlocal).compressed().reshape(np_dmd,2)
    
    trange = np.arange(u1dmd.shape[1]//1) + u1dmd.shape[1]//1*0
    E, Phi,Phi_proj = dmd_pair(u1dmd[:,trange], u2dmd[:,trange], nModes, dt)
    E2, Phi2,Phi2_proj  = dmd_pair(d1dmd[:,trange], d2dmd[:,trange], nModes, dt)
    print('Done...\n')
    #%%
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.rcParams['text.usetex'] = True
    plt.style.use('default')
    ax1 = plt.subplot(111)
    ax1.plot(np.real(E),np.imag(E),'s',np.real(E2),np.imag(E2),'s',mfc='none')
    plt.axis('equal')
    plt.legend(['TR-PIV', 'Standard PIV'])

    #%%
    mod2plot1 =2
    mod2plot2 = np.argmin(np.abs(E2-E[mod2plot1]))
    # mod2plot2 = 2
    dim2plot = 0
    Phivel2 = (Phi2-Phi2_proj)
    angle_corr = -np.angle(np.sum(Phi[:,mod2plot1]*Phi2[:,mod2plot2])/
                          np.sum(np.abs(Phi[:,mod2plot1]*Phivel2[:,mod2plot2])))
    p2plot = np.real(Phi[:,mod2plot1].reshape(np_dmd,2))
    p2plot2 = np.imag((Phivel2*np.exp(1j*angle_corr))[:,mod2plot2].reshape(np_dmd,2))
    
    plt.figure(2)
    plt.clf()
    ax21,ax22 = plt.subplot(211), plt.subplot(212)
    # plt.style.use('_mpl-gallery-nogrid')
    plt.ion()
    ax21.tripcolor(xdmd[:,0], xdmd[:,1], p2plot[:,dim2plot])
    ax21.set_xlim(xlim_dmd)
    ax21.set_ylim(ylim_dmd)
    ax21.title.set_text('TR-PIV')
    ax22.tripcolor(xdmd[:,0], xdmd[:,1], p2plot2[:,dim2plot])
    # ax22.tripcolor(xdmd[:,0], xdmd[:,1], p2plot2[:,1])
    ax22.set_xlim(xlim_dmd)
    ax22.set_ylim(ylim_dmd)
    ax22.title.set_text('Standard PIV')
    plt.tight_layout()



