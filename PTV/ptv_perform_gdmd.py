import numpy as np
import ray
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from dmd_dict import dmd_pair
#%%

@ray.remote
def seed_dens_par_ptv(i,xlocal_ref,nnbors,pwr,sym_flag,y_axis):
    # FOR LINUX
    # xlocal = ray.get(xlocal_ref)
    # FOR WINDOWS OR WSLS
    xlocal = ray.get(xlocal_ref[0])
    f_root = 'field_data_denoised/'
    fname = f'{f_root}{i:04d}.npy'
    seeds = np.unique(np.load(fname),axis=0)
    if sym_flag:
        seeds[:,:,1] -= y_axis
        seeds[:,:,1] *= -1
        seeds[:,:,1] += y_axis
    seeds_to_store1 = np.squeeze(seeds[:,0,:])
    seeds_to_store2 = np.squeeze(seeds[:,1,:])

    dens1_slice = np.zeros((xlocal.shape[0], 2))
    dens2_slice = np.zeros((xlocal.shape[0], 2))
    def cart2pol(x, y, x_offset, y_offset):
        xnew = x - x_offset
        ynew = y - y_offset
        rho = np.sqrt(xnew**2 + ynew**2)
        phi = np.arctan2(ynew, xnew)
        return(rho, phi)
    
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


##%% Define parameters
# Load local grid
# mean_to_load = 'meanvel_chebspace.npz'
# mean_to_load = 'meanvel_linspace.npz'
mean_to_load = 'meanvel_customspace.npz'
meanvel = np.load(mean_to_load)
y_lip = meanvel['ylip']
y_axis = meanvel['yaxis']
XX = meanvel['X']
YY = meanvel['Y']
lgcy = YY[:,0] <= 2*y_axis
XX, YY = XX[lgcy,:], YY[lgcy,:]
xlocal = np.vstack([XX.flatten(),YY.flatten()]).transpose()
npts = xlocal.shape[0]

nnbors = 5
nModes = 8
x_st,y_st = 50, 5
xlim_dmd = [x_st, 400]
ylim_dmd = [y_st, y_axis]
ylim_dmd = [y_axis,2*y_axis-y_st]
ylim_dmd = [y_st,2*y_axis-y_st]
pwr = 5

nproc = 36

num_stored = 1198
dt = 1

##%% main function  
if __name__ == '__main__':   
    if not ray.is_initialized():
        ray.init(num_cpus=nproc,ignore_reinit_error=True)
    # FOR LINUX
    # xlocal_ref = ray.put(xlocal)
    # FOR WINDOWS OR WSL
    xlocal_ref = ray.put([ray.put(xlocal)])
    # Calculating the directional density of the seed particles 
    sym_flag = False
    futures = [
        seed_dens_par_ptv.remote(i, xlocal_ref, nnbors, pwr, sym_flag, y_axis)
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
    
    sym_flag = True
    futures = [
        seed_dens_par_ptv.remote(i, xlocal_ref, nnbors, pwr, sym_flag, y_axis)
        for i in range(num_stored)
    ]
    # results = ray.get(futures)
    # Track progress as tasks finish
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
    #      
    # density_local1 =  1/density_local1
    # density_local2 =  1/density_local2
    # idt = 4000
    # plt.clf()
    # ax2 = plt.subplot(211)
    # sc2 = ax2.scatter(xlocal[:,0],xlocal[:,1],c=density_local1[:,0,idt]); plt.colorbar(sc2,ax=ax2,pad=0)
    # ax2.scatter(seeds_to_store1[:,0,idt],seeds_to_store1[:,1,idt],c='r',s=3)
    # ax3 = plt.subplot(212,sharex=ax2,sharey=ax2)
    # sc3 = ax3.scatter(xlocal[:,0],xlocal[:,1],c=density_local1[:,1,idt]); plt.colorbar(sc3,ax=ax3,pad=0)
    # ax3.scatter(seeds_to_store1[:,0,idt],seeds_to_store1[:,1,idt],c='r',s=3)
    # plt.xlim(0.1,0.15)
    # plt.ylim(0.1,0.15)
    
    ##%% Plotting the seeds
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
    
    ##%% Perform DMD
    def cart2pol(x, y, x_offset, y_offset):
        xnew = x - x_offset
        ynew = y - y_offset
        rho = np.sqrt(xnew**2 + ynew**2)
        phi = np.arctan2(ynew, xnew)
        return(rho, phi)
    print('\n')
    print('Performing DMD...\n')
    # Limit the area to apply DMD
    lgcx = np.logical_and(xlocal[:,0]>xlim_dmd[0],xlocal[:,0]<xlim_dmd[1])
    lgcy = np.logical_and(xlocal[:,1]>ylim_dmd[0],xlocal[:,1]<ylim_dmd[1])
    lgcxy = np.logical_and(lgcx,lgcy)
    lgcd = np.tile(~lgcxy.reshape(npts,1,1),(1,2,num_stored*2))
    lgcg = np.tile(~lgcxy.reshape(npts,1),(1,2))
    np_dmd = np.sum(lgcxy)
    d1dmd = np.ma.masked_where(lgcd, density_local1).compressed().reshape(np_dmd*2, num_stored*2)
    d2dmd = np.ma.masked_where(lgcd, density_local2).compressed().reshape(np_dmd*2, num_stored*2)
    xdmd = np.ma.masked_where(lgcg, xlocal).compressed().reshape(np_dmd,2)
    
    E2, Phi2,Phi2_proj  = dmd_pair(d1dmd, d2dmd, nModes, dt)
    print('Done...\n')
    #%%
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.rcParams['text.usetex'] = True
    plt.style.use('default')
    ax1 = plt.subplot(111)
    ax1.plot(np.real(E2),np.imag(E2),'s',mfc='none')
    plt.axis('equal')
    plt.legend(['TR-PIV', 'Standard PIV'])
    #%%
    lgcE = np.imag(E2) > 0.05
    plt.figure(2)
    plt.clf()
    # for iplt in range(0,nModes,2):
    #     pltid = int(220 + iplt/2+1)
    for iplt in range(0,np.sum(lgcE)):
        pltid = int(220 + iplt+1)
    # for iplt in range(0,nModes):
    #     pltid = int(220 + iplt+1)
        mod2plot1 =np.arange(nModes)[lgcE][iplt]
        # mod2plot2 = 2
        dim2plot = 0
        Phivel2 = (Phi2-Phi2_proj)
        # Phivel2 = (Phi2)
        p2plot = np.real(Phivel2[:,mod2plot1].reshape(np_dmd,2))
        ax11 = plt.subplot(pltid)
        # plt.style.use('_mpl-gallery-nogrid')
        plt.ion()
        ax11.tripcolor(xdmd[:,0], xdmd[:,1], p2plot[:,dim2plot])
        ax11.set_xlim(xlim_dmd)
        ax11.set_ylim(ylim_dmd)
        ax11.title.set_text('gDMD')
        plt.tight_layout()



