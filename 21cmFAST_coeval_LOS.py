#Code to generate 1D LOS data of IGM fields from 21cmFAST coeval boxes. Version of 3.3.1 of 21cmFAST is used here.
#The code generates a number of LOS uniformly distributed in the simulation box along all three axes.
#The LOS data is saved in a single binary file for each IGM model.
from ast import Del
import matplotlib.pyplot as plt
import numpy as np
import os
#We change the default level of the logger so that we can see what's happening with caching.
import logging, sys, os
from matplotlib import gridspec
import constants

logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

import py21cmfast as p21c

# For plotting the cubes, we use the plotting submodule:
from py21cmfast import plotting

# For interacting with the cache
from py21cmfast import cache_tools

import py21cmfast as p21c
print(f' Using 21cmFAST version {p21c.__version__}')

import time					#Just to see how much of your life did this code take
from datetime import datetime
now = datetime.now()
print('Time: %s' % now.strftime('%H:%M:%S'))
ti_whole = time.time()

#Set input parameters here
Mpc = constants.Mpc/1e5 #Megaparsec in km
Lbox = 50           #Simulation box size in cMpc
Nbins = 256         #Number of pixels in the simulation box on each side
Nlos = 10           #Number of LOS to generate (uniformly distributed in the simulation box along all three axes)
z = 6               #Redshift of the simulation
#Cosmological parameters taken from Planck 2013 results (Planck Collaboration et al. 2014, A&A , 571, A16)
sigma_8 = 0.829     #Normalization of the matter power spectrum on 8 Mpc/h scales
omega_0 = 0.308     #Matter density parameter
omega_L = 0.692     #Dark energy density parameter
omega_b = 0.0482    #Baryon density parameter
n_s = 0.961         #Spectral index
h = 0.678           #Hubble parameter
X_H = 0.76          #Hydrogen fraction
YHe = 0.24          #Helium fraction
mfp_ion = 0.75*(1+z)#Maximum mean free path of ionizing photons in cMpc taken from Becker et al. 2021, MNRAS , 508, 1853
#Varying IGM models
ioneff = [5]       #Ionizing efficiency of galaxies (zeta) - see 21cmFAST documentation for details
logfX =  [-3]       #X-ray efficiency of galaxies (log10(f_X))
logLXray = np.log10(3.4e40*np.power(10.,logfX)) #X-ray luminosity as parametrized by Furlanetto 2006, MNRAS , 371, 867

seed = 37   #Important for AI/ML work.


#Distribute LOS uniformly in the simulation box along all three axes
Nlos_perside = int(np.ceil(Nlos/3))
Nlos_perrow = int(np.ceil(np.sqrt(Nlos_perside)))
ind_increment = int(np.floor(Nbins/Nlos_perrow))

x_ind = np.empty((Nlos_perrow**2,2))
y_ind = np.empty((Nlos_perrow**2,2))
z_ind = np.empty((Nlos_perrow**2,2))

for i in range(Nlos_perrow):

    for j in range(Nlos_perrow):

        x_ind[i*Nlos_perrow+j] = [int((i+0.5)*ind_increment),int((j+0.5)*ind_increment)]
        y_ind[i*Nlos_perrow+j] = [int(Nbins-(i+0.5)*ind_increment),int(Nbins-(j+0.5)*ind_increment)]
        z_ind[i*Nlos_perrow+j] = [int((i+0.5)*ind_increment),int(Nbins-(j+0.5)*ind_increment)]

x_ind = x_ind[:Nlos_perside,:]
y_ind = y_ind[:Nlos_perside,:]
z_ind = z_ind[:Nlos-len(x_ind)-len(y_ind),:]


#Generate positional and velocity axes for the LOS data
pos_axis = np.arange(0,Lbox*1000,Lbox*1000/Nbins)
omega_k = 1-omega_0-omega_L
r_conv = 1./(1.+z)/1000./h
H = 100.*h*np.sqrt(omega_L+omega_0*(1+z)**3+omega_k*(1+z)**2)
vmax = H*Lbox*1000*r_conv
vel_axis = np.arange(0.,vmax,vmax/Nbins)


print('Start simulation')
#Generate initial conditions for the simulation
initial_conditions = p21c.initial_conditions(
    user_params = {"HII_DIM": Nbins, "BOX_LEN": Lbox, "USE_INTERPOLATION_TABLES": True, "N_THREADS": 6, "MINIMIZE_MEMORY": True},
    cosmo_params = p21c.CosmoParams(SIGMA_8=sigma_8,hlittle=h,OMm=omega_0,OMb=omega_b,POWER_INDEX=n_s),
    Y_He=YHe,
    random_seed=seed
)

print('Initial conditions')
tf = time.time()
print('Time taken: '+str(tf-ti_whole))

#Generate perturbed density, velocity and halo fields from the initial conditions
perturbed_field = p21c.perturb_field(
    redshift = z,
    init_boxes = initial_conditions,
)

print('Perturbed field')
tf = time.time()
print('Time taken: '+str(tf-ti_whole))

#Start the looop over the varying IGM models
for i in range(len(ioneff)):

    print('MFP_ion=%.2fcMpc, ION_eff=%.2f, f_X=%.2f' % (mfp_ion,ioneff[i],logfX[i]))

    #Generate the spin temperature and ionized fraction fields from the perturbed density, velocity and halo fields
    spin_temp = p21c.spin_temperature(
        perturbed_field=perturbed_field,
        astro_params = p21c.AstroParams({"HII_EFF_FACTOR":ioneff[i],"R_BUBBLE_MAX":mfp_ion,"L_X":logLXray[i]}),
        flag_options = p21c.FlagOptions({'USE_TS_FLUCT':True}),
    )
    
    print('Spin temperature')
    tf = time.time()
    print('Time taken: '+str(tf-ti_whole))

    ionized_field = p21c.ionize_box(
        perturbed_field=perturbed_field,
        spin_temp = spin_temp,
        astro_params = p21c.AstroParams({"HII_EFF_FACTOR":ioneff[i],"R_BUBBLE_MAX":mfp_ion,"L_X":logLXray[i]}),
        flag_options = p21c.FlagOptions({'USE_TS_FLUCT':True}),
    )

    print('Ionize box')
    tf = time.time()
    print('Time taken: '+str(tf-ti_whole))

    xHI_mean = np.mean(ionized_field.xH_box)
    print('<x_HI>=%.5f' % xHI_mean)


    #Extract LOS data of IGM fields
    Delta    = np.empty((Nlos,Nbins))   #Overdensity Delta = rho/<rho> - 1
    xHI      = np.empty((Nlos,Nbins))   #Neutral hydrogen fraction
    TK       = np.empty((Nlos,Nbins))   #Gas kinetic temperature in K
    vpec     = np.empty((Nlos,Nbins))   #Peculiar velocity in km/s
    Gamma_HI = np.empty((Nlos,Nbins))   #Photoionization rate in s^-1
    J21      = np.empty((Nlos,Nbins))   #Lyman-Werner background in units of 10^-21 erg/s/cm^2/Hz/sr

    for j in range(len(x_ind)):

        Delta[j,:]    = perturbed_field.density[:,int(x_ind[j,0]),int(x_ind[j,1])]
        xHI[j,:]      = ionized_field.xH_box[:,int(x_ind[j,0]),int(x_ind[j,1])]
        TK[j,:]       = ionized_field.temp_kinetic_all_gas[:,int(x_ind[j,0]),int(x_ind[j,1])]
        vpec[j,:]     = perturbed_field.velocity[:,int(x_ind[j,0]),int(x_ind[j,1])]
        Gamma_HI[j,:] = ionized_field.Gamma12_box[:,int(x_ind[j,0]),int(x_ind[j,1])]
        J21[j,:]      = spin_temp.J_21_LW_box[:,int(x_ind[j,0]),int(x_ind[j,1])]

    for j in range(len(y_ind)):

        Delta[len(x_ind)+j,:]    = perturbed_field.density[:,int(y_ind[j,0]),int(y_ind[j,1])]
        xHI[len(x_ind)+j,:]      = ionized_field.xH_box[:,int(y_ind[j,0]),int(y_ind[j,1])]
        TK[len(x_ind)+j,:]       = ionized_field.temp_kinetic_all_gas[:,int(y_ind[j,0]),int(y_ind[j,1])]
        vpec[len(x_ind)+j,:]     = perturbed_field.velocity[:,int(y_ind[j,0]),int(y_ind[j,1])]
        Gamma_HI[len(x_ind)+j,:] = ionized_field.Gamma12_box[:,int(y_ind[j,0]),int(y_ind[j,1])]
        J21[len(x_ind)+j,:]      = spin_temp.J_21_LW_box[:,int(y_ind[j,0]),int(y_ind[j,1])]

    for j in range(len(z_ind)):

        Delta[len(x_ind)+len(y_ind)+j,:]    = perturbed_field.density[:,int(z_ind[j,0]),int(z_ind[j,1])]
        xHI[len(x_ind)+len(y_ind)+j,:]      = ionized_field.xH_box[:,int(z_ind[j,0]),int(z_ind[j,1])]
        TK[len(x_ind)+len(y_ind)+j,:]       = ionized_field.temp_kinetic_all_gas[:,int(z_ind[j,0]),int(z_ind[j,1])]
        vpec[len(x_ind)+len(y_ind)+j,:]     = perturbed_field.velocity[:,int(z_ind[j,0]),int(z_ind[j,1])]
        Gamma_HI[len(x_ind)+len(y_ind)+j,:] = ionized_field.Gamma12_box[:,int(z_ind[j,0]),int(z_ind[j,1])]
        J21[len(x_ind)+len(y_ind)+j,:]      = spin_temp.J_21_LW_box[:,int(z_ind[j,0]),int(z_ind[j,1])]

    Delta = np.reshape(Delta,Nlos*Nbins)
    xHI = np.reshape(xHI,Nlos*Nbins)
    TK = np.reshape(TK,Nlos*Nbins)
    vpec = np.reshape(vpec,Nlos*Nbins)
    Gamma_HI = np.reshape(Gamma_HI,Nlos*Nbins)
    J21 = np.reshape(xHI,Nlos*Nbins)

    #Conversions to desired units/quantities
    Delta = 1+Delta
    vpec = vpec*Mpc
    Gamma_HI = Gamma_HI*1e-12

    #Compute mean gas temperature in neutral regions defined as xHI>=0.9
    ind_neu = np.where(xHI>=0.9)[0]
    TK_neu = np.mean(TK[ind_neu])
    print('<TK (xHI>=0.9)>=%.3f' % TK_neu)

    #Save the data in a single binary file with the following format:
    array = np.array([z,omega_0,omega_L,omega_b,h,Lbox*1000,X_H,Nbins,Nlos,logfX[i],ioneff[i],xHI_mean])
    array = np.append(array,pos_axis)
    array = np.append(array,vel_axis)
    array = np.append(array,Delta)
    array = np.append(array,xHI)
    array = np.append(array,TK)
    array = np.append(array,vpec)
    array = np.append(array,Gamma_HI)
    array = np.append(array,J21)

    array = array.astype('float32')
    array.tofile('datasets/21cmFAST_los/los/los_%dMpc_%d_n%d_z%.3f_fX%.1f_xHI%.2f.dat' % (Lbox,Nbins,Nlos,z,logfX[i],xHI_mean),sep='',format='float32')

    #Delete the cached data to save disk space. Note that these files are the most computationally expensive.
    p21c.cache_tools.clear_cache(kind=['IonizedBox','TsBox'])

    tf = time.time()
    print('Time taken: '+str(tf-ti_whole))
    now = datetime.now()
    print('Time: %s' % now.strftime('%H:%M:%S'))