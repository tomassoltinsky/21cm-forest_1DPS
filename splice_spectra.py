#Code to generate longer 21-cm forest flux density spectra by splicing together multiple lines-of-sight.
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as scisi
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.patches as patches
import time					#Just to see how much of your life did this code take
ti_whole = time.time()
import glob, os
from numpy import random
import constants
import instrumental_features

fsize = 16
path = 'datasets/21cmFAST_los/'
z_name = 6.             #redshift
dvH = 250               #Velocity resolution in km/s. 
Nlos = 10               #Number of lines-of-sight in the original data
nfiles = 2              #Number of files in the original data
Nlos_new = Nlos*nfiles  #Number of lines-of-sight in the new data
extension_factor = 4    #Factor by which the length of the new LOS will be extended. The new data will have Nbins*extension_factor pixels.

#Find all of the datasets
files = glob.glob(path+'los/*.dat')
files_to_remove = glob.glob(path+'los/*fXnone*.dat')  #Ignore the IGM models with no X-ray heating
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*seed*.dat')    #Ignore the IGM models with different random seeds
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])

#Or you can select which files to use manually
files = ['datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-3.0_xHI0.80.dat'
        ,'datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-3.0_xHI0.11.dat']

print('Number of files: %d' % len(files))

logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))

#Find all of the parameter values in simulated data and read the data for interpolation
for j in range(len(files)):
    
    data = np.fromfile(str(files[j]),dtype=np.float32)
    logfX[j] = data[9]
    xHI_mean[j] = data[11]


#Read the data for the first file to get the box size and number of pixels
datafile = str('%slos_regrid/los_50Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat' % (path,Nlos,z_name,-2,0.25,dvH,0))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]	   #redshift
Nbins = int(data[7])	#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[8])	#Number of lines-of-sight
omega_0 = data[1]	   #matter density
omega_L = data[2]	   #vacuum density
omega_b = data[3]	   #baryon density
h = data[4]		      #small h (Hubble parameter)
box_size = data[5]	#box size in Mpc

Npix = Nbins*extension_factor
box_size = box_size*Npix/Nbins
dpos_axis = box_size/Npix
pos_axis = np.arange(0.,box_size,dpos_axis)
omega_k = 1-omega_0-omega_L		#curvature
H_0 = 1e7*h/constants.Mpc		#Hubble constant in s^-1
H = H_0*np.sqrt(omega_L+omega_0*(1+z)**3+omega_k*(1+z)**2)
vHmax = box_size/1000.*constants.Mpc*H/(1.+z)/h
dvH1 = vHmax/Npix
vH1 = np.arange(0.,vHmax,dvH1)/1.e5


#Generate a file with the axis information for the new data
array = np.array([z,box_size,Npix,Nlos_new])
array = np.append(array,pos_axis)
array = np.append(array,vH1)
array = array.astype('float32')
array.tofile('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,Nlos_new,z_name,dvH),sep='')


n_models = len(logfX)

#Randomly select LOS to be spliced together to generate longer LOS. The random seed is set to ensure reproducibility of the results.
#Note that all IGM models will have the same set of LOS spliced together.
seed = 370
random.seed(seed)
LOS = random.randint(0,Nlos*nfiles-1,size=Nlos_new*extension_factor)
LOS = np.reshape(LOS,(Nlos_new,extension_factor))


#Loop over all IGM models
done_perc = 0.01
for l in range(0,len(xHI_mean)):

  print('<xHI>=%.2f, log10(fX)=%.2f, %d/%d' % (xHI_mean[l],logfX[l],l+1,len(xHI_mean)))

  tau = np.empty((Nlos*nfiles,Nbins))

  for i in range(nfiles):

    #Read the tau data
    datafile = str('%stau/tau_50Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat' %(path,Nlos,z_name,logfX[l],xHI_mean[l],dvH,i))
    data = np.fromfile(str(datafile),dtype=np.float32)
    tau[i*Nlos:(i+1)*Nlos,:] = np.reshape(data,(Nlos,Nbins))

  tau_new = np.empty((Nlos_new,Nbins*extension_factor))

  #Splice the tau LOS
  for i in range(Nlos_new):
    tau_new[i][:] = np.concatenate(tau[LOS[i,:]])

  #Save the data
  array = tau_new
  array = array.astype('float32')
  array.tofile('%stau_long/tau_200Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d.dat' %(path,Nlos_new,z_name,logfX[l],xHI_mean[l],dvH),sep='')

  #Track the progress
  done = (l+1)/len(xHI_mean)
  if done>=done_perc:
    print('Done %.2f' % done)
    done_perc += 0.01

tf = time.time()
print('Time taken: '+str(tf-ti_whole))