#Code for regridding the 1D LOS data generated from 21cmFAST coeval boxes. 
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import constants

Nlos_new = int(sys.argv[1]) #Number of LOS to be outputted in a single file. The total number of LOS in the original data is Nlos.
dvH = float(sys.argv[2])    #Velocity resolution in m/s. This should be smaller than Doppler parameter b (in m/s) to properly convolve over the line profile in the next step.
z_name = float(sys.argv[3]) #Redshift

Lbox = 50   #Size of the simulated box in cMpc
ngrid = 256 #Number of pixels in the simulation box on each side
Nlos = 1000 #Total number of LOS in the original data

#IGM model parameters
xHI_mean_name = [0.80,0.11]#,0.52,0.80,0.11]
fX_name       = [-3.0,-3.0]#,-2.0,-1.0,-1.0]

for j in range(len(xHI_mean_name)):

    datafile = str('datasets/21cmFAST_los/los/los_%dMpc_%d_n%d_z%.3f_fX%.1f_xHI%.2f.dat' % (Lbox,ngrid,Nlos,z_name,fX_name[j],xHI_mean_name[j]))
    data = np.fromfile(str(datafile),dtype=np.float32)

    #cosmological parameters
    z = data[0]		      #redshift
    omega_0 = data[1]	  #matter density
    omega_L = data[2]	  #vacuum density
    omega_b = data[3]	  #baryon density
    h = data[4]		      #small h (Hubble parameter)
    box_size = data[5]  #box size in Mpc
    X_H = data[6]	      #primordial hydrogen fraction
    logfX = data[9]     #X-ray efficiency of galaxies (log10(f_X))
    ioneff = data[10]   #Ionizing efficiency of galaxies (zeta)
    xHI_mean = data[11] #mean neutral hydrogen fraction

    print('ION_eff=%.2f, <x_HI>=%.5f, log(f_X)=%.2f' % (ioneff,xHI_mean,logfX))

    #line-of-sight information
    Nbins = int(data[7])	#number of pixels/cells/bins in one line-of-sight
    Nlos = int(data[8]) 	#number of lines-of-sight
    x_initial = 12

    #read axis information
    pos_axis = data[(x_initial):(x_initial+Nbins)]	#position along LoS in kpc
    vH = data[(x_initial+Nbins):(x_initial+2*Nbins)]#Hubble velocity along LoS in km/s

    #extract normalized density (rho/rho_mean), neutral fraction, temperature of HI, peculiar velocity of HI for each pixel in each LoS
    Delta = data[(x_initial+2*Nbins):(x_initial+2*Nbins+Nlos*Nbins)]
    Delta_ori = np.reshape(Delta,(Nlos,Nbins))

    xHI = data[(x_initial+2*Nbins+1*Nlos*Nbins):(x_initial+2*Nbins+2*Nlos*Nbins)]
    xHI_ori = np.reshape(xHI,(Nlos,Nbins))

    TK = data[(x_initial+2*Nbins+2*Nlos*Nbins):(x_initial+2*Nbins+3*Nlos*Nbins)]
    TK_ori = np.reshape(TK,(Nlos,Nbins))

    vpec = data[(x_initial+2*Nbins+3*Nlos*Nbins):(x_initial+2*Nbins+4*Nlos*Nbins)]
    vpec_ori = np.reshape(vpec,(Nlos,Nbins)) 

    Gamma_HI = data[(x_initial+2*Nbins+4*Nlos*Nbins):(x_initial+2*Nbins+5*Nlos*Nbins)]
    Gamma_HI_ori = np.reshape(Gamma_HI,(Nlos,Nbins)) 

    TK_min = np.amin(TK)
    print('TK_min = %.2f' % TK_min)
    #Compute the smallest Doppler parameter b in m/s
    b_doppler = np.sqrt(2*constants.k_B*1e-7*TK_min/(constants.m_p*1e-3))
    print('Doppler parameter b = %.2fm/s' % b_doppler)


    #Regrid the LOS data to a new velocity resolution dvH (in m/s)
    a = 1./(1.+z)
    r_conv = a/1000./h
    omega_k = 1.-omega_0-omega_L
    H_regrid = 100.*h*np.sqrt(omega_L+omega_0*(1+z)**3+omega_k*(1+z)**2)
    vmax = H_regrid*box_size*r_conv
    vH_regrid = np.arange(0.,vmax,dvH/1000.)
    Nbins_regrid = int(vmax/dvH*1000+1)
    vH_regrid = np.linspace(0,vmax,num=Nbins_regrid,endpoint=True)
    pos_axis_regrid = np.linspace(0,box_size,num=Nbins_regrid,endpoint=True)
    vH = np.append(vH,vmax)

    #Interpolate linearly to the new velocity grid. Will use the following weights.
    index_interpol = np.digitize(vH_regrid,vH,right=True)

    vH_low  = vH[index_interpol-1]
    vH_high = vH[index_interpol]

    w_low = 1./(vH_regrid-vH_low)/(1./(vH_high-vH_regrid)+1./(vH_regrid-vH_low))
    w_high = 1./(vH_high-vH_regrid)/(1./(vH_high-vH_regrid)+1./(vH_regrid-vH_low))



    numfiles = int(Nlos/Nlos_new) #How many files will be generated after regridding the original data
    for filenum in range(numfiles*0,2):
      
      print(str(filenum+1)+'/'+str(numfiles))

      Delta = Delta_ori[filenum*Nlos_new:(filenum+1)*Nlos_new]
      xHI = xHI_ori[filenum*Nlos_new:(filenum+1)*Nlos_new]
      TK = TK_ori[filenum*Nlos_new:(filenum+1)*Nlos_new]
      vpec = vpec_ori[filenum*Nlos_new:(filenum+1)*Nlos_new]
      Gamma_HI = Gamma_HI_ori[filenum*Nlos_new:(filenum+1)*Nlos_new]

      Delta_regrid = np.empty((Nlos_new,Nbins_regrid))
      xHI_regrid = np.empty((Nlos_new,Nbins_regrid))
      TK_regrid = np.empty((Nlos_new,Nbins_regrid))
      vpec_regrid = np.empty((Nlos_new,Nbins_regrid))
      Gamma_HI_regrid = np.empty((Nlos_new,Nbins_regrid))

      for i in range(Nlos_new):
        
        Delta_los = np.append(Delta[i,:],Delta[i,0])
        Delta_regrid[i,:] = w_low*Delta_los[index_interpol-1]+w_high*Delta_los[index_interpol]
        Delta_regrid[i,0] = Delta_los[0]

        xHI_los = np.append(xHI[i,:],xHI[i,0])
        xHI_regrid[i,:] = w_low*xHI_los[index_interpol-1]+w_high*xHI_los[index_interpol]
        xHI_regrid[i,0] = xHI_los[0]
        
        TK_los = np.append(TK[i,:],TK[i,0])
        TK_regrid[i,:] = w_low*TK_los[index_interpol-1]+w_high*TK_los[index_interpol]
        TK_regrid[i,0] = TK_los[0]

        vpec_los = np.append(vpec[i,:],vpec[i,0])
        vpec_regrid[i,:] = w_low*vpec_los[index_interpol-1]+w_high*vpec_los[index_interpol]
        vpec_regrid[i,0] = vpec_los[0]
        vpec_regrid[i,-1] = vpec_los[0]
        
        Gamma_HI_los = np.append(Gamma_HI[i,:],Gamma_HI[i,0])
        Gamma_HI_regrid[i,:] = w_low*Gamma_HI_los[index_interpol-1]+w_high*Gamma_HI_los[index_interpol]
        Gamma_HI_regrid[i,0] = Gamma_HI_los[0]


      
      Delta_regrid = np.reshape(Delta_regrid,Nlos_new*Nbins_regrid)
      xHI_regrid = np.reshape(xHI_regrid,Nlos_new*Nbins_regrid)
      TK_regrid = np.reshape(TK_regrid,Nlos_new*Nbins_regrid)
      vpec_regrid = np.reshape(vpec_regrid,Nlos_new*Nbins_regrid)
      Gamma_HI_regrid = np.reshape(Gamma_HI_regrid,Nlos_new*Nbins_regrid)


      #Remove the last element of pos_axis_regrid if it is longer than vH_regrid.
      if len(pos_axis_regrid)>len(vH_regrid):
        pos_axis_regrid = pos_axis_regrid[:-1]


      #Save the data in a single binary file with the following format:
      array = np.array([z,omega_0,omega_L,omega_b,h,box_size,X_H,Nbins_regrid,Nlos_new,logfX,ioneff,xHI_mean])
      array = np.append(array,pos_axis_regrid)
      array = np.append(array,vH_regrid)
      array = np.append(array,Delta_regrid)
      array = np.append(array,xHI_regrid)
      array = np.append(array,TK_regrid)
      array = np.append(array,vpec_regrid)
      array = np.append(array,Gamma_HI_regrid)
      array = array.astype('float32')
      array.tofile('datasets/21cmFAST_los/los_regrid/los_%dMpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat' % (box_size/1000,Nlos_new,z_name,logfX,xHI_mean,dvH,filenum),sep='',format='float32')