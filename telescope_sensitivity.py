#Code to generate plot of telescope sensitivity (Aeff/Tsys) for LOFAR, uGMRT, SKA1-low and SKA2

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as scisi
from astropy.convolution import convolve, Box1DKernel
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec

import openpyxl

#Functions to read data from excel table
def excel_column(data_name,n_col):

    wb = openpyxl.load_workbook(data_name)
    sheet = wb.active
    N_rows = sheet.max_row-1

    col = np.empty(N_rows)

    for i_row in range(N_rows):
        col[i_row] = sheet.cell(row=i_row+2,column=n_col).value

    return col

#Function to read telescope sensitivity (Aeff/Tsys) from excel table
def sens_read(frequency,telescope):
  
  #read uGMRT Aeff/Tsys from Fig. 8 in Braun et al. 2019 and stored in sensitivity folder
  filename = 'sensitivity/sens_%s.xlsx' % telescope
  freq_0 = excel_column(filename,1)*1000
  ATsys_0 = excel_column(filename,2)  

  index_freq = np.digitize(frequency,freq_0,right=True)	#Interpolate in the density (similar to redshift)

  freq_low  = freq_0[index_freq-1]
  freq_high = freq_0[index_freq]

  w_low = 1./(frequency-freq_low)/(1./(freq_high-frequency)+1./(frequency-freq_low))
  w_low = np.where(frequency==freq_low,1.,w_low)
  w_low = np.where(frequency==freq_high,0.,w_low)

  w_high = 1./(freq_high-frequency)/(1./(freq_high-frequency)+1./(frequency-freq_low))
  w_high = np.where(frequency==freq_high,1.,w_high)
  w_high = np.where(frequency==freq_low,0.,w_high)

  ATsys  = w_low*ATsys_0[index_freq-1]+w_high*ATsys_0[index_freq]

  return ATsys


#Plot the sensitivities as a function of frequency
fsize = 24
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)
ax = plt.subplot(gs[0,0])

#Frequency range relevant for the 21-cm forest observations
#Limits given by the most distant known RLQSO (z~7) and the end of reionization $z~5.5$
z_min = 1420./(1+7.)
z_max = 1420./(1+5.5)
freq = np.append(np.arange(z_min,z_max,10),z_max)
print('Frequencies relevant for 21-cm forest: %.2f-%.2f MHz' % (z_min, z_max))

'''
telescopes = ['LOFAR','uGMRT']
colour = ['royalblue','fuchsia']
y_label = [62,90]
rot = [-17,45]
y_lim = [0,120]
'''
telescopes = ['LOFAR','uGMRT','SKA1-low']
colour = ['royalblue','fuchsia','darkorange']
y_label = [70,110,520]
x_label = [216,180,180]
rot = [0,8,8]
y_lim = [0,700]

for i in range(len(telescopes)):

    sens = sens_read(freq,telescopes[i])

    ax.plot(freq,sens,linewidth=(i+1)*2.,color=colour[i])
    ax.text(x_label[i],y_label[i],'%s' % telescopes[i],rotation=rot[i],fontsize=fsize-3)
#plt.legend(frameon=False,loc='upper left',fontsize=fsize,ncols=4)

#Indicate frequency range relevant for 21-cm forest observations
v_band = [z_min,z_max]
ax.fill_between(v_band,[y_lim[0],y_lim[0]],[y_lim[1],y_lim[1]],alpha=0.25,color='red')

#Indicate frequency ranges of TGSS - Intema et al. 2017, A&A , 598, A78
paper = 'TGSS'
v_band = [140,156]
ax.fill_between(v_band,[y_lim[0],y_lim[0]],[y_lim[1],y_lim[1]],alpha=0.25,color='forestgreen')
ax.text(v_band[0]+2,300,paper,rotation=90,fontsize=fsize-3)
paper = '(Intema+17)'
ax.text(v_band[0]+7.5,200,paper,rotation=90,fontsize=fsize-3)

#Indicate frequency ranges of Band 3 of uGMRT used in Gloudemans et al. 2023, A&A , 678, A128
paper = 'Band 3'
v_band = [250,300]
ax.fill_between(v_band,[y_lim[0],y_lim[0]],[y_lim[1],y_lim[1]],alpha=0.25,color='grey')
ax.text(v_band[0]+17,275,paper,rotation=90,fontsize=fsize-3)
paper = '(Gloudemans+23)'
ax.text(v_band[0]+24.5,100,paper,rotation=90,fontsize=fsize-3)

ax.set_xlim(120,300)
ax.set_ylim(y_lim[0],y_lim[1])
#ax.set_yticks(np.arange(0.98,1.011,0.01))
#ax.set_yscale('log')
ax.set_xlabel(r'$\nu\,\rm [MHz]$', fontsize=fsize)
ax.set_ylabel(r'$A_{\mathrm{eff}}/T_{\mathrm{sys}}\,\rm [m^2K^{-1}]$', fontsize=fsize)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

#Upper x-axis with redshift corresponding to the frequency
z = np.array([4,4.5,5.5,7,9])
z_posit = (1420/(1+z)-120)/(300-120)
axup = ax.twiny()
axup.set_xlabel(r'$z$', fontsize=fsize)
axup.set_xticks(z_posit)
axup.set_xticklabels(z)
axup.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)

z = np.arange(3.8,10.81,0.1)
z_posit = (1420/(1+z)-120)/(300-120)
axupmin = ax.twiny()
axupmin.set_xticks(z_posit)
axupmin.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=2,labeltop=False)

plt.tight_layout()
plt.savefig('plots/telescope_sensitivity.png')
plt.show()