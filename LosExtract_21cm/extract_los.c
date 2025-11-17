#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "global_vars.h"
#include "proto.h"

#include <omp.h>

/* Evaluate Gaussian within this range of sigma */
#define SIGMA 5.0

static double *exp_v2_b2;
static double dx_inv;

double ztime_file, fX_name, xHI_mean;
int Nlos_name;
char *model;
char *path;

/********************************************************************************/

int main(int argc, char **argv)
{

  int i,j;
  double start, end, tot_cpu_used;
  
  start = clock();
  double clock_start = omp_get_wtime();
  
  if(argc != 2)
    {
      fprintf(stderr, "\n\nIncorrect argument(s).  Specify:\n\n");
      fprintf(stderr, "<path>     (path to output files)\n");
      fprintf(stderr, "Usage: ./LosExtract <path>\n");
      fprintf(stderr, "\n\n");
      exit(0);
    }
  
  path = argv[1];

  //Parameters for file names
  ztime_file = 6.000;
  fX_name    = -2.;
  Nlos_name  = 10;
  dvH        = 250;
  xHI_mean   = 0.25;

  if(SIGMA > fabs(XMIN))
    {
      printf("\nSIGMA must be less than %.1f\n\n",fabs(XMIN));
      exit(0);
    }
   
  MakeLineProfileTable();
    
    //Loop over file parameters
    for(j=0; j<1; j++)
    {

      filenum = 0;
      
      //Loop over parts of each file (split by LOS)
      for(i=0; i<1; i++)
      {
        printf("\nReading los output at z=%.3f:, file%d:\n",ztime_file,filenum);
        read_los();
        
        printf("\nComputing optical depths...\n");
        compute_absorption();
        printf("Done.\n\n");
        
        write_tau();
        
        free_los_memory();
        
        filenum += 1;

        end = clock();
        tot_cpu_used = (end - start) / CLOCKS_PER_SEC;
        printf("\nTotal CPU time used %lf s\n",tot_cpu_used);

        double clock_end = omp_get_wtime();
        double clock_time = clock_end-clock_start;
        printf("Time taken: %g\n",clock_time);
        
      }

      /*ztime_file += 1.;*/
      /*fX_name += 0.2;*/
      /*xHI_mean += 0.08;*/
      
    }
       
  return(0);
}

/********************************************************************************/

void compute_absorption()
{
  
  double atime,vmax,vmax2,rscale,escale,drbin,Hz;
  double H0,rhoc,critH;
  double v2_b2, T_spin_inv;
  double k1_conv,k2_conv,k_21;
  double profile_H1,vdiff_H1;
  double u_H1[nbins],b_H1_inv[nbins],b2_H1_inv[nbins],tau_H1_dR[nbins];
#ifdef TAUWEIGHT
  double rhoker_Hw[nbins],tempker_H1w[nbins];
#endif
  double pcdone,pccount=0.0,iproc_done=0.0;
  double t,fhi,flow;
    
  long int i,j,iproc,convol_index,pixel_index,tint;
 
  //Whole equation can be found in Eq. 9 of Šoltinský et al. 2021, MNRAS 506, 5818
  
  //Compute constants used in the calculation
  atime  = 1.0/(1.0+ztime);
  rscale = (KPC*atime)/h100;        /* comoving kpc/h to cm */
  escale = 1.0e10;                     /* (km s^-1)^2 to (cm s^-1)^2 */
  drbin  = box100/(double)nbins; /* comoving kpc/h */
  
  Hz     = 100.0*h100*sqrt(omegam/(atime*atime*atime)+omegal); /* km s^-1 Mpc^-1 */
  vmax   = box100*Hz*rscale/MPC; /* box size km s^-1 */
  vmax2  = 0.5*vmax;
  H0     = 1.0e7/MPC; /* 100 km s^-1 Mpc^-1 in cgs */ 
  rhoc   = 3.0*(H0*h100)*(H0*h100)/(8.0*PI*GRAVITY); /* g cm^-3 */
  critH  = rhoc*omegab*Xh/(atime*atime*atime); /* g cm^-3*/
  
  k_21    = 3.0*PLANCK*pow(C,3.0)*A10_21/(32.0*PI*pow(NU_21,2.0)*BOLTZMANN);
  k1_conv = 2.0*BOLTZMANN/(HMASS*AMU);
  k2_conv = k_21*rscale*drbin*critH/(sqrt(PI)*HMASS*AMU);



  //Parallelize the computation using OpenMP
  /*omp_set_num_threads(4);*/
  #pragma omp parallel for num_threads(8) default(none) shared(velaxis,velker_H1,T_K_corrected,rhoker_H,x_H1_corrected,escale,k1_conv,k2_conv,nlos,nbins,vmax,vmax2,dx_inv,exp_v2_b2,pcdone,pccount,iproc_done,floatArray) private(iproc,convol_index,tau_H1,i,pixel_index,j,T_spin_inv,v2_b2,vdiff_H1,profile_H1,t,tint,fhi,flow,u_H1,b_H1_inv,b2_H1_inv,tau_H1_dR)
  for(iproc=0; iproc<nlos; iproc++)
    {
      
      /*printf("blem_%d\n",iproc);*/
      for(j=0; j<nbins; j++)
	{
	  convol_index =  j + nbins*iproc;
	  
	  T_spin_inv = 1.0/T_K_corrected[convol_index]; /* assumes Ts=TK */
	  /*T_spin_inv = 1.0/T_S_corrected[convol_index];*/

    //Do the convolution over line profile
#ifdef NOPECVEL
  	u_H1[j] = velaxis[j]; /* km s^-1 */
#else
	  u_H1[j] = velaxis[j] + velker_H1[convol_index]; /* km s^-1 */
#endif	  
	  b_H1_inv[j]  = 1.0/sqrt(k1_conv*T_K_corrected[convol_index]); /*Doppler parameter in cm s^-1 */
	  b2_H1_inv[j] = b_H1_inv[j]*b_H1_inv[j]*escale;  /* (km s^-1)^-2 */
	  
	  tau_H1_dR[j] = k2_conv*rhoker_H[convol_index]*x_H1_corrected[convol_index]*b_H1_inv[j]*T_spin_inv;

#ifdef TAUWEIGHT
	  rhoker_Hw[j]    = rhoker_H[convol_index];
	  tempker_H1w[j]  = T_K_corrected[convol_index];
#endif
	  
      	}
      	
      /*omp_set_num_threads(4);*/
      /*#pragma omp parallel for num_threads(4) default(none) shared(iproc,velaxis,u_H1,b2_H1_inv,exp_v2_b2,tau_H1_dR,nbins,vmax,vmax2,dx_inv,floatArray) private(tau_H1,i,pixel_index,j,v2_b2,vdiff_H1,profile_H1,t,tint,fhi,flow)
      */
      for(i=0; i<nbins; i++)
	{
	     
	  pixel_index =  i + nbins*iproc;
	  
  	  tau_H1 = 0;
	  
	  /*omp_set_num_threads(4);*/
	  /*#pragma omp parallel for num_threads(4) default(none) shared(tau_H1,i,pixel_index,iproc,velaxis,u_H1,b2_H1_inv,exp_v2_b2,tau_H1_dR,nbins,vmax,vmax2,dx_inv,floatArray) private(j,v2_b2,vdiff_H1,profile_H1,t,tint,fhi,flow)
	  */
	  for(j=0; j<nbins; j++)
	    {
	      
	      vdiff_H1 = fabs(velaxis[i] - u_H1[j]);
	      
	      /*if (vdiff_H1 > vmax2) vdiff_H1 = vmax - vdiff_H1;*/
	      
	      v2_b2 = vdiff_H1*vdiff_H1*b2_H1_inv[j];
	      
	      t    = (v2_b2 - XMIN) * dx_inv;
	      tint = (int)t;
	      fhi  = t - tint;
	      flow = 1 - fhi;
	      
	      profile_H1 = v2_b2 < SIGMA ? flow*exp_v2_b2[tint]+fhi*exp_v2_b2[tint+1] : 0.0;
	      
	      if (profile_H1 > 0) tau_H1 += tau_H1_dR[j]*profile_H1;
	      

#ifdef TAUWEIGHT
	      /* HI optical depth weighted quantities */
	      rho_tau_H1[pixel_index]  += rhoker_Hw[j]*tau_H1_dR[j]*profile_H1;
	      temp_tau_H1[pixel_index] += tempker_H1w[j]*tau_H1_dR[j]*profile_H1;
#endif	      
	    }

	  floatArray[pixel_index] = (float) tau_H1;   
	    
#ifdef TAUWEIGHT	  
	  rho_tau_H1[pixel_index]  /= tau_H1[pixel_index];
	  temp_tau_H1[pixel_index] /= tau_H1[pixel_index];
#endif
	    
	
	/*printf("Thread: %d\n",omp_get_thread_num());
	printf("Num of threads: %d\n",omp_get_num_threads());*/
	}
      
      iproc_done += 1.;
      pcdone = 100.0*iproc_done/((double)nlos-1.0);
      if(pcdone >= pccount)
  	{
  	  printf("%3.2f%%\n",pcdone);
  	  pccount += 10.0;
  	}
      
    }
    
}
 
/********************************************************************************/

void read_los()
{
  char fname[400];
  FILE *input;
  

  sprintf(fname, "%slos_regrid/los_50Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat",path,Nlos_name,ztime_file,fX_name,xHI_mean,dvH,filenum);

  if(!(input=fopen(fname,"rb")))
    {
      printf("can't open file `%s`\n\n",fname);
      exit(0);
    }
  
  fread(&ztime,sizeof(float),1,input);          // redshift
  fread(&omegam,sizeof(float),1,input);         // Omega_matter
  fread(&omegal,sizeof(float),1,input);         // Omega_Lambda
  fread(&omegab,sizeof(float),1,input);         // Omega_baryon
  fread(&h100,sizeof(float),1,input);           // Hubble constant parameter h
  fread(&box100,sizeof(float),1,input);         // box size in comoving Mpc/h
  fread(&Xh,sizeof(float),1,input);             // hydrogen fraction by mass
  fread(&nbins_read,sizeof(float),1,input);     // number of bins per LOS
  fread(&nlos_read,sizeof(float),1,input);      // number of LOS
  fread(&fX_name_read,sizeof(float),1,input);   // log fX
  fread(&ioneff_read,sizeof(float),1,input);    // ionizing efficiency
  fread(&xHI_mean_read,sizeof(float),1,input);  // mean neutral fraction
  
  nbins = (int) nbins_read;
  nlos  = (int) nlos_read;
  
  printf("z         = %f\n",ztime);
  printf("logfX     = %f\n",fX_name_read);
  printf("ion_eff   = %f\n",ioneff_read);
  printf("<xHI>     = %f\n",xHI_mean_read);
  printf("omegaM    = %f\n",omegam);
  printf("omegaL    = %f\n",omegal);
  printf("omegab    = %f\n",omegab);
  printf("h100      = %f\n",h100);
  printf("box100    = %f\n",box100);
  printf("Xh        = %f\n",Xh);
  printf("nbins     = %d\n",nbins);
  printf("nlos      = %d\n",nlos);

  allocate_los_memory();
  
  fread(posaxis,sizeof(float),nbins,input);       /* pixel positions, comoving kpc/h */
  fread(velaxis,sizeof(float),nbins,input);       /* pixel positions, km s^-1 */
  
  fread(rhoker_H,sizeof(float),nbins*nlos,input);       /* gas overdensity, Delta=rho/rho_crit */
  fread(x_H1_corrected,sizeof(float),nbins*nlos,input); /* n_HI/n_H */
  fread(T_K_corrected,sizeof(float),nbins*nlos,input);  /* T [K], HI weighted */
  /*fread(T_S_corrected,sizeof(float),nbins*nlos,input);*/  /* T spin, HI weighted */
  fread(velker_H1,sizeof(float),nbins*nlos,input);      /* v_pec [km s^-1], HI weighted */

  fclose(input);  


}

/********************************************************************************/

void write_tau()
{
  char fname[400];
  FILE *output;
   
  sprintf(fname, "%s/tau/tau_50Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat",path,nlos,ztime_file,fX_name,xHI_mean,dvH,filenum);

  if(!(output=fopen(fname,"wb")))
    {
      printf("can't open file `%s`\n\n",fname);
      exit(0);
    }
  fwrite(floatArray,sizeof(float),nbins*nlos,output);
  fclose(output);
 
#ifdef TAUWEIGHT
  sprintf(fname, "%s/weighted_tau%d_n%d_z%.3f_fX%.2f_dv%d_file%d.dat",path,nbins,nlos,ztime_file,fX_name,dvH,filenum);
  if(!(output=fopen(fname,"wb")))
    {
      printf("can't open file `%s`\n\n",fname);
      exit(0);
    }
  fwrite(rho_tau_H1,sizeof(float),nbins*nlos,output);
  fwrite(temp_tau_H1,sizeof(float),nbins*nlos,output);
  fclose(output);
#endif  
  
}

/********************************************************************************/

void allocate_los_memory()
{  
   posaxis      = (float *)calloc(nbins, sizeof(float));
  if(NULL==posaxis){free(posaxis);printf("Memory allocation failed in extract_spectra.c\n");exit(0);}
  velaxis      = (float *)calloc(nbins, sizeof(float));
  if(NULL==velaxis){free(velaxis);printf("Memory allocation failed in extract_spectra.c\n");exit(0);}
  
  rhoker_H     = (float *)calloc(nlos*nbins, sizeof(float));
  if(NULL==rhoker_H){free(rhoker_H);printf("Memory allocation failed in extract_spectra.c\n");exit(0);}  
  velker_H1    = (float *)calloc(nlos*nbins, sizeof(float));
  if(NULL==velker_H1){free(velker_H1);printf("Memory allocation failed in extract_spectra.c\n");exit(0);}
   x_H1_corrected = (float *)calloc(nlos*nbins, sizeof(float));
  if(NULL==x_H1_corrected ){free(x_H1_corrected );printf("Memory allocation failed in extract_spectra.c\n");exit(0);}
  T_K_corrected  = (float *)calloc(nlos*nbins, sizeof(float));
  if(NULL==T_K_corrected ){free(T_K_corrected );printf("Memory allocation failed in extract_spectra.c\n");exit(0);}
  /*T_S_corrected  = (float *)calloc(nlos*nbins, sizeof(float));
  if(NULL==T_S_corrected ){free(T_S_corrected );printf("Memory allocation failed in extract_spectra.c\n");exit(0);}*/
  
  floatArray      = (float *)calloc(nlos*nbins, sizeof(float));
  if(NULL==floatArray ){free(floatArray );printf("Memory allocation failed in extract_spectra.c\n");exit(0);}  
  
#ifdef TAUWEIGHT
  rho_tau_H1       = (float *)calloc(nlos*nbins, sizeof(float));
  if(NULL==rho_tau_H1){free(rho_tau_H1);printf("Memory allocation failed in extract_spectra.c\n");exit(0);}
  temp_tau_H1       = (float *)calloc(nlos*nbins, sizeof(float));
  if(NULL==temp_tau_H1){free(temp_tau_H1);printf("Memory allocation failed in extract_spectra.c\n");exit(0);}
#endif
 
  
  
  
}

/********************************************************************************/

void free_los_memory()
{  
  free(posaxis);
  free(velaxis);

  free(rhoker_H);  
  free(velker_H1);
  free(x_H1_corrected);
  free(T_K_corrected);
  /*free(T_S_corrected);*/

  free(floatArray);

#ifdef TAUWEIGHT
  free(rho_tau_H1);
  free(temp_tau_H1);
#endif
  
}

/********************************************************************************/
/* Look-up table for Gaussian line profile */

void MakeLineProfileTable(void)
{
  double x, dx;
  int i;
  
  exp_v2_b2 = (double *)calloc(NXTAB+1, sizeof(double));

  if(NULL==exp_v2_b2)
    {
      free(exp_v2_b2);
      printf("Memory allocation failed in extract_spectra.c\n");
      exit(0);
    }

  dx     = (fabs(XMIN)-XMIN) / NXTAB;
  dx_inv = 1.0 / dx;
  
  for(i = 0; i <= NXTAB; i++)
    {
      x = XMIN + dx*i;
      exp_v2_b2[i] = exp(-x);
    }
}

/********************************************************************************/
