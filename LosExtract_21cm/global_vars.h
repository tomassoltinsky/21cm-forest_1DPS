float nbins_read,nlos_read;

float ztime,omegam,omegal,omegab,h100,box100,Xh;
float fX_name_read,ioneff_read,xHI_mean_read;
float *rhoker_H,*posaxis,*velaxis;
float *rhoker_H1,*velker_H1,*x_H1_corrected,*T_K_corrected,tau_H1,*floatArray;
/*float *T_S_corrected;*/

int nbins,nlos;
int dvH, filenum;



#ifdef TAUWEIGHT
float *rho_tau_H1, *temp_tau_H1;
#endif

/* Numbers */
#define  PI    3.14159265358979
#define  GAMMA (5.0/3.0)

/* Physical constants (cgs units) */
/* See http://physics.nist.gov/cuu/Constants/index.html */ 
#define  GRAVITY      6.67384e-8
#define  BOLTZMANN    1.3806488e-16
#define  C            2.99792458e10
#define  AMU          1.66053886e-24 /* 1 a.m.u */
#define  MPC          3.08568025e24
#define  KPC          3.08568025e21
#define  SOLAR_MASS   1.989e33
#define  ELECTRONVOLT 1.602176565e-12
#define  PROTONMASS   1.672621777e-24
#define  PLANCK       6.62606957e-27

/* Atomic data */
#define  HMASS        1.00794  /* Hydrogen mass in a.m.u. */
#define  A10_21       2.85e-15  /* s^-1 */
#define  NU_21        1420.405751e6  /*  Hz */


/* Line profile table */
#define XMIN -10.0
#define NXTAB 10000
