#ifndef FDTD_COMMON_H
#define FDTD_COMMON_H
#define PI 3.141592653589793238
#define alpha_max 0.01
#define alpha_min 0.000
#define eps0 8.85418e-12
#define sigma_factor 1.0
#define ncells 10
#define mu0 (PI*4e-7)
#define center_freq (5e9)
#define eta0 (sqrt(mu0/eps0))
#define c0 (1.0/sqrt(mu0*eps0))
#define dt (dx/c0/2)// dx/c0/2
#define domain_size 0.18
#define dx (0.001)
#define NF2FFdistfromboundary ((int)floor((3.2*breast_radius/dx)))
#define source_position 0.5
#define dy (0.001)
#define number_of_time_steps 5000
#define f1x (nx/2 - 150)       
#define f2x (nx/2+150) 
#define f1y (ny/2)
#define f2y (ny/2)
//#define nx ((int)ceil(domain_size/dx))
//#define ny ((int)ceil(domain_size/dy))
#define nx ((int)ceil(12.7*breast_radius/dx))
#define ny ((int)ceil(12.7*breast_radius/dy))
#define d (10*dx)
#define npml 2
#define kmax 10
#define numberofexcitationangles 4
#define isPW 1
#define isscattering 1
#define sigma_max_pml (3/(200*PI*dx))
#define size_NF2FF_total (2*nx-8*NF2FFdistfromboundary+2*ny-4)
#define startfrequency (1e9)
#define stopfrequency (1e10)
#define deltafreq ((stopfrequency-startfrequency)/(numberoffrequencies-1))
#define numberoffrequencies 10
#define size_cjzy ((nx-2*NF2FFdistfromboundary-2)*(numberoffrequencies))
#define size_cjzx ((ny-2*NF2FFdistfromboundary)*(numberoffrequencies))
#define size_y_side (nx-2*NF2FFdistfromboundary-2)
#define size_x_side (ny-2*NF2FFdistfromboundary)
#define numberofobservationangles  100
#define t0 (sqrt(20.0)*tau) // t0 = sqrt(20)*tau
#define l0 (nx*dx/2-breast_radius) 
#define pwidth 10
#define nc 20 // 20 cells per wavelength
#define  fmax  (c0/(nc*dx))// change if dy is bigger though now they're the same  fmax is the highest frequency this program can handle
#define tau (3.3445267e-11) // float ta bu = sqrt(2.3)*nc*dx/(PI*c0*1/sqrt(eps_r_MAX));  from a calculation of fmax.
//#define tau (5.288161e-11)
#define target_x (nx/2+15)//105 is breast_radius / dx
#define target_y (ny/2-15)
#define source_x (nx/2)      //(target_x-105-80)
#define source_y (ny/2)
#define breast_radius 0.0315 //87.535 mm  .  Sample size = 1.
#define tumor_size (0.01)
#define relaxation_time (7e-12)
#define Kp ((1-dt/(2*relaxation_time))/(1+dt/(2*relaxation_time))) // Parameter for (FD)2TD.  Using Polarization current formulation
#define numpatches 3
/**
 * I made getCell a macro which should speed things up 
 * a bit, and make the code simpler
 */
#define getCell(x,y,size) ((x) + ((y) * (size)))
#define dgetCell(x,y,size) ((x) + ((y) * (size)))

int CPUgetxfromthreadIdNF2FF(int index);

int CPUgetyfromthreadIdNF2FF(int index);

float calc_incident_power(float freq);

float fitness(float* D,int max_index, float* measurement);

#endif
#ifndef FDTD_COMMON_H
#define FDTD_COMMON_H

#define GL_GLEXT_PROTOTYPES
#define PI 3.141592653589793238
#define alpha_max 0.01
#define alpha_min 0.000
#define eps0 8.85418e-12
#define sigma_factor 1.0
#define ncells 10
#define mu0 (PI*4e-7)
#define center_freq (5e9)
#define eta0 (sqrt(mu0/eps0))
#define c0 (1.0/sqrt(mu0*eps0))
#define dt (dx/c0/2)// dx/c0/2
#define domain_size 0.18
#define dx (0.001)
#define NF2FFdistfromboundary ((int)floor((3.2*breast_radius/dx)))
#define source_position 0.5
#define dy (0.001)
#define number_of_time_steps 3000
#define f1x (nx/2 - 150)       
#define f2x (nx/2+150) 
#define f1y (ny/2)
#define f2y (ny/2)
//#define nx ((int)ceil(domain_size/dx))
//#define ny ((int)ceil(domain_size/dy))
#define nx ((int)ceil(12.7*breast_radius/dx))
#define ny ((int)ceil(12.7*breast_radius/dy))
#define d (10*dx)
#define npml 2
#define kmax 10
#define numberofexcitationangles 4
#define isPW 1
#define isscattering 1
#define HANDLE_ERROR( err ) err
#define sigma_max_pml (3/(200*PI*dx))
#define size_NF2FF_total (2*nx-8*NF2FFdistfromboundary+2*ny-4)
#define startfrequency (1e9)
#define stopfrequency (1e10)
#define deltafreq ((stopfrequency-startfrequency)/(numberoffrequencies-1))
#define numberoffrequencies 10
#define size_cjzy ((nx-2*NF2FFdistfromboundary-2)*(numberoffrequencies))
#define size_cjzx ((ny-2*NF2FFdistfromboundary)*(numberoffrequencies))
#define size_y_side (nx-2*NF2FFdistfromboundary-2)
#define size_x_side (ny-2*NF2FFdistfromboundary)
#define numberofobservationangles  100
#define t0 (sqrt(20.0)*tau) // t0 = sqrt(20)*tau
#define l0 (nx*dx/2-breast_radius) 
#define pwidth 10
#define nc 20 // 20 cells per wavelength
#define  fmax  (c0/(nc*dx))// change if dy is bigger though now they're the same  fmax is the highest frequency this program can handle
#define tau (3.3445267e-11) // float ta bu = sqrt(2.3)*nc*dx/(PI*c0*1/sqrt(eps_r_MAX));  from a calculation of fmax.
//#define tau (5.288161e-11)
#define target_x (nx/2+15)//105 is breast_radius / dx
#define target_y (ny/2-15)
#define source_x (nx/2)      //(target_x-105-80)
#define source_y (ny/2)
#define breast_radius 0.0315 //87.535 mm  .  Sample size = 1.
#define tumor_size (0.01)
#define relaxation_time (7e-12)
#define Kp ((1-dt/(2*relaxation_time))/(1+dt/(2*relaxation_time))) // Parameter for (FD)2TD.  Using Polarization current formulation
#define numpatches 3
/**
 * I made getCell a macro which should speed things up 
 * a bit, and make the code simpler
 */
#define getCell(x,y,size) ((x) + ((y) * (size)))
#define dgetCell(x,y,size) ((x) + ((y) * (size)))

int CPUgetxfromthreadIdNF2FF(int index);

int CPUgetyfromthreadIdNF2FF(int index);

float calc_incident_power(float freq);

float fitness(float* D,int max_index, float* measurement);

#endif
