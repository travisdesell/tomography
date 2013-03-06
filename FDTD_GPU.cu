//#define GLEW_STATIC
//#pragma comment(lib,"glew32.lib")
//#include <windows.h>
//#include <gl/glew.h>
//#include <glut.h>
#include <complex>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <fstream>
#include <cuda.h>
//#include "stdafx.h"
#include <iomanip>
#include <time.h>
//#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
//#include <cuComplex.h>
#include <vector>
#include <math_functions.h>
//#include "EasyBMP.h"
//#include "EasyBMP_DataStructures.h"
//#include "EasyBMP_VariousBMPutilities.h"

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
#define NF2FFdistfromboundary 100
#define source_position 0.5
#define dy (0.001)
#define number_of_time_steps 3000
#define f1x (nx/2 - 150)       
#define f2x (nx/2+150) 
#define f1y (ny/2)
#define f2y (ny/2)
//#define nx ((int)ceil(domain_size/dx))
//#define ny ((int)ceil(domain_size/dy))
#define nx 400
#define ny 400
#define d (10*dx)
#define npml 2
#define kmax 10
#define isPW 1
#define isscattering 1
#define HANDLE_ERROR( err ) err
#define sigma_max_pml (3/(200*PI*dx))
#define size_NF2FF_total (2*nx-8*NF2FFdistfromboundary+2*ny-4)
#define size_cjzy (nx-2*NF2FFdistfromboundary-2)
#define size_cjzx (ny-2*NF2FFdistfromboundary)
#define numberofobservationangles  60
#define t0 (sqrt(20.0)*tau) // t0 = sqrt(20)*tau
#define l0 (nx*dx/2-breast_radius) 
#define pwidth 10
#define nc 20 // 20 cells per wavelength
#define  fmax  (c0/(nc*dx))// change if dy is bigger though now they're the same  fmax is the highest frequency this program can handle
#define tau (3.3445267e-11) // float ta bu = sqrt(2.3)*nc*dx/(PI*c0*1/sqrt(eps_r_MAX));  from a calculation of fmax.
//#define tau (5.288161e-11)
#define target_x (nx/2+105+25)//105 is breast_radius / dx
#define target_y (ny/2)
#define source_x (nx/2)      //(target_x-105-80)
#define source_y (ny/2)
#define breast_radius 0.0315 //87.535 mm  .  Sample size = 1.
#define tumor_size (20)
//#include <unistd.h>
//const cuComplex jcmpx (0.0, 1.0);
/*static void HandleError( cudaError_t err, const char *file,  int line ) {
  if (err != cudaSuccess) {
  printf( "%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );
  exit( EXIT_FAILURE );
  }
  }*/


//__constant__ float*dev_Ceze,*dev_Cezhy,*dev_Cezhx,*dev_Cezj,*dev_Jz,*dev_Chyh,*dev_Chxh,*dev_Chyez,*dev_Chxez,*dev_bex,*dev_bey,*dev_aex,*dev_aey,*dev_bmy,*dev_bmx,*dev_amy,*dev_amx,*dev_C_Psi_ezy,
//*dev_C_Psi_ezx,*dev_C_Psi_hxy,*dev_C_Psi_hyx;
struct cuComplex {
    float   r;
    float   i;
    __host__  __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __host__ __device__ cuComplex(float a): r(a), i(0) {}
    float magnitude2( void ) { return r * r + i * i; }
    __host__  __device__  cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __host__ __device__ cuComplex operator*(const float& a){
        return cuComplex(r*a,i*a);
    }

    __host__  __device__  cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
    __host__ __device__ cuComplex operator+(const float& a){
        return cuComplex(r+a,i);
    }
    __host__ __device__ void operator+=(const float& f){
        r += f;
    }
    __host__ __device__ void operator+=(const cuComplex& C);
    cuComplex();
};

__host__ __device__ cuComplex operator*(const float &f, const cuComplex &C)
{

    return cuComplex(C.r*f,C.i*f);
}

__host__ __device__ void cuComplex::operator+=(const cuComplex& C)
{
    r +=C.r;
    i += C.i;
}

__host__ __device__ float cuabs(cuComplex x)
{
    return sqrt(x.i*x.i + x.r*x.r);
}

__host__ __device__ cuComplex cuexp(cuComplex arg)
{
    cuComplex res(0,0);
    float s, c;
    float e = expf(arg.r);
    sincosf(arg.i,&s,&c);
    res.r = c * e;
    res.i = s * e;
    return res;

}

__device__ int isOnNF2FFBound(int x, int y)
{
    if(x==NF2FFdistfromboundary||x==nx-NF2FFdistfromboundary||y==NF2FFdistfromboundary||y==ny-NF2FFdistfromboundary)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

__device__ int getxfromthreadIdNF2FF(int index)
{
    int x=0;
    if(index<(nx-2*NF2FFdistfromboundary-2))//yn
    {
        x = index+NF2FFdistfromboundary+1;
    }
    else if(index<(nx-4*NF2FFdistfromboundary+ny-2))//xp
    {
        x = nx-NF2FFdistfromboundary-1;
    }
    else if(index<(2*nx-6*NF2FFdistfromboundary+ny-4))//yp
    {
        x = nx-NF2FFdistfromboundary  - (index-(nx-4*NF2FFdistfromboundary+ny-2))-2;
    }
    else if(index<(2*nx-8*NF2FFdistfromboundary+2*ny-4))//xn notice 2*nx-8*NF2FFdistfromboundary+2*ny-4 is the max index term.
    {
        x = NF2FFdistfromboundary;
    }
    return x;
}

__device__ int getyfromthreadIdNF2FF(int index)
{
    int y=0;
    if(index<(nx-2*NF2FFdistfromboundary-2))
    {
        y = NF2FFdistfromboundary;
    }
    else if(index<(nx-4*NF2FFdistfromboundary+ny-2))
    {
        y = (index-(nx-2*NF2FFdistfromboundary-2))+NF2FFdistfromboundary;
    }
    else if(index<(2*nx-6*NF2FFdistfromboundary+ny-4))
    {
        y = ny-NF2FFdistfromboundary-1;
    }
    else if(index<(2*nx-8*NF2FFdistfromboundary+2*ny-4))
    {
        y = ny-NF2FFdistfromboundary-(index-(2*nx-6*NF2FFdistfromboundary+ny-4))-1;
    }
    return y;
}

int CPUgetxfromthreadIdNF2FF(int index)
{
    int x=0;
    if(index<(nx-2*NF2FFdistfromboundary-2))//yn
    {
        x = index+NF2FFdistfromboundary+1;
    }
    else if(index<(nx-4*NF2FFdistfromboundary+ny-2))//xp
    {
        x = nx-NF2FFdistfromboundary-1;
    }
    else if(index<(2*nx-6*NF2FFdistfromboundary+ny-4))//yp
    {
        x = nx-NF2FFdistfromboundary  - (index-(nx-4*NF2FFdistfromboundary+ny-2))-2;
    }
    else if(index<(2*nx-8*NF2FFdistfromboundary+2*ny-4))//xn notice 2*nx-8*NF2FFdistfromboundary+2*ny-4 is the max index term.
    {
        x = NF2FFdistfromboundary;
    }
    return x;
}

int CPUgetyfromthreadIdNF2FF(int index)
{
    int y=0;
    if(index<(nx-2*NF2FFdistfromboundary-2))
    {
        y = NF2FFdistfromboundary;
    }
    else if(index<(nx-4*NF2FFdistfromboundary+ny-2))
    {
        y = (index-(nx-2*NF2FFdistfromboundary-2))+NF2FFdistfromboundary;
    }
    else if(index<(2*nx-6*NF2FFdistfromboundary+ny-4))
    {
        y = ny-NF2FFdistfromboundary-1;
    }
    else if(index<(2*nx-8*NF2FFdistfromboundary+2*ny-4))
    {
        y = ny-NF2FFdistfromboundary-(index-(2*nx-6*NF2FFdistfromboundary+ny-4))-1;
    }
    return y;
}

__device__ __host__ int isOnxn(int x)
{
    if(x==(NF2FFdistfromboundary))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

__device__ __host__ int isOnxp(int x)
{
    if(x==(nx-NF2FFdistfromboundary-1))
    { 
        return 1;
    }
    else
    {
        return 0;
    }
}

__device__ __host__ int isOnyp(int x,int y)
{
    if(y==(ny-NF2FFdistfromboundary-1)&&!isOnxn(x)&&!isOnxp(x))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

__device__ __host__ int isOnyn(int x, int y)
{
    if((y==(NF2FFdistfromboundary))&&!isOnxn(x)&&!(isOnxp(x)))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

__device__ int dgetCell(int x, int y, int size)
{
    return x +y*size;
}

__global__ void calculate_JandM(float* f,int* timestep,float*dev_Ez,float*dev_Hy,float*dev_Hx,cuComplex *cjzxp,cuComplex *cjzyp,cuComplex*cjzxn,cuComplex*cjzyn,cuComplex*cmxyp,cuComplex*cmyxp,cuComplex*cmxyn,cuComplex*cmyxn)
{
    float freq = *f;
    int index = threadIdx.x+blockIdx.x*blockDim.x;// should launch 2*nx-8*NF2FFdistfromboundary+2*ny-4 threads. 
    if(index<=size_NF2FF_total)
    {
        const cuComplex j(0.0,1.0);
        int x = getxfromthreadIdNF2FF(index);
        int y = getyfromthreadIdNF2FF(index);

        float Ez;
        cuComplex pi(PI , 0);
        cuComplex two(2.0,0.0);
        cuComplex negativeone(-1.0,0);
        cuComplex deltatime(dt,0);

        if(isOnyp(x,y))
        {
            Ez = (dev_Ez[dgetCell(x,y+1,nx+1)]+dev_Ez[dgetCell(x,y,nx+1)])/2;
            float Hx = dev_Hx[dgetCell(x,y,nx)];
            cjzyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Hx*deltatime*cuexp((float)(-1)*j*(float)2*pi*freq*(float)(*timestep)*deltatime);//cjzyp and cmxyp have nx - 2*NF2FFBoundary -2 elements

            cmxyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Ez*deltatime*cuexp((float)-1.0*j*(float)2.0*(float)PI*freq*((float)(*timestep)+0.5)*(float)dt);
        }
        else if(isOnxp(x))//X faces override y faces at their intersections
        {
            Ez = (dev_Ez[dgetCell(x,y,nx+1)]+dev_Ez[dgetCell(x+1,y,nx+1)])/2;
            float Hy = dev_Hy[dgetCell(x,y,nx)];

            cjzxp[index-(nx-2*NF2FFdistfromboundary-2)] += Hy*deltatime*cuexp(-1*j*2*pi*freq*(float)(*timestep)*(float)dt);//cjzxp and cmyxp have ny-2*NF2FFBound elements

            cmyxp[index-(nx-2*NF2FFdistfromboundary-2)] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*pi*freq*((float)(*timestep)+0.5)*(float)dt);// this is the discrete fourier transform, by the way.
        }
        else if(isOnyn(x,y))
        {  
            Ez = (dev_Ez[dgetCell(x,y,nx+1)]+dev_Ez[dgetCell(x,y+1,nx+1)])/2;
            float Hx=dev_Hx[dgetCell(x,y,nx)];

            cjzyn[index] += Hx*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt);	//cjzyn and cmxyn need to have nx-2*NF2FFbound-2 elements
            cmxyn[index] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*((float)(*timestep)+0.5)*(float)dt);
        }
        else if(isOnxn(x))
        {
            Ez = (dev_Ez[dgetCell(x,y,nx+1)]+dev_Ez[dgetCell(x+1,y,nx+1)])/2;
            cjzxn[index-(2*nx+ny-6*NF2FFdistfromboundary-4)] += -1*dev_Hy[dgetCell(x,y,nx)]*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt); // cjzxn and cmyxn must have ny-2*NFdistfromboundary elements
            cmyxn[index-(2*nx+ny-6*NF2FFdistfromboundary-4)] += -1*Ez*(float)dt*cuexp(-1.0*j*2.0*(float)PI*freq*((float)(*timestep)+0.5)*(float)dt);
        }
    }

}

__host__ __device__ float fwf(float timestep,float x, float y,float Phi_inc,float l)
{

    float ar;
    float ky, kx;//k hat
    sincosf(Phi_inc,&ky,&kx);

    ar = (float)timestep*dt-(float)t0-(1/(float)c0)*(ky*y*dx+kx*x*dy-l);
    //ar = timestep*dt-t0;

    //return exp(-1*(ar*ar)/(tau*tau));// gaussian pulse  argument is k dot r, 
    return exp(-1*ar*ar/(tau*tau));
    //return sin(2*PI*1e9*timestep*dt);
}

__global__ void H_field_update(float*dev_Hy,float*dev_Hx,float*dev_Ez,float*dev_bmx,float*dev_Psi_hyx,float*dev_amx,float*dev_bmy,float*dev_amy,float*dev_Psi_hxy,float*kex)
{
    float buffer_Hy;
    float buffer_Hx;
    float Chez = (dt/dx)/(mu0);
    int x = threadIdx.x +blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if(x<nx&&y<nx)
    {
        buffer_Hy = dev_Hy[dgetCell(x,y,nx)]+Chez*(dev_Ez[dgetCell(x+1,y,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
        buffer_Hx = dev_Hx[dgetCell(x,y,nx)]-Chez*(dev_Ez[dgetCell(x,y+1,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
        if(x<ncells)
        {
            buffer_Hy= dev_Hy[dgetCell(x,y,nx)]+Chez*(dev_Ez[dgetCell(x+1,y,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)])/kex[ncells-1-x];
            dev_Psi_hyx[dgetCell(x,y,20)]=dev_bmx[ncells-1-x]*dev_Psi_hyx[dgetCell(x,y,20)]+dev_amx[ncells-1-x]*(dev_Ez[dgetCell(x+1,y,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
            buffer_Hy+=Chez*dx*dev_Psi_hyx[dgetCell(x,y,20)] ;
        } 
        if(x>=(nx-ncells))
        {
            buffer_Hy=dev_Hy[dgetCell(x,y,nx)]+Chez*(dev_Ez[dgetCell(x+1,y,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)])/kex[x-nx+ncells];
            dev_Psi_hyx[dgetCell(x-nx+20,y,2*ncells)]=dev_bmx[x-nx+ncells]*dev_Psi_hyx[dgetCell(x-nx+20,y,20)]+dev_amx[x-nx+ncells]*(dev_Ez[dgetCell(x+1,y,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
            buffer_Hy+=Chez*dx*dev_Psi_hyx[dgetCell(x-nx+20,y,20)];
        }
        if(y<ncells)
        {
            buffer_Hx=dev_Hx[dgetCell(x,y,nx)]-Chez*(dev_Ez[dgetCell(x,y+1,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)])/kex[ncells-1-y];
            dev_Psi_hxy[dgetCell(x,y,nx)]=dev_bmy[ncells-1-y]*dev_Psi_hxy[dgetCell(x,y,nx)]+dev_amy[ncells-1-y]*(dev_Ez[dgetCell(x,y+1,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
            buffer_Hx-=Chez*dy*dev_Psi_hxy[dgetCell(x,y,nx)];  
        }
        if(y>=(ny-ncells))
        {
            buffer_Hx=dev_Hx[dgetCell(x,y,nx)]-Chez*(dev_Ez[dgetCell(x,y+1,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)])/kex[y-ny+ncells];
            dev_Psi_hxy[dgetCell(x,y-ny+20,nx)]=dev_bmy[y-ny+ncells]*dev_Psi_hxy[dgetCell(x,y-ny+20,nx)]+dev_amy[y-ny+ncells]*(dev_Ez[dgetCell(x,y+1,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
            buffer_Hx-=Chez*dy*dev_Psi_hxy[dgetCell(x,y-nx+20,nx)];
        }
        //__syncthreads();
        if(isnan(buffer_Hx)) 
        {
            dev_Hx[dgetCell(x,y,nx)] = 0.0;
        }
        else 
        {
            dev_Hx[dgetCell(x,y,nx)] = buffer_Hx;
        }

        if(isnan(buffer_Hy)) {
            dev_Hy[dgetCell(x,y,nx)] = 0.0;
        }
        else
        {
            dev_Hy[dgetCell(x,y,nx)] = buffer_Hy;
        }

        //dev_Hx[dgetCell(x,y,nx)] = buffer_Hx;
        //dev_Hy[dgetCell(x,y,nx)] = buffer_Hy;
    }
}

__global__ void H_inc_update(float*dev_Hy,float*dev_Hx,float*dev_Ez,float*dev_bmx,float*dev_Psi_hyx,float*dev_amx,float*dev_bmy,float*dev_amy,float*dev_Psi_hxy,float*kex)
{
    float buffer_Hy;
    float buffer_Hx;
    float Chez = (dt/dx)/(mu0);
    int x = threadIdx.x +blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    if(x<nx&&y<nx)
    {
        buffer_Hy = dev_Hy[dgetCell(x,y,nx)]+Chez*(dev_Ez[dgetCell(x+1,y,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
        buffer_Hx = dev_Hx[dgetCell(x,y,nx)]-Chez*(dev_Ez[dgetCell(x,y+1,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
        if(x<ncells)
        {
            buffer_Hy= dev_Hy[dgetCell(x,y,nx)]+Chez*(dev_Ez[dgetCell(x+1,y,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)])/kex[ncells-1-x];
            dev_Psi_hyx[dgetCell(x,y,20)]=dev_bmx[ncells-1-x]*dev_Psi_hyx[dgetCell(x,y,20)]+dev_amx[ncells-1-x]*(dev_Ez[dgetCell(x+1,y,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
            buffer_Hy+=Chez*dx*dev_Psi_hyx[dgetCell(x,y,20)] ;
        } 
        if(x>=(nx-ncells))
        {
            buffer_Hy=dev_Hy[dgetCell(x,y,nx)]+Chez*(dev_Ez[dgetCell(x+1,y,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)])/kex[x-nx+ncells];
            dev_Psi_hyx[dgetCell(x-nx+20,y,2*ncells)]=dev_bmx[x-nx+ncells]*dev_Psi_hyx[dgetCell(x-nx+20,y,20)]+dev_amx[x-nx+ncells]*(dev_Ez[dgetCell(x+1,y,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
            buffer_Hy+=Chez*dx*dev_Psi_hyx[dgetCell(x-nx+20,y,20)];
        }
        if(y<ncells)
        {
            buffer_Hx=dev_Hx[dgetCell(x,y,nx)]-Chez*(dev_Ez[dgetCell(x,y+1,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)])/kex[ncells-1-y];
            dev_Psi_hxy[dgetCell(x,y,nx)]=dev_bmy[ncells-1-y]*dev_Psi_hxy[dgetCell(x,y,nx)]+dev_amy[ncells-1-y]*(dev_Ez[dgetCell(x,y+1,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
            buffer_Hx-=Chez*dy*dev_Psi_hxy[dgetCell(x,y,nx)];  
        }
        if(y>=(ny-ncells))
        {
            buffer_Hx=dev_Hx[dgetCell(x,y,nx)]-Chez*(dev_Ez[dgetCell(x,y+1,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)])/kex[y-ny+ncells];
            dev_Psi_hxy[dgetCell(x,y-ny+20,nx)]=dev_bmy[y-ny+ncells]*dev_Psi_hxy[dgetCell(x,y-ny+20,nx)]+dev_amy[y-ny+ncells]*(dev_Ez[dgetCell(x,y+1,nx+1)]-dev_Ez[dgetCell(x,y,nx+1)]);
            buffer_Hx-=Chez*dy*dev_Psi_hxy[dgetCell(x,y-nx+20,nx)];
        }
        //__syncthreads();
        if(isnan(buffer_Hx)) 
        {
            dev_Hx[dgetCell(x,y,nx)] = 0.0;
        }
        else 
        {
            dev_Hx[dgetCell(x,y,nx)] = buffer_Hx;
        }

        if(isnan(buffer_Hy)) {
            dev_Hy[dgetCell(x,y,nx)] = 0.0;
        }
        else
        {
            dev_Hy[dgetCell(x,y,nx)] = buffer_Hy;
        }

        //dev_Hx[dgetCell(x,y,nx)] = buffer_Hx;
        //dev_Hy[dgetCell(x,y,nx)] = buffer_Hy;
    }
}

__global__ void E_field_update(int *i,float*dev_Ez,float*dev_Hy,float*dev_Hx,float*dev_Psi_ezx,float*dev_aex,float*dev_aey,float*dev_bex,float*dev_bey,float*dev_Psi_ezy,float*kex,float*Cezhy,float*Cezhx,float*Ceze,float*Cezeip,float*Cezeic,float*Ezip,float*Ezic)
{
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    int y=threadIdx.y+blockDim.y*blockIdx.y;
    //	int offset = x+y*blockDim.x*gridDim.x;
    float buffer_Ez;
    //float Ceh = (dt/dx)/(eps0);
    float Cezj = -dt/eps0;

    if(x<=nx&&y<=ny)
    {

        //if(x==0||x==nx||y==0||y==ny)
        if(x==nx||y==ny||x==0||y==0)
        {
            buffer_Ez=0.0;
        }
        else
        {
            if(isscattering)
            {
                if(!isPW)
                {
                    buffer_Ez = Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])
                        -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)])
                        +Cezeic[dgetCell(x,y,nx+1)]*Ezic[dgetCell(x,y,nx+1)]
                        +Cezeip[dgetCell(x,y,nx+1)]*Ezip[dgetCell(x,y,nx+1)];
                }
                else
                {
                    buffer_Ez = Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])
                        -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)])
                        +Cezeic[dgetCell(x,y,nx+1)]*fwf((float)(*i)+0.5,x,y,0,l0)
                        +Cezeip[dgetCell(x,y,nx+1)]*fwf((float)(*i)-0.5,x,y,0,l0);
                }
            }
            else
            {
                buffer_Ez = Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])
                    -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)]);
                if(x==(int)(source_x)&&y==(int)(source_y))
                {
                    buffer_Ez=buffer_Ez + 100*Cezj*fwf((float)(*i),0,0,0,0);
                }
            }

            //if(x==((int)nx/2)&&y==((int)nx/2))
            //{
            //	//buffer_Ez=buffer_Ez + Cezj*dev_Jz[*i];
            //	buffer_Ez=buffer_Ez + Cezj*fwf((float)(*i),0,0,0,0);
            //}
            if(x<=ncells&&x!=0)
            {
                buffer_Ez = Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])/kex[ncells-x]
                    -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)])/kex[ncells-x];
                dev_Psi_ezx[dgetCell(x-1,y-1,20)] = dev_bex[ncells-x]*dev_Psi_ezx[dgetCell(x-1,y-1,20)]+dev_aex[ncells-x]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)]);
                buffer_Ez += Cezhy[dgetCell(x,y,nx+1)]*dx*dev_Psi_ezx[dgetCell(x-1,y-1,2*ncells)];
            }
            if(x>=(nx-ncells)&&x!=nx)
            {
                buffer_Ez = Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])/kex[x-nx+ncells]
                    -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)])/kex[x-nx+ncells];
                dev_Psi_ezx[dgetCell(x-nx+20,y-1,20)]=dev_bex[x-nx+ncells]*dev_Psi_ezx[dgetCell(x-nx+20,y-1,20)]+dev_aex[x-nx+ncells]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)]);
                buffer_Ez+=Cezhy[dgetCell(x,y,nx+1)]*dx*dev_Psi_ezx[dgetCell(x-nx+20,y-1,2*ncells)];
            }
            if(y<=ncells&&y!=0)
            {
                buffer_Ez = Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])/kex[ncells-y]
                    -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)])/kex[ncells-y];
                dev_Psi_ezy[dgetCell(x-1,y-1,nx)]=dev_bey[(ncells-y)]*dev_Psi_ezy[dgetCell(x-1,y-1,nx)]+dev_aey[(ncells-y)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)]);
                buffer_Ez-=Cezhx[dgetCell(x,y,nx+1)]*dy*dev_Psi_ezy[dgetCell(x-1,y-1,nx)];
            }
            if(y>=(ny-ncells)&&y!=ny)
            {
                buffer_Ez = Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])/kex[y-ny+ncells]
                    -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)])/kex[y-ny+ncells];
                dev_Psi_ezy[dgetCell(x-1,y-ny+20,nx)]=dev_bey[y-ny+ncells]*dev_Psi_ezy[dgetCell(x-1,y-ny+20,nx)]+dev_aey[y-ny+ncells]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)]);
                buffer_Ez-=Cezhx[dgetCell(x,y,nx+1)]*dy*dev_Psi_ezy[dgetCell(x-1,y-ny+20,nx)];
            }
        }
        //		unsigned char green = 128+127*buffer_Ez/0.4;
        /*ptr[offset].x = 0;
          ptr[offset].y = green;
          ptr[offset].z = 0;
          ptr[offset].w = 255;*///OpenGL stuff

        //__syncthreads();
        if(isnan(buffer_Ez)) {
            dev_Ez[dgetCell(x,y,nx+1)] = 0.0;
        }
        else {
            dev_Ez[dgetCell(x,y,nx+1)] = buffer_Ez;
        }
        //dev_Ez[dgetCell(x,y,nx+1)] = buffer_Ez;
    }

}

__global__ void E_inc_update(int *i,float*dev_Hy_inc,float*dev_Hx_inc,float*dev_Psi_ezx_inc,float*dev_aex,float*dev_aey,float*dev_bex,float*dev_bey,float*dev_Psi_ezy_inc,float*kex,float*dev_Ezip,float*dev_Ezic)
{
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    int y=threadIdx.y+blockDim.y*blockIdx.y;
    //	int offset = x+y*blockDim.x*gridDim.x;
    float buffer_Ez;
    //float Ceh = (dt/dx)/(eps0);
    float Cezj = -dt/eps0;
    float Ceze = 1;
    float Cezhy = (dt/(dx*eps0));

    if(x<=nx&&y<=ny)
    {

        //if(x==0||x==nx||y==0||y==ny)
        if(x==nx||y==ny||x==0||y==0)
        {
            buffer_Ez=0.0;
        }
        else
        {
            buffer_Ez = Ceze*dev_Ezic[dgetCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[dgetCell(x,y,nx)]-dev_Hy_inc[dgetCell(x-1,y,nx)])
                -Cezhy*(dev_Hx_inc[dgetCell(x,y,nx)]-dev_Hx_inc[dgetCell(x,y-1,nx)]);

            if(x==((int)source_x)&&y==(int)(source_y))
            {
                //buffer_Ez=buffer_Ez + Cezj*dev_Jz[*i];
                buffer_Ez=buffer_Ez + 100*Cezj*fwf((float)(*i),0,0,0,0);
            }
            if(x<=ncells&&x!=0)
            {
                buffer_Ez = Ceze*dev_Ezic[dgetCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[dgetCell(x,y,nx)]-dev_Hy_inc[dgetCell(x-1,y,nx)])/kex[ncells-x]
                    -Cezhy*(dev_Hx_inc[dgetCell(x,y,nx)]-dev_Hx_inc[dgetCell(x,y-1,nx)])/kex[ncells-x];
                dev_Psi_ezx_inc[dgetCell(x-1,y-1,20)] = dev_bex[ncells-x]*dev_Psi_ezx_inc[dgetCell(x-1,y-1,20)]+dev_aex[ncells-x]*(dev_Hy_inc[dgetCell(x,y,nx)]-dev_Hy_inc[dgetCell(x-1,y,nx)]);
                buffer_Ez += Cezhy*dx*dev_Psi_ezx_inc[dgetCell(x-1,y-1,2*ncells)];
            }
            if(x>=(nx-ncells)&&x!=nx)
            {
                buffer_Ez = Ceze*dev_Ezic[dgetCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[dgetCell(x,y,nx)]-dev_Hy_inc[dgetCell(x-1,y,nx)])/kex[x-nx+ncells]
                    -Cezhy*(dev_Hx_inc[dgetCell(x,y,nx)]-dev_Hx_inc[dgetCell(x,y-1,nx)])/kex[x-nx+ncells];
                dev_Psi_ezx_inc[dgetCell(x-nx+20,y-1,20)]=dev_bex[x-nx+ncells]*dev_Psi_ezx_inc[dgetCell(x-nx+20,y-1,20)]+dev_aex[x-nx+ncells]*(dev_Hy_inc[dgetCell(x,y,nx)]-dev_Hy_inc[dgetCell(x-1,y,nx)]);
                buffer_Ez+=Cezhy*dx*dev_Psi_ezx_inc[dgetCell(x-nx+20,y-1,2*ncells)];
            }
            if(y<=ncells&&y!=0)
            {
                buffer_Ez = Ceze*dev_Ezic[dgetCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[dgetCell(x,y,nx)]-dev_Hy_inc[dgetCell(x-1,y,nx)])/kex[ncells-y]
                    -Cezhy*(dev_Hx_inc[dgetCell(x,y,nx)]-dev_Hx_inc[dgetCell(x,y-1,nx)])/kex[ncells-y];
                dev_Psi_ezy_inc[dgetCell(x-1,y-1,nx)]=dev_bey[(ncells-y)]*dev_Psi_ezy_inc[dgetCell(x-1,y-1,nx)]+dev_aey[(ncells-y)]*(dev_Hx_inc[dgetCell(x,y,nx)]-dev_Hx_inc[dgetCell(x,y-1,nx)]);
                buffer_Ez-=Cezhy*dy*dev_Psi_ezy_inc[dgetCell(x-1,y-1,nx)];
            }
            if(y>=(ny-ncells)&&y!=ny)
            {
                buffer_Ez = Ceze*dev_Ezic[dgetCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[dgetCell(x,y,nx)]-dev_Hy_inc[dgetCell(x-1,y,nx)])/kex[y-ny+ncells]
                    -Cezhy*(dev_Hx_inc[dgetCell(x,y,nx)]-dev_Hx_inc[dgetCell(x,y-1,nx)])/kex[y-ny+ncells];
                dev_Psi_ezy_inc[dgetCell(x-1,y-ny+20,nx)]=dev_bey[y-ny+ncells]*dev_Psi_ezy_inc[dgetCell(x-1,y-ny+20,nx)]+dev_aey[y-ny+ncells]*(dev_Hx_inc[dgetCell(x,y,nx)]-dev_Hx_inc[dgetCell(x,y-1,nx)]);
                buffer_Ez-=Cezhy*dy*dev_Psi_ezy_inc[dgetCell(x-1,y-ny+20,nx)];
            }
        }
        dev_Ezip[dgetCell(x,y,nx+1)] = dev_Ezic[dgetCell(x,y,nx+1)];
        dev_Ezic[dgetCell(x,y,nx+1)] = buffer_Ez;
    }

}

float calc_radiated_power(cuComplex *cjzxp,cuComplex *cjzyp,cuComplex *cjzxn,cuComplex *cjzyn,cuComplex *cmxyp,cuComplex *cmyxp,cuComplex *cmxyn,cuComplex *cmyxn)
{
    int indexofleg1 = nx-2*NF2FFdistfromboundary-2;
    int indexofleg2 = nx+ny-4*NF2FFdistfromboundary-2;
    int indexofleg3 = 2*nx+ny-6*NF2FFdistfromboundary-4;
    int maxindex = 2*nx-8*NF2FFdistfromboundary+2*ny-4;
    int index;
    cuComplex cjz(0,0);
    cuComplex power = 0;

    for(index = 0; index<indexofleg1;index++)
    {   cjz = cuComplex(cjzyn[index].r,-1.0*cjzyn[index].i);//conjugation
        //z x x = y dot -y = -1
        power+=-1.0*cjz*cmxyn[index]*dx;// the negative one comes from the dot product between JxM and the n hat vector
    }
    for(index = indexofleg1; index<indexofleg2;index++)
    {
        cjz = cuComplex(cjzxp[index-indexofleg1].r,-1.0*cjzxp[index-indexofleg1].i);//making the conjugate
        // z cross y = -x dot x = -1
        power+= -1.0*cjz*cmyxp[index-indexofleg1]*dy;//positive x unit normal vector
    }
    for(index = indexofleg2;index<indexofleg3;index++)
    {
        // z cross x = y dot y = 1
        cjz = cuComplex(cjzyp[index-indexofleg2].r,-1.0*cjzyp[index-indexofleg2].i);
        power+= cjz*cmxyp[index-indexofleg2]*dx;//postive y unit normal vector
    }
    for(index = indexofleg3;index<maxindex;index++)
    {
        // z cross y = -x dot -x = 1 
        cjz = cuComplex(cjzxn[index-indexofleg3].r,-1.0*cjzxn[index-indexofleg3].i);
        power += cjz*cmyxn[index-indexofleg3]*dy;// negative x hat n vector
    }
    float realpower = power.r;
    realpower *= 0.5;
    return realpower;
}

float calc_incident_power(float freq)
{
    return (0.5/eta0)*pow(tau*sqrt(PI)*exp(-tau*tau*2*PI*freq*2*PI*freq/4),2);// just gonna assume gaussian pulse.  This is the fourier transform of the gaussian pulse.
}

__global__ void calculate_JandM_total(float* f,int* timestep,float*dev_Ez,float*dev_Hy,float*dev_Hx,cuComplex *cjzxp,cuComplex *cjzyp,cuComplex*cjzxn,cuComplex*cjzyn,cuComplex*cmxyp,cuComplex*cmyxp,cuComplex*cmxyn,cuComplex*cmyxn,float*dev_Ezic,float*dev_Ezip,float*dev_Hx_inc,float*dev_Hy_inc)
{
    float freq = *f;
    int index = threadIdx.x+blockIdx.x*blockDim.x;// should launch 2*nx-8*NF2FFdistfromboundary+2*ny-4 threads. 
    if(index<=size_NF2FF_total)
    {
        const cuComplex j(0.0,1.0);
        int x = getxfromthreadIdNF2FF(index);
        int y = getyfromthreadIdNF2FF(index);

        float Ez;
        cuComplex pi(PI , 0);
        cuComplex two(2.0,0.0);
        cuComplex negativeone(-1.0,0);
        cuComplex deltatime(dt,0);

        if(isOnyp(x,y))
        {
            Ez = (dev_Ez[dgetCell(x,y+1,nx+1)]+dev_Ez[dgetCell(x,y,nx+1)])/2;
            Ez += (dev_Ezic[dgetCell(x,y+1,nx+1)] + dev_Ezic[dgetCell(x,y,nx+1)] + dev_Ezip[dgetCell(x,y+1,nx+1)] + dev_Ezip[dgetCell(x,y,nx+1)])/4;
            float Hx = dev_Hx[dgetCell(x,y,nx)] + dev_Hx_inc[dgetCell(x,y,nx)];
            cjzyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Hx*deltatime*cuexp((float)(-1)*j*(float)2*pi*freq*(float)(*timestep)*deltatime);//cjzyp and cmxyp have nx - 2*NF2FFBoundary -2 elements
            cmxyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Ez*deltatime*cuexp((float)-1.0*j*(float)2.0*(float)PI*freq*((float)(*timestep)-0.5)*(float)dt);
        }
        else if(isOnxp(x))//X faces override y faces at their intersections
        {
            Ez = (dev_Ez[dgetCell(x,y,nx+1)]+dev_Ez[dgetCell(x+1,y,nx+1)])/2;
            Ez += (dev_Ezic[dgetCell(x+1,y,nx+1)] + dev_Ezic[dgetCell(x,y,nx+1)] + dev_Ezip[dgetCell(x+1,y,nx+1)] + dev_Ezip[dgetCell(x,y,nx+1)])/4;
            float Hy = dev_Hy[dgetCell(x,y,nx)] + dev_Hy_inc[dgetCell(x,y,nx)];

            cjzxp[index-(nx-2*NF2FFdistfromboundary-2)] += Hy*deltatime*cuexp(-1*j*2*pi*freq*(float)(*timestep)*(float)dt);//cjzxp and cmyxp have ny-2*NF2FFBound elements

            cmyxp[index-(nx-2*NF2FFdistfromboundary-2)] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*pi*freq*((float)(*timestep)-0.5)*(float)dt);// this is the discrete fourier transform, by the way.
        }
        else if(isOnyn(x,y))
        {  
            Ez = (dev_Ez[dgetCell(x,y,nx+1)]+dev_Ez[dgetCell(x,y+1,nx+1)])/2;
            Ez += (dev_Ezic[dgetCell(x,y+1,nx+1)] + dev_Ezic[dgetCell(x,y,nx+1)] + dev_Ezip[dgetCell(x,y+1,nx+1)] + dev_Ezip[dgetCell(x,y,nx+1)])/4;
            float Hx=dev_Hx[dgetCell(x,y,nx)]+dev_Hx_inc[dgetCell(x,y,nx)];

            cjzyn[index] += Hx*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt);	//cjzyn and cmxyn need to have nx-2*NF2FFbound-2 elements
            cmxyn[index] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*((float)(*timestep)-0.5)*(float)dt);
        }
        else if(isOnxn(x))
        {
            Ez = (dev_Ez[dgetCell(x,y,nx+1)]+dev_Ez[dgetCell(x+1,y,nx+1)])/2;
            Ez += (dev_Ezic[dgetCell(x+1,y,nx+1)] + dev_Ezic[dgetCell(x,y,nx+1)] + dev_Ezip[dgetCell(x+1,y,nx+1)] + dev_Ezip[dgetCell(x,y,nx+1)])/4;
            float Hy = dev_Hy[dgetCell(x,y,nx)] + dev_Hy_inc[dgetCell(x,y,nx)];
            cjzxn[index-(2*nx+ny-6*NF2FFdistfromboundary-4)] += -1*Hy*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt); // cjzxn and cmyxn must have ny-2*NFdistfromboundary elements
            cmyxn[index-(2*nx+ny-6*NF2FFdistfromboundary-4)] += -1*Ez*(float)dt*cuexp(-1.0*j*2.0*(float)PI*freq*((float)(*timestep)-0.5)*(float)dt);
        }
    }

}

__host__ __device__ int getOptimizationCell(int x, int y)
{
    int x_coord,y_coord;
    x_coord = (x-(nx/2-(int)(breast_radius/dx)))/(2*breast_radius/(9*dx));
    y_coord = (y-(ny/2-breast_radius/dy))/(2*breast_radius/(9*dy));//the optimization space is 216 FDTD cells wide and high. //The optimization space is split into 25 by 25 optimization cells. 
    //each optimization cell has 24 by 24 FDTD cells within it. That's what the 108, 24 and 25 are about.  
    return x_coord+9*y_coord;//The max return should be, 9*9-1, hopefully.
}

void N2FPostProcess (float* D,float f, cuComplex *N,cuComplex *L,cuComplex *cjzxp,cuComplex *cjzyp,cuComplex *cjzxn,cuComplex *cjzyn,cuComplex *cmxyp,cuComplex *cmyxp,cuComplex *cmxyn,cuComplex *cmyxn)
{
    int indexofleg1 = nx-2*NF2FFdistfromboundary-2;
    int indexofleg2 = nx+ny-4*NF2FFdistfromboundary-2;
    int indexofleg3 = 2*nx+ny-6*NF2FFdistfromboundary-4;
    int maxindex = 2*nx-8*NF2FFdistfromboundary+2*ny-4;
    int x,y;

    float rhoprime;
    float Psi;
    int Phi_index;
    cuComplex  Mphi(0,0);
    float Phi;


    float k = 2*PI*f/c0;
    cuComplex  negativeone(-1.0,0.0);
    int index = 0;
    cuComplex jcmpx(0,1);
    //float Prad = calc_radiated_power(cjzxp,cjzyp,cjzxn,cjzyn,cmxyp,cmyxp,cmxyn,cmyxn);
    float Prad = calc_incident_power(f);
    //std::cout<<"Prad = "<<Prad<<std::endl;
    float flx, fly;
    for(Phi_index = 0; Phi_index<numberofobservationangles;Phi_index++)
    {
        Phi = 2*PI/numberofobservationangles*(float)Phi_index;
        for(index = 0;index<indexofleg1;index++)
        {

            x = CPUgetxfromthreadIdNF2FF(index);
            y = CPUgetyfromthreadIdNF2FF(index);
            flx = (float)x;//float x
            fly = (float)y + 0.5;
            rhoprime = sqrt(pow((dx*((-1.0*(float)nx/2)+1+flx)),2)+pow((dy*(-1.0*(float)ny/2+1+fly)),2));
            Psi = atan2(-1*((float)ny/2)+1+fly,-1*((float)nx/2)+1+flx)-Phi;
            N[Phi_index]+=-1.0*cjzyn[index]*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*dx;
            L[Phi_index]+=-1.0*sin(Phi)*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*cmxyn[index]*dx;//Lphi = 

        }
        for(index = indexofleg1;index<indexofleg2;index++)
        {

            x = CPUgetxfromthreadIdNF2FF(index);
            y = CPUgetyfromthreadIdNF2FF(index);
            flx = (float)x+0.5;
            fly = (float)y;
            rhoprime = sqrt(pow((dx*(((float)nx/2)-1-flx)),2)+pow((dy*(((float)ny/2)-1-fly)),2)); 
            Psi = atan2(-1*((float)ny/2)+1+fly,(-1*((float)nx/2)+1+flx))-Phi;
            N[Phi_index]+=-1.0*cjzxp[index-indexofleg1]*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*dy;
            L[Phi_index]+=cos(Phi)*cmyxp[index-indexofleg1]*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*dy;//L_phi = -Lxsin(phi)+Lycos(Phi) here we only have Ly
        }
        for(index=indexofleg2;index<indexofleg3;index++)
        {
            x = CPUgetxfromthreadIdNF2FF(index);
            y = CPUgetyfromthreadIdNF2FF(index);
            flx = (float)x;
            fly = (float)y + 0.5;
            rhoprime = sqrt(pow((dx*(((float)nx/2)-1-flx)),2)+pow((dy*(((float)ny/2)-1-fly)),2)); 
            Psi = atan2((-1*(float)ny/2+1+fly),(-1*((float)nx/2)+1+flx))-Phi;
            N[Phi_index]+=-1.0*cjzyp[index-indexofleg2]*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*dx;
            L[Phi_index]+=-1.0*sin(Phi)*cmxyp[index-indexofleg2]*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*dx;//
        }
        for(index = indexofleg3;index<maxindex;index++)
        {
            x = CPUgetxfromthreadIdNF2FF(index);
            y = CPUgetyfromthreadIdNF2FF(index);
            flx = (float)x+0.5;
            fly = (float)y;
            rhoprime = sqrt(pow(dx*(((float)nx/2)-1-flx),2)+pow((dy*(((float)ny/2)-1-fly)),2)); 
            Psi = atan2(-1*((float)ny/2)+1+fly,-1*(float)nx/2+1+flx)-Phi;
            N[Phi_index]+=-1.0*cjzxn[index-indexofleg3]*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*dy;
            L[Phi_index]+= cos(Phi)*cmyxn[index-indexofleg3]*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*dy;
        }
        D[Phi_index] = (k*k*cuabs(L[Phi_index]+(float)eta0*N[Phi_index])*cuabs(L[Phi_index]+(float)eta0*N[Phi_index])/((float)8*(float)PI*(float)eta0*Prad*33.329));//why 33.329?  I dunno, something is probably wrong with Prad.
    }

}

float fitness(float* D,int numberobservationangles, float* measurement)
{
    float fit = 0;
    for(int i =0;i<numberobservationangles;i++)
    {
        fit -= pow((measurement[i]-D[i]),2)/pow(measurement[i],2);
    }

    return fit;
}

//static void draw_func(void){
//	glDrawPixels(nx,ny,GL_RGBA,GL_UNSIGNED_BYTE,0);
//	glutSwapBuffers;
//}

using namespace std;

void Ceze_init(float * eps_r_z, float* sig_e_z, float* Ceze);
void Cezhy_init(float* eps_r_z, float* sigma_e_z,float*Cezhy,float*kex);
void Cezhx_init(float* eps_r_z,float*sigma_e_z,float*Cezhx,float*kex);
void eps_r_z_init(float * eps_r_z,const vector<float> &argument);
void sigma_e_z_init(float *sigma_e_z,float*sigma_e_pml,const vector<float> &argument);
void Cezj_init(float*eps_r_z,float*sigma_e_z,float*Cezj);
void Ez_init(float*Ez);
void Ey_init(float*Ey);
//void Jz_init(float*Jz);
void Chxh_init(float*mu_r_x,float*sigma_m_x,float*Chxh);
void Chxez_init(float*mu_r_x,float*sigma_m_x,float*Chxez);
//void Chxm_init(float*mu_r_x,float*sigma_m_x,float*Chxm);
void Chyh_init(float*mu_r_y,float*sigma_m_y,float*Chyh);
void Chyez_init(float*mu_r_y,float*sigma_m_y,float*Chyez);
//void Chym_init(float*mu_r_y,float*sigma_m_y,float*Chym);
void Hy_init(float*Hy);
void Hx_init(float*Hx);
void My_init(float*My);
void Mx_init(float*Mx);
void mu_r_y_init(float*mu_r_y);
void mu_r_x_init(float*mu_r_x);
void sigma_m_y_init(float*sigma_m_y_init);
void sigma_m_x_init(float*sigma_m_x_init);
int getCell(int x,int y,int size);
void Jz_waveform(float * time,float*Jz_impressed);
void waveform_time_init(float*time1);
float* Make2DfloatArray(int arraySizeX, int arraySizeY);
void C_Psi_ezy_init(float *C_Psi_ezy,float*Cezhx);
void C_Psi_ezx_init(float* C_Psi_ezx,float*Cezhy);
void C_Psi_hyx_init(float*C_Psi_hyx,float*Chyez);
void C_psi_hxy_init(float *C_Psi_hxy,float*Chxez);

void aex_init(float*aex,float*sigma_e_pml,float*kex,float*alpha_e_x,float*bex);
void bex_init(float*bex ,float*sigma_e_pml,float*kex,float*alpha_e_x);   
void bey_init(float*bey,float*sigma_e_pml,float*key,float*alpha_e_y);
void amy_init(float*amy,float*sigma_m_pml,float*kmy,float*alpha_m_y,float*bmy);
void bmy_init(float*bmy,float*sigma_m_pml,float*kmy,float*alpha_m_y);
void amx_init(float*amx,float*sigma_m_pml,float*kmx,float*alpha_m_x,float*bmx);
void bmx_init(float*bmx,float*sigma_m_pml,float*kmx,float*alpha_m_x);
void alpha_e_init(float*alpha_e);
void alpha_m_init(float*alpha_e,float*alpha_m);
void k_e_init(float*k);
void k_m_init(float*k);
void sigma_e_pml_init(float* sigma_e_pml);
void sigma_m_pml_init(float*sigma_m_pml,float*sigma_e_pml);
void Psi_ezy_init(float*Psi_ezy);
void Psi_ezx_init(float*Psi_ezx);
void Psi_hyx_init(float*Psi_hyx);
void Psi_hxy_init(float*Psi_hxy);
void CJ_Init(cuComplex * cjzyn,int size);
__global__ void scattered_parameter_init(float*eps_r_z,float*sigma_e_z,float*Cezeic,float*Cezeip);

double FDTD_GPU(const vector<double> &arguments)
{
    //BMP Output_Image;
    //BMP Scattered_Field_snapshot;
    //Output_Image.SetSize((nx+1),(ny+1));
    //Output_Image.SetBitDepth(16);
    //Scattered_Field_snapshot.SetSize((nx+1),(ny+1));
    //Scattered_Field_snapshot.SetBitDepth(16);
    //RGBApixel Temp;
    //string outputfilename;
    //ebmpBYTE StepSize;

    cout << "calculating FDTD GPU" << endl;

    cudaSetDevice(0);

    vector<float> image;
    for(int lerp = 0; lerp<81;lerp++)//This is setting the material parameters of the optimization cells.
    {
        image.push_back((float)arguments.at(lerp));
        //image.push_back(20);
    }
    for(int lerp = 81; lerp<81*2;lerp++)
    {
        image.push_back((float)arguments.at(lerp));
        //image.push_back(0);
    }


    //GLuint bufferObj;
    //cudaGraphicsResource *resource;
    cudaError_t error;
    //int dev;
    //cudaDeviceProp prop;
    //memset(&prop,sizeof(cudaDeviceProp),sizeof(cudaDeviceProp));
    //prop.major = 1;
    //prop.minor = 1;
    //cudaChooseDevice(&dev,&prop);
    //	cudaGLSetGLDevice(dev);
    /*glutInit(&argc,argv);
      glewInit();
      glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);  
      glutInitWindowSize(nx,ny);
      glutCreateWindow("bitmap");
      glGenBuffers(1,&bufferObj);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
      glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, nx*ny*4,NULL,GL_DYNAMIC_DRAW_ARB);
      cudaGraphicsGLRegisterBuffer(&resource,bufferObj,cudaGraphicsMapFlagsNone);*/
    //uchar4* devPtr;
    //size_t size;
    //cudaGraphicsMapResources(1,&resource,NULL);
    //cudaGraphicsResourceGetMappedPointer((void**)&devPtr,&size,resource);

    float*Ceze,*Cezhy,*Cezhx,*dev_Cezeic,*dev_Cezeip,*Ez,*eps_r_z,*sigma_e_z,*Hy,*Hx,
        *kex,*aex,*bex,*amx,*bmx,*alpha_e,*alpha_m,*sigma_e_pml,*sigma_m_pml
            ,*Psi_ezy,*Psi_ezx,*Psi_hyx,*Psi_hxy,*kmx;//*Cezj later if using loop current source
    float* dev_sigma_e_z,*dev_eps_r_z;
    float freq = center_freq;
    float *dev_freq,*D,*D_tot;
    float* Ezip,*Ezic,*dev_Ezip,*dev_Ezic,*Hy_inc,*Hx_inc,*dev_Hy_inc,*dev_Hx_inc,*dev_Psi_ezy_inc,*dev_Psi_ezx_inc,*dev_Psi_hyx_inc,*dev_Psi_hxy_inc,
        *Psi_ezy_inc,*Psi_ezx_inc,*Psi_hyx_inc,*Psi_hxy_inc;

    cuComplex *cjzxp,*cjzyp,*cjzxn,*cjzyn,*cmxyp,*cmyxp,*cmxyn,*cmyxn,*cjzxp_tot,*cjzyp_tot,*cjzxn_tot,*cjzyn_tot,*cmxyp_tot,*cmyxp_tot,*cmxyn_tot,*cmyxn_tot;
    cuComplex *hcjzxp,*hcjzyp,*hcjzxn,*hcjzyn,*hcmxyp,*hcmyxp,*hcmxyn,*hcmyxn,*hcjzxp_tot,*hcjzyp_tot,*hcjzxn_tot,*hcjzyn_tot,*hcmxyp_tot,*hcmyxp_tot,*hcmxyn_tot
        ,*hcmyxn_tot;
    int grid_x = int(ceil((float)nx/22));
    int grid_y = int(ceil((float)ny/22));
    dim3 grid(grid_x,grid_y);
    dim3 block(22,22);
    Hy_inc = (float*)malloc(sizeof(float)*(nx*ny));
    Hx_inc = (float*)malloc(sizeof(float)*(nx*ny));
    Ezip = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    Ezic = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    for(int index = 0;index<(1+nx)*(1+ny);index++)
    {
        Ezip[index] = 0;
        Ezic[index] = 0;
        if(index<(nx*ny))
        {
            Hy_inc[index] = 0;
            Hx_inc[index] = 0;
        }
    }
    cudaMalloc(&dev_Ezip,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_Ezic,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_Hy_inc,sizeof(float)*(nx)*(ny));
    cudaMalloc(&dev_Hx_inc,sizeof(float)*(nx)*(ny));
    cudaMemcpy(dev_Ezip,Ezip,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Ezic,Ezic,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Hy_inc,Hy_inc,sizeof(float)*ny*nx,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Hx_inc,Hx_inc,sizeof(float)*ny*nx,cudaMemcpyHostToDevice);

    Ceze = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    Cezhy = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    Cezhx = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    //Cezj = (float*)malloc(sizeof(float)*(1+nx)*(1+ny)); // if using loop current source
    Ez = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    eps_r_z =  (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    sigma_e_z = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    D = (float*)malloc(sizeof(float)*numberofobservationangles);//D = (float*)malloc(numberofobservationangles*sizeof(float));
    Hy=(float*)malloc(sizeof(float)*nx*ny);
    Hx=(float*)malloc(sizeof(float)*nx*ny);
    kex = (float*)malloc(sizeof(float)*10);
    kmx = (float*)malloc(sizeof(float)*10);
    aex=(float*)malloc(sizeof(float)*10);
    bex=(float*)malloc(sizeof(float)*10);
    amx=(float*)malloc(sizeof(float)*10);
    bmx=(float*)malloc(sizeof(float)*10);
    alpha_e=(float*)malloc(sizeof(float)*10);
    alpha_m=(float*)malloc(sizeof(float)*10);
    sigma_e_pml=(float*)malloc(sizeof(float)*10);
    sigma_m_pml=(float*)malloc(sizeof(float)*10);
    Psi_ezy=(float*)malloc(sizeof(float)*ny*20);
    Psi_ezx=(float*)malloc(sizeof(float)*nx*20);
    Psi_hyx=(float*)malloc(sizeof(float)*ny*20);
    Psi_hxy=(float*)malloc(sizeof(float)*nx*20);
    Psi_ezy_inc=(float*)malloc(sizeof(float)*ny*20);
    Psi_ezx_inc=(float*)malloc(sizeof(float)*nx*20);
    Psi_hyx_inc=(float*)malloc(sizeof(float)*ny*20);
    Psi_hxy_inc=(float*)malloc(sizeof(float)*nx*20);
    hcjzyp = (cuComplex*)malloc(sizeof(cuComplex )*size_cjzy);
    hcjzyn = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzy);
    hcjzxp = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzx);
    hcjzxn = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzx);
    hcmxyn = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzy);
    hcmxyp = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzy);
    hcmyxp = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzx);
    hcmyxn = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzx);

    hcjzyp_tot = (cuComplex*)malloc(sizeof(cuComplex )*size_cjzy);
    hcjzyn_tot = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzy);
    hcjzxp_tot = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzx);
    hcjzxn_tot = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzx);
    hcmxyn_tot = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzy);
    hcmxyp_tot = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzy);
    hcmyxp_tot = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzx);
    hcmyxn_tot = (cuComplex *)malloc(sizeof(cuComplex )*size_cjzx);
    CJ_Init(hcjzyp,size_cjzy);//C**** coefficients are for surface current/ field duality for NF2FF processing.
    CJ_Init(hcjzyn,size_cjzy);
    CJ_Init(hcjzxp,size_cjzx);
    CJ_Init(hcjzxn,size_cjzx);
    CJ_Init(hcmxyn,size_cjzy);
    CJ_Init(hcmxyp,size_cjzy);
    CJ_Init(hcmyxp,size_cjzx);
    CJ_Init(hcmyxn,size_cjzx);
    CJ_Init(hcjzyp_tot,size_cjzy);//C**** coefficients are for surface current/ field duality for NF2FF processing.
    CJ_Init(hcjzyn_tot,size_cjzy);
    CJ_Init(hcjzxp_tot,size_cjzx);
    CJ_Init(hcjzxn_tot,size_cjzx);
    CJ_Init(hcmxyn_tot,size_cjzy);
    CJ_Init(hcmxyp_tot,size_cjzy);
    CJ_Init(hcmyxp_tot,size_cjzx);
    CJ_Init(hcmyxn_tot,size_cjzx);
    Psi_ezy_init(Psi_ezy);
    Psi_ezx_init(Psi_ezx);
    Psi_hyx_init(Psi_hyx);
    Psi_hxy_init(Psi_hxy);
    Psi_ezy_init(Psi_ezy_inc);
    Psi_ezx_init(Psi_ezx_inc);
    Psi_hyx_init(Psi_hyx_inc);
    Psi_hxy_init(Psi_hxy_inc);
    eps_r_z_init(eps_r_z,image);
    sigma_e_z_init(sigma_e_z,sigma_e_pml,image);
    Hy_init(Hy);
    Hx_init(Hx);
    //float*time1;
    //time1 = (float*)malloc(sizeof(float)*number_of_time_steps);
    Ceze_init(eps_r_z,sigma_e_z,Ceze);	
    k_e_init(kex);
    k_m_init(kmx);
    Cezhy_init(eps_r_z,sigma_e_z,Cezhy,kex);
    Cezhx_init(eps_r_z,sigma_e_z,Cezhx,kex);
    sigma_e_pml_init(sigma_e_pml);
    sigma_m_pml_init(sigma_m_pml,sigma_e_pml);
    alpha_e_init(alpha_e);
    alpha_m_init(alpha_e,alpha_m);
    bex_init(bex ,sigma_e_pml,kex,alpha_e);
    aex_init(aex,sigma_e_pml,kex,alpha_e,bex);
    bmx_init(bmx,sigma_m_pml,kmx,alpha_m);
    amx_init(amx,sigma_m_pml,kmx,alpha_m,bmx);
    Ez_init(Ez);
    //Jz_init(Jz);
    //system("pause");   
    //FILE* file = fopen("results.txt", "w");

    //float*Jz_impressed = (float*)malloc(sizeof(float)*number_of_time_steps);
    //waveform_time_init(time1);
    //Jz_waveform(time1,Jz_impressed);

    //int source_position_index_x = int(nx*source_position/domain_size)+1;

    //	int source_position_index_y = int(ny*source_position/domain_size)+1;
    float*dev_Ceze,*dev_Cezhy,*dev_Cezhx,*dev_bex,*dev_aex,*dev_bmx,*dev_amx,*dev_kex,*dev_kmx;//dev_Cezj if using loop current source
    float *dev_Ez,*dev_Hy,*dev_Hx;

    float*dev_Psi_ezy,*dev_Psi_ezx,*dev_Psi_hyx,*dev_Psi_hxy;

    cudaMalloc(&dev_eps_r_z,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_sigma_e_z,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_Cezeic,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_Cezeip,sizeof(float)*(nx+1)*(ny+1));
    cudaMemcpy(dev_eps_r_z,eps_r_z,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sigma_e_z,sigma_e_z,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice);
    scattered_parameter_init<<<grid,block>>>(dev_eps_r_z,dev_sigma_e_z,dev_Cezeic,dev_Cezeip);
    float *Cezeic = (float*)malloc((sizeof(float))*(nx+1)*(ny+1));
    float *Cezeip = (float*)malloc((sizeof(float))*(nx+1)*(ny+1));
    cudaMemcpy(Cezeic,dev_Cezeic,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyDeviceToHost);
    cudaMemcpy(Cezeip,dev_Cezeip,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyDeviceToHost);
    float radius;
    //for(int i = 0; i<(nx+1);i++)
    //{
    //	for(int j =0; j<(ny+1);j++)
    //	{
    //		radius = sqrt((i-nx/2)*(i-nx/2)*dx*dx+(j-ny/2)*(j-ny/2)*dy*dy);
    //		if(radius<breast_radius)
    //		{
    //		cout<<"Cezeip = "<<Cezeip[getCell(i,j,nx+1)]<<"(i,j) = ("<<i<<","<<j<<")"<<endl;
    //		//cin.ignore();
    //		}
    //	}
    //}

    cudaMalloc(&dev_kex,sizeof(float)*10);
    cudaMalloc(&dev_kmx,sizeof(float)*10);
    cudaMalloc(&dev_Ez,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_Hy,sizeof(float)*nx*ny);
    cudaMalloc(&dev_freq ,sizeof(float));
    cudaMalloc(&dev_Hx,sizeof(float)*nx*ny);
    cudaMalloc(&dev_Psi_ezy,sizeof(float)*20*(nx+1));
    cudaMalloc(&dev_Psi_ezx,sizeof(float)*20*(ny+1));
    cudaMalloc(&dev_Psi_hyx,sizeof(float)*20*(ny));
    cudaMalloc(&dev_Psi_hxy,sizeof(float)*20*(nx));
    cudaMalloc(&dev_Psi_ezy_inc,sizeof(float)*20*(nx+1));
    cudaMalloc(&dev_Psi_ezx_inc,sizeof(float)*20*(ny+1));
    cudaMalloc(&dev_Psi_hyx_inc,sizeof(float)*20*(ny));
    cudaMalloc(&dev_Psi_hxy_inc,sizeof(float)*20*(nx));
    cudaMalloc(&cjzxp,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cjzyp,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cjzxn,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cjzyn,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmxyp,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmxyn,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmyxp,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmyxn,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cjzxp_tot,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cjzyp_tot,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cjzxn_tot,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cjzyn_tot,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmxyp_tot,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmxyn_tot,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmyxp_tot,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmyxn_tot,sizeof(cuComplex)*size_NF2FF_total);
    cudaMemcpy(dev_freq,&freq,sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Psi_ezy,Psi_ezy,sizeof(float)*20*(nx),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Psi_ezx,Psi_ezx,sizeof(float)*20*(ny),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Psi_hyx,Psi_hyx,sizeof(float)*20*(ny),cudaMemcpyHostToDevice );
    cudaMemcpy(dev_Psi_hxy,Psi_hxy,sizeof(float)*20*nx,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Psi_ezy_inc,Psi_ezy_inc,sizeof(float)*20*nx,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Psi_ezx_inc,Psi_ezx_inc,sizeof(float)*20*ny,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Psi_hyx_inc,Psi_hyx_inc,sizeof(float)*20*ny,cudaMemcpyHostToDevice );
    cudaMemcpy(dev_Psi_hxy_inc,Psi_hxy_inc,sizeof(float)*20*nx,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Ez,Ez,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Hy,Hy,sizeof(float)*nx*ny,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Hx,Hx,sizeof(float)*nx*ny,cudaMemcpyHostToDevice);
    cudaMalloc(&dev_bex,sizeof(float)*10);
    cudaMalloc(&dev_bmx,sizeof(float)*10);
    cudaMalloc(&dev_amx,sizeof(float)*10);
    cudaMalloc(&dev_aex,sizeof(float)*10);
    cudaMalloc(&dev_Ceze,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_Cezhy,sizeof(float)*(nx+1)*(ny+1));


    //cudaMalloc(&dev_Cezj,sizeof(float)*(nx+1)*(ny+1)); if using current source

    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(error));
    }
    cudaMemcpy(cjzyn,hcjzyn,sizeof(cuComplex)*size_cjzy,cudaMemcpyHostToDevice);
    cudaMemcpy(cjzxp,hcjzxp,sizeof(cuComplex)*size_cjzx,cudaMemcpyHostToDevice);
    cudaMemcpy(cjzyp,hcjzyp,sizeof(cuComplex)*size_cjzy,cudaMemcpyHostToDevice);
    cudaMemcpy(cjzxn,hcjzxn,sizeof(cuComplex)*size_cjzx,cudaMemcpyHostToDevice);
    cudaMemcpy(cmxyn,hcmxyn,sizeof(cuComplex)*size_cjzy,cudaMemcpyHostToDevice);
    cudaMemcpy(cmxyp,hcmxyp,sizeof(cuComplex)*size_cjzy,cudaMemcpyHostToDevice);
    cudaMemcpy(cmyxn,hcmyxn,sizeof(cuComplex)*size_cjzx,cudaMemcpyHostToDevice);
    cudaMemcpy(cmyxp,hcmyxp,sizeof(cuComplex)*size_cjzx,cudaMemcpyHostToDevice);

    cudaMemcpy(cjzyn_tot,hcjzyn_tot,sizeof(cuComplex)*size_cjzy,cudaMemcpyHostToDevice);
    cudaMemcpy(cjzxp_tot,hcjzxp_tot,sizeof(cuComplex)*size_cjzx,cudaMemcpyHostToDevice);
    cudaMemcpy(cjzyp_tot,hcjzyp_tot,sizeof(cuComplex)*size_cjzy,cudaMemcpyHostToDevice);
    cudaMemcpy(cjzxn_tot,hcjzxn_tot,sizeof(cuComplex)*size_cjzx,cudaMemcpyHostToDevice);
    cudaMemcpy(cmxyn_tot,hcmxyn_tot,sizeof(cuComplex)*size_cjzy,cudaMemcpyHostToDevice);
    cudaMemcpy(cmxyp_tot,hcmxyp_tot,sizeof(cuComplex)*size_cjzy,cudaMemcpyHostToDevice);
    cudaMemcpy(cmyxn_tot,hcmyxn_tot,sizeof(cuComplex)*size_cjzx,cudaMemcpyHostToDevice);
    cudaMemcpy(cmyxp_tot,hcmyxp_tot,sizeof(cuComplex)*size_cjzx,cudaMemcpyHostToDevice);

    cudaMemcpy(dev_kex,kex,sizeof(float)*10,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kmx,kmx,sizeof(float)*10,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_aex,aex,sizeof(float)*10,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bex,bex,sizeof(float)*10,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bmx,bmx,sizeof(float)*10,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_amx,amx,sizeof(float)*10,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Ceze,Ceze,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Cezhy,Cezhy,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(error));
    }

    int*dev_i;
    cudaMalloc(&dev_i,sizeof(int));
    float test_Ez;

    dim3 gridNF2FF((int)ceil(size_NF2FF_total/512.0));
    dim3 blockNF2FF(512);
    //system("PAUSE");
    //time_t start,end;
    float test_Ez_2;
    //time(&start);
    //error = cudaGetLastError();
    //printf("%s\n",cudaGetErrorString(error));
    //ofstream myfile;
    //ifstream fields;
    //fields.open("Field_snapshot.txt");
    //if(!fields)
    //{
    //	cout<<"Couldn't open file"<<endl;
    //	cin>>test_Ez_2;
    //}
    //else
    //{
    //	cout<<"File opening success!"<<endl;
    //}
    //myfile.open("Scattered_Ez_at_f2.txt");
    //myfile<<"foci1      foci2"<<endl;
    //ostringstream convert;
    //int x,y;
    /* The calculation part! */
    for(int i=0;i<number_of_time_steps;i++)
    {
        cudaMemcpy(dev_i,&i,sizeof(int),cudaMemcpyHostToDevice);

        //H_inc_update<<<grid,block>>>(dev_Hy_inc,dev_Hx_inc,dev_Ezic,dev_bmx,dev_Psi_hyx_inc,dev_amx,dev_bmx,dev_amx,dev_Psi_hxy_inc,dev_kmx);
        //E_inc_update<<<grid,block>>>(dev_i,dev_Hy_inc,dev_Hx_inc,dev_Psi_ezx_inc,dev_aex,dev_aex,dev_bex,dev_bex,dev_Psi_ezy_inc,dev_kex,dev_Ezip,dev_Ezic);

        H_field_update<<<grid,block>>>(dev_Hy,dev_Hx,dev_Ez,dev_bmx,dev_Psi_hyx,dev_amx,dev_bmx,dev_amx,dev_Psi_hxy,dev_kmx);

        // H_field_update(float*dev_Hy,float*dev_Hx,float*dev_Ez,float*dev_bmx,float*dev_Psi_hyx,float*dev_amx,float*dev_bmy,float*dev_amy,float*dev_Psi_hxy,float*kex,float*Chxez,float*Chyez,float*Chyh,float*Chxh)
        //cudaMemcpy(dev_i,&i,sizeof(int),cudaMemcpyHostToDevice);
        E_field_update<<<grid,block>>>(dev_i,dev_Ez,dev_Hy,dev_Hx,dev_Psi_ezx,dev_aex,dev_aex,dev_bex,dev_bex,dev_Psi_ezy,dev_kex,dev_Cezhy,dev_Cezhy,dev_Ceze,dev_Cezeip,dev_Cezeic,dev_Ezic,dev_Ezip);

        calculate_JandM<<<gridNF2FF,blockNF2FF>>>(dev_freq, dev_i,dev_Ez,dev_Hy,dev_Hx,cjzxp,cjzyp,cjzxn,cjzyn,cmxyp,cmyxp,cmxyn,cmyxn);
        //float* f,int* timestep,float*dev_Ez,float*dev_Hy,float*dev_Hx,cuComplex *cjzxp,cuComplex *cjzyp,cuComplex*cjzxn,cuComplex*cjzyn,cuComplex*cmxyp,cuComplex*cmyxp,cuComplex*cmxyn,cuComplex*cmyxn,float*dev_Ezic,float*dev_Ezip,float*dev_Hx_inc,float*dev_Hy_inc
        //if(isscattering)
        //{
        //calculate_JandM_total<<<gridNF2FF,blockNF2FF>>>(dev_freq,dev_i,dev_Ez,dev_Hy,dev_Hx,cjzxp_tot,cjzyp_tot,cjzxn_tot,cjzyn_tot,cmxyp_tot,cmyxp_tot,cmxyn_tot,cmyxn_tot,dev_Ezic,dev_Ezip,dev_Hx_inc,dev_Hy_inc);
        //}
        //		unsigned char green = 128+127*buffer_Ez/0.4;
        /*ptr[offset].x = 0;
          ptr[offset].y = green;
          ptr[offset].z = 0;
          ptr[offset].w = 255;*///OpenGL stuff
        //	cudaMemcpy(&test_Ez,&dev_Ez[getCell(nx/2,ny/2,nx+1)],sizeof(float),cudaMemcpyDeviceToHost);
        //cudaMemcpy(&test_Ez,(dev_Ez+5000),sizeof(float),cudaMemcpyDeviceToHost);
        //cudaMemcpy(&test_Ez_2,&dev_Ez[getCell(nx/2-100,ny/2,nx+1)],sizeof(float),cudaMemcpyDeviceToHost);
        //cudaMemcpy(&test_Ez,(dev_Ez+getCell(nx/2,ny/2,nx+1)),sizeof(float),cudaMemcpyDeviceToHost);
        //cout<<"Ez (V/m) "<<test_Ez<<" "<<i<<endl;
        //myfile<<test_Ez<<"          "<<test_Ez_2<<endl;
        //if(i==0)
        //{
        //	//cudaMemcpy(Ez,dev_Ez,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyDeviceToHost);
        //	for(int ind = 0;ind<(nx+1)*(ny+1);ind++)
        //	{
        //	fields>>test_Ez;
        //	Ez[ind] = Ez[ind]-test_Ez;// before this operation, Ez[ind] is the total field.  After this step, Ez[ind] is just the scattered field
        //	}
        //	for(int x = 0;x<=nx;x++)
        //	{
        //		for(int y =0;y<=ny;y++)
        //		{
        //		//Temp.Red = 255*((eps_r_z[getCell(x,y,nx+1)]-1))/25;
        //		Temp.Red =0;
        //		Temp.Green = (ebmpBYTE)(127+128*(Ez[getCell(x,y,nx+1)])/(1));
        //	//	Temp.Green = 0;
        //	//	if((x == f2x)||(y==f2y))
        //		//{
        //		//	Temp.Blue = (ebmpBYTE)254;
        //		//}
        //		//else
        //		//{
        //			Temp.Blue = 255*sqrt(sigma_e_z[getCell(x,y,nx+1)]-1)/sqrt(10000.0);
        //		//}
        //		//Temp.Blue = 0; 
        //		Temp.Alpha = 0;
        //		/*if((sqrt(((float)x-f1x)*((float)x-f1x)+((float)y-f1y)*((float)y-f1y))+sqrt(((float)x-f2x)*((float)x-f2x)+((float)y-f2y)*((float)y-f2y)))>500)
        //		{
        //			Temp.Green = (ebmpBYTE)0;
        //		}*/
        //		Scattered_Field_snapshot.SetPixel(x,y,Temp);
        //	

        //		}
        //	}
        //	for(y  = 0;y<=nx;y++)
        //	{
        //		myfile<<Ez[getCell(f2x,y,nx+1)]<<" ";
        //	}
        //	myfile.close();

        //	
        //	Scattered_Field_snapshot.WriteToFile("Target_image.bmp");
        //}

        //if(!(i%20 - 4))
        //{
        //	cudaMemcpy(Ez,dev_Ez,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyDeviceToHost);
        //	for(int x = 0;x<=nx;x++)
        //	{
        //		for(int y = 0;y<=ny;y++)
        //		{
        //			

        //			radius = sqrt((x-nx/2)*dx*(x-nx/2)*dx+(y-ny/2)*dy*dy*(y-ny/2));
        //			Temp.Red = (ebmpBYTE)255*((eps_r_z[getCell(x,y,nx+1)]-1)/60);
        //			Temp.Green = (ebmpBYTE)(127+128*(Ez[getCell(x,y,nx+1)])/(0.2));
        //			if(isOnxp(x)||isOnyp(x,y)||isOnyn(x,y)||isOnxn(x))
        //			{
        //				Temp.Green = 0;
        //			}
        //			Temp.Blue = 0;
        //			
        //			//if(radius>breast_radius)
        //			//{
        //			//	Temp.Blue = 50;
        //			//}
        //			Temp.Alpha =0;
        //			Output_Image.SetPixel(x,y,Temp);
        //			
        //		
        //		}
        //	}
        //			ostringstream convert;
        //			convert << i/20;
        //			outputfilename = "Current_source_times_r_"+convert.str()+".bmp";
        //			
        //			Output_Image.WriteToFile(outputfilename.c_str());
        //}

        //cout<<"Ez (V/m) "<<test_Ez<<" "<<i<<endl;
        //cout<<fwf( i,nx, ny, 0)<<endl;
    }
    cudaMemcpy(hcjzyn,cjzyn,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcjzxp,cjzxp,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcjzyp,cjzyp,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcjzxn,cjzxn,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcmxyn,cmxyn,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcmyxp,cmyxp,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcmxyp,cmxyp,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcmyxn,cmyxn,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);

    cudaMemcpy(hcjzyn_tot,cjzyn_tot,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcjzxp_tot,cjzxp_tot,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcjzyp_tot,cjzyp_tot,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcjzxn_tot,cjzxn_tot,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcmxyn_tot,cmxyn_tot,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcmyxp_tot,cmyxp_tot,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcmxyp_tot,cmxyp_tot,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
    cudaMemcpy(hcmyxn_tot,cmyxn_tot,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);

    cuComplex *L,*N;
    for(int i=0;i<numberofobservationangles;i++)
    {
        //cout<<"hcmyxp = "<<hcmyxp[i].r<<" + i"<<hcmyxp[i].i<<endl;
    }

    L  = (cuComplex*)malloc(sizeof(cuComplex)*size_NF2FF_total);
    N = (cuComplex*)malloc(sizeof(cuComplex)*size_NF2FF_total);

    CJ_Init(L,size_NF2FF_total);
    CJ_Init(N,size_NF2FF_total);

    N2FPostProcess(D, freq,N,L,hcjzxp,hcjzyp,hcjzxn,hcjzyn,hcmxyp,hcmyxp,hcmxyn,hcmyxn);
    CJ_Init(L,size_NF2FF_total);
    CJ_Init(N,size_NF2FF_total);
    D_tot = (float*)malloc(sizeof(float)*numberofobservationangles);
    N2FPostProcess(D_tot,freq,N,L,hcjzxp_tot,hcjzyp_tot,hcjzxn_tot,hcjzyn_tot,hcmxyp_tot,hcmyxp_tot,hcmxyn_tot,hcmyxn_tot);
    float measurement[numberofobservationangles] = {0.38446 , 0.362389 , 0.309065 , 0.237687 , 0.162638 , 0.101565 , 0.0642376 , 0.0457471 , 0.0406768 , 0.0462104 , 0.0534992 , 0.0586805 , 0.0681197 , 0.0845823 , 0.105639 , 0.130494 , 0.15567 , 0.169704 , 0.162106 , 0.135797 , 0.102823 , 0.0717831 , 0.0478674 , 0.0364377 , 0.0385978 , 0.0501895 , 0.067232 , 0.0870665 , 0.10573 , 0.118834 , 0.123803 , 0.11963 , 0.106446 , 0.087042 , 0.0667677 , 0.0498503 , 0.0385427 , 0.0365824 , 0.0478755 , 0.0714402 , 0.102585 , 0.135752 , 0.161719 , 0.169557 , 0.156789 , 0.13172 , 0.104872 , 0.0826252 , 0.0674893 , 0.0588946 , 0.0528211 , 0.0459529 , 0.0410625 , 0.0449233 , 0.0647455 , 0.105435 , 0.165446 , 0.237711 , 0.310644 , 0.365682 };
    //measurement = (float*)malloc(sizeof(float)*numberofobservationangles);

    //float Phi;

    //	for(int i = 0;i<numberofobservationangles;i++)
    //	{
    //	Phi = 360*(float)i/numberofobservationangles;
    //	cout<<"D "<<D[i]<<" Phi = "<<Phi<<endl;
    //}
    //	for(int i=0;i<numberofobservationangles;i++)
    //	{
    //	cout<<"L = "<<L[i].r<<" + i"<<L[i].i<<" "<<i<<endl;
    //	}
    //for(int i=0;i<numberofobservationangles;i++)
    //	{
    //	cout<<"N = "<<N[i].r<<" + i"<<N[i].i<<" "<<i<<endl;
    //	}

    float fit;
    fit=fitness(D,numberofobservationangles, measurement);
    //ofstream D_file;
    //ofstream D_tot_file;
    //if(isscattering)
    //{
    //if(isPW)
    //	{
    //		D_tot_file.open("Total_fieldPW.txt");
    //		D_file.open("RCSPW.txt");
    //	}
    //	else
    //	{
    //		D_tot_file.open("Total_field.txt");
    //		D_file.open("RCS.txt");
    //	}

    //	for(int i = 0;i<numberofobservationangles;i++)
    //		{
    //			D_file<<D[i]<<" , ";
    //			D_tot_file<<D_tot[i]<<" ";

    //		}
    //	}
    //else
    //	{
    //	D_file.open("Directivity.txt");
    //	for(int i = 0;i<numberofobservationangles;i++)
    //	{
    //D_file<<D[i]<<" , ";
    //	}
    //}

    /*cudaMemcpy(Hy,dev_Hy,sizeof(float)*(nx)*(ny),cudaMemcpyDeviceToHost);
      for(i = 0; i < (nx+1);i++) {
      for(j = 0; j <(ny+1);j++) {
      cout << Ez[getCell(i,j,nx+1)] << " ";
      }
      cout << endl;
      }
      cout << endl;*/
    //cout<<" fitness = "<<fit<<endl;
    error = cudaGetLastError();
    //if(error != cudaSuccess) {
    //	printf("%s\n",cudaGetErrorString(error));
    //}
    //time(&end);
    //double dif = difftime(end,start);
    //cout<<"It took "<<dif<<" seconds"<<endl;
    //system("PAUSE");

    free(Ceze);
    free(Cezhy);
    free(Cezhx);
    free(Ez);
    free(eps_r_z);
    free(sigma_e_z);
    free(Hy);
    free(Hx);
    free(kex);
    free(aex);
    free(bex);
    free(amx);
    free(bmx);
    free(alpha_e);
    free(alpha_m);
    free(sigma_e_pml);
    free(sigma_m_pml);
    free(Psi_ezy);
    free(Psi_ezx);
    free(Psi_hyx);
    free(Psi_hxy);
    free(kmx);
    free(D);
    free(D_tot);
    free(Ezip);
    free(Ezic);
    free(Hy_inc);
    free(Hx_inc);
    free(Psi_ezy_inc);
    free(Psi_ezx_inc);
    free(Psi_hyx_inc);
    free(Psi_hxy_inc);
    free(hcjzxp);
    free(hcjzyp);
    free(hcjzxn);
    free(hcjzyn);
    free(hcmxyp);
    free(hcmyxp);
    free(hcmxyn);
    free(hcmyxn);
    free(hcjzxp_tot);
    free(hcjzyp_tot);
    free(hcjzxn_tot);
    free(hcjzyn_tot);
    free(hcmxyp_tot);
    free(hcmyxp_tot);
    free(hcmxyn_tot);
    free(hcmyxn_tot);
    free(Cezeic);
    free(Cezeip);
    free(L);
    free(N);
    //free(measurement);
    //float *Cezeic = (float*)malloc((sizeof(float))*(nx+1)*(ny+1));
    //float *Cezeip = (float*)malloc((sizeof(float))*(nx+1)*(ny+1));

    //float*Ceze,*Cezhy,*Cezhx,*dev_Cezeic,*dev_Cezeip,*Ez,*eps_r_z,*sigma_e_z,*Hy,*Hx,
    //	*kex,*aex,*bex,*amx,*bmx,*alpha_e,*alpha_m,*sigma_e_pml,*sigma_m_pml
    //	,*Psi_ezy,*Psi_ezx,*Psi_hyx,*Psi_hxy,*kmx;//*Cezj later if using loop current source
    //float* dev_sigma_e_z,*dev_eps_r_z;
    //float freq = center_freq;
    //float *dev_freq,*D,*D_tot;
    //float* Ezip,*Ezic,*dev_Ezip,*dev_Ezic,*Hy_inc,*Hx_inc,*dev_Hy_inc,*dev_Hx_inc,*dev_Psi_ezy_inc,*dev_Psi_ezx_inc,*dev_Psi_hyx_inc,*dev_Psi_hxy_inc,
    //	*Psi_ezy_inc,*Psi_ezx_inc,*Psi_hyx_inc,*Psi_hxy_inc;
    //
    //cuComplex *cjzxp,*cjzyp,*cjzxn,*cjzyn,*cmxyp,*cmyxp,*cmxyn,*cmyxn,*cjzxp_tot,*cjzyp_tot,*cjzxn_tot,*cjzyn_tot,*cmxyp_tot,*cmyxp_tot,*cmxyn_tot,*cmyxn_tot;
    //cuComplex *hcjzxp,*hcjzyp,*hcjzxn,*hcjzyn,*hcmxyp,*hcmyxp,*hcmxyn,*hcmyxn,*hcjzxp_tot,*hcjzyp_tot,*hcjzxn_tot,*hcjzyn_tot,*hcmxyp_tot,*hcmyxp_tot,*hcmxyn_tot
    //	,*hcmyxn_tot;


    cudaFree(dev_Cezeic);
    cudaFree(dev_Cezeip);
    cudaFree(dev_sigma_e_z);
    cudaFree(dev_eps_r_z);
    cudaFree(dev_freq);
    cudaFree(dev_Ezip);
    cudaFree(dev_Ezic);
    cudaFree(dev_Hy_inc);
    cudaFree(dev_Hx_inc);
    cudaFree(dev_Psi_ezy_inc);
    cudaFree(dev_Psi_ezx_inc);
    cudaFree(dev_Psi_hyx_inc);
    cudaFree(dev_Psi_hxy_inc);
    cudaFree(cjzxp);
    cudaFree(cjzyp);
    cudaFree(cjzxn);
    cudaFree(cjzyn);
    cudaFree(cmxyp);
    cudaFree(cmyxp);
    cudaFree(cmxyn);
    cudaFree(cmyxn);
    cudaFree(cjzxp_tot);
    cudaFree(cjzyp_tot);
    cudaFree(cjzxn_tot);
    cudaFree(cjzyn_tot);
    cudaFree(cmxyp_tot);
    cudaFree(cmyxp_tot);
    cudaFree(cmxyn_tot);
    cudaFree(cmyxn_tot);
    cudaFree(dev_Ceze);
    cudaFree(dev_Cezhy);
    cudaFree(dev_Cezhx);

    cudaFree(dev_bex);
    cudaFree(dev_aex);
    cudaFree(dev_bmx);
    cudaFree(dev_amx);
    cudaFree(dev_kex);
    cudaFree(dev_kmx);
    cudaFree(dev_Ez);
    cudaFree(dev_Hy);
    cudaFree(dev_Hx);
    cudaFree(dev_Psi_ezy);
    cudaFree(dev_Psi_ezx);
    cudaFree(dev_Psi_hyx);
    cudaFree(dev_Psi_hxy);
    //float*dev_Ceze,*dev_Cezhy,*dev_Cezhx,*dev_Jz,*dev_bex,*dev_aex,*dev_bmx,*dev_amx,*dev_kex,*dev_kmx;//dev_Cezj if using loop current source
    //float *dev_Ez,*dev_Hy,*dev_Hx;

    //float*dev_Psi_ezy,*dev_Psi_ezx,*dev_Psi_hyx,*dev_Psi_hxy;

    cout << "fitness is: " << fit << endl;
    return (double)fit;
}

__global__ void scattered_parameter_init(float*eps_r_z,float*sigma_e_z,float*Cezeic,float*Cezeip)
{
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    int y=threadIdx.y+blockDim.y*blockIdx.y;
    if(x<(nx+1)&&y<(ny+1))
    {
        Cezeic[dgetCell(x,y,nx+1)] = (2*(eps0-eps0*eps_r_z[dgetCell(x,y,nx+1)])-sigma_e_z[dgetCell(x,y,nx+1)]*dt)/(2*eps0*eps_r_z[dgetCell(x,y,nx+1)]+sigma_e_z[dgetCell(x,y,nx+1)]*dt);
        Cezeip[dgetCell(x,y,nx+1)] = -1*(2*(eps0-eps0*eps_r_z[dgetCell(x,y,nx+1)])+sigma_e_z[dgetCell(x,y,nx+1)]*dt)/(2*eps0*eps_r_z[dgetCell(x,y,nx+1)]+sigma_e_z[dgetCell(x,y,nx+1)]*dt);

    }
}

int getCell(int x, int y,int size)//size will just be the width in the x dimension of the array.
{
    return x+y*size;
}

float* Make2DfloatArray(int arraySizeX, int arraySizeY)
{
    float* theArray;
    theArray = (float*) malloc(arraySizeX*arraySizeY*sizeof(float*));

    return theArray;
} 

void waveform_time_init(float*time1)
{

    int size = number_of_time_steps;
    for(int i = 0;i<size;i++) 
    {
        time1[i]=(float)i*dt;
    }
}

void Jz_waveform(float * time,float*Jz_impressed)
{
    float w = 2*PI*center_freq;//center_freq is the frequency
    for(int i = 0;i<number_of_time_steps;i++)
    {
        Jz_impressed[i]= 10*sin(w*time[i]);
        //Jz_impressed[i]=exp(-1*((time[i]-2e-10)/5e-11)*(time[i]-2e-10)/(5e-11));

    }
}

void Ceze_init(float * eps_r_z, float* sig_e_z, float* Ceze)
{
    int size = nx+1;
    for(int j=0;j<ny+1;j++)
    {
        for(int i=0;i<size;i++)
        {
            Ceze[getCell(i,j,nx+1)] = (2*eps_r_z[getCell(i,j,nx+1)]*eps0-dt*sig_e_z[getCell(i,j,nx+1)])/(2*eps_r_z[getCell(i,j,nx+1)]*eps0+dt*sig_e_z[getCell(i,j,nx+1)]);
        }
    }
}

void Cezhy_init(float*eps_r_z, float* sigma_e_z,float* Cezhy,float*kex)
{
    int size = nx+1;
    for(int j =0;j<ny+1;j++)
    {
        for(int i=0;i<size;i++)
        {
            Cezhy[getCell(i,j,size)] = (2*dt/dx)/(2*eps_r_z[getCell(i,j,size)]*eps0+dt*sigma_e_z[getCell(i,j,size)]);

        }
    }
}

void Cezhx_init(float* eps_r_z,float*sigma_e_z,float*Cezhx,float*kex)
{
    int size=nx+1;
    for(int j=0;j<ny+1;j++)
    {
        for(int i =0;i<nx+1;i++)
        {
            Cezhx[getCell(i,j,size)]=(2*dt/dy)/(2*eps_r_z[getCell(i,j,size)]*eps0+dt*sigma_e_z[getCell(i,j,size)]);

        }
    }
}

void Cezj_init(float*eps_r_z,float*sigma_e_z,float*Cezj)
{
    int size =nx+1;
    for(int j=0;j<ny+1;j++)
    {
        for(int i=0;i<nx+1;i++)
        {
            Cezj[getCell(i,j,size)] = (-2*dt)/(2*eps_r_z[getCell(i,j,size)]*eps0+dt*sigma_e_z[getCell(i,j,size)]);

        }
    }
}

void Ez_init(float*Ez)
{
    int size=nx+1;
    for(int j = 0;j<ny+1;j++)
    {
        for(int i = 0;i<nx+1;i++)
        {
            Ez[getCell(i,j,size)] = (float)0;
        }
    }
}

/*void Jz_init(float*Jz)
  {
  for(int j =0;j<ny+1;j++)
  {
  for(int i = 0;i<nx+1;i++)
  {
  Jz[getCell(i,j,nx+1)] = 0;
  }
  }
  }*/

void Chyh_init(float*mu_r_y,float*sigma_m_y,float*Chyh)
{
    int size=nx;
    for(int i = 0;i<nx;i++)
        for(int j =0;j<ny;j++)
        {
            {
                Chyh[getCell(i,j,size)] = (2*mu_r_y[getCell(i,j,size)]*mu0-dt*sigma_m_y[getCell(i,j,size)])/(2*mu_r_y[getCell(i,j,size)]*mu0+dt*sigma_m_y[getCell(i,j,size)]);
            }
        }
}

void Chxh_init(float*mu_r_x,float*sigma_m_x,float*Chxh)
{
    int size=nx;
    for(int i = 0;i<nx;i++)
        for(int j =0;j<ny;j++)
        {
            {
                Chxh[getCell(i,j,size)] = (2*mu_r_x[getCell(i,j,size)]*mu0-dt*sigma_m_x[getCell(i,j,size)])/(2*mu_r_x[getCell(i,j,size)]*mu0+dt*sigma_m_x[getCell(i,j,size)]);
            }
        }
}

void Chyez_init(float*mu_r_y,float*sigma_m_y,float*Chyez)
{
    int size = nx;
    for(int j =0;j<ny;j++)
    {
        for(int i = 0;i<size;i++)
        {
            Chyez[getCell(i,j,size)] = (2*dt/dx)/(2*mu_r_y[getCell(i,j,size)]*mu0+dt*sigma_m_y[getCell(i,j,size)]);
        }
    }
}

void Chxez_init(float*mu_r_x,float*sigma_m_x,float*Chxez)
{
    int size = nx;
    for(int j =0;j<ny;j++)
    {
        for(int i = 0;i<size;i++)
        {
            Chxez[getCell(i,j,size)] = (2*dt/dy)/(2*mu_r_x[getCell(i,j,size)]*mu0+dt*sigma_m_x[getCell(i,j,size)]);
        }
    }
}

/*void Chym_init(float*mu_r_y,float*sigma_m_y,float*Chym)
  {
  int size = nx;
  for(int j =0;j<ny;j++)
  {
  for(int i = 0;i<size;i++)
  {
  Chym[getCell(i,j,size)] = (-2*dt)/(2*mu_r_y[getCell(i,j,size)]*mu0+dt*sigma_m_y[getCell(i,j,size)]);

  }
  }
  }
  void Chxm_init(float*mu_r_x,float*sigma_m_x,float*Chxm)
  {
  int size = nx;
  for(int j =0;j<ny;j++)
  {
  for(int i = 0;i<size;i++)
  {
  Chxm[getCell(i,j,size)] = (-2*dt)/(2*mu_r_x[getCell(i,j,size)]*mu0+dt*sigma_m_x[getCell(i,j,size)]);

  }
  }
  }*/

void eps_r_z_init(float * eps_r_z,const vector<float> &argument)
{
    int size = nx+1;
    float radius,tumor_radius,tumor_radius_2,tumor_radius_3;
    for(int j =0;j<ny+1;j++)
    {
        for(int i = 0;i<nx+1;i++)
        {


            eps_r_z[getCell(i,j,size)] = 1;
            radius = sqrt(pow( ((float)i-nx/2)*dx,2) + pow( ((float)j-ny/2)*dy,2));
            if(radius<=breast_radius)
            {
                eps_r_z[getCell(i,j,size)] = (float)argument.at(getOptimizationCell(i,j));
                //eps_r_z[getCell(i,j,size)] = 10;

            }


            //radius = sqrt(((float)i-(target_x))*dx*((float)i-(target_x))*dx+((float)j-(target_y))*dy*((float)j-target_y)*dy);
            //tumor_radius = sqrt(((float)i-(target_x-25))*dx*((float)i-(target_x-25))*dx+((float)j-(target_y+50))*dy*((float)j-(target_y+50))*dy);
            //tumor_radius_2 = sqrt(((float)i-(target_x+25))*dx*((float)i-(target_x+25))*dx+((float)j-(target_y-50))*dy*((float)j-(target_y-50))*dy);
            //tumor_radius_3 = sqrt(((float)i-(target_x-25))*dx*((float)i-(target_x-25))*dx+((float)j-(target_y-25))*dy*((float)j-(target_y-25))*dy);
            //if(radius>breast_radius)
            //{
            //	eps_r_z[getCell(i,j,size)] = 1;

            //}
            //
            //if(radius>breast_radius)
            //{
            //	eps_r_z[getCell(i,j,size)] = 1;
            //	//cout<<"eps_r_z = "<<eps_r_z[getCell(i,j,size)]<<" (i,j) = ("<<i<<","<<j<<")"<<endl;
            //}
            //else if(i>=(nx/2-108)&&i<(nx/2+108)&&j>=(ny/2-108)&&j<(ny/2+108))
            //{
            //	eps_r_z[getCell(i,j,size)] = (float)argument.at(getOptimizationCell(i,j));
            ////	cout<<"eps_r_z = "<<eps_r_z[getCell(i,j,size)]<<" (i,j) = ("<<i<<","<<j<<")"<<endl;
            //}

        }
    }
}

void sigma_e_z_init(float * sigma_e_z,float*sigma_e_pml, const vector<float> &argument)
{
    int size = nx+1;
    float radius;


    for(int j =0;j<ny+1;j++)
    {
        for(int i = 0;i<nx+1;i++)
        {
            sigma_e_z[getCell(i,j,size)] = 0;
            radius = sqrt(pow( ((float)i-nx/2)*dx,2) + pow( ((float)j-ny/2)*dy,2));
            if(radius<=breast_radius)
            {
                sigma_e_z[getCell(i,j,size)] = (float)argument.at(getOptimizationCell(i,j)+9*9);
                //eps_r_z[getCell(i,j,size)] = 10;

            }
            //if(sqrt(pow((float)i-f2x,2)*dx+pow((float)j-f2y,2)*dy)<(100*dx))
            //{
            //	sigma_e_z[getCell(i,j,size)] = 0.00;
            //}
            //else
            //{
            //	sigma_e_z[getCell(i,j,size)] = 0;
            //}
            //	radius = sqrt(((float)i-nx/2)*dx*((float)i-nx/2)*dx+((float)j-ny/2)*dy*((float)j-ny/2)*dy);
            //if(radius>breast_radius)
            //{
            //	sigma_e_z[getCell(i,j,size)] = 0;
            //	//cout<<"sigma_e_z = "<<sigma_e_z[getCell(i,j,size)]<<" (i,j) = ("<<i<<","<<j<<")"<<endl;
            //}
            //else if(i>=(nx/2-108)&&i<(nx/2+108)&&j>=(ny/2-108)&&j<(ny/2+108))
            //{
            //	sigma_e_z[getCell(i,j,size)] = (float)argument.at(getOptimizationCell(i,j)+9*9);//total of 81 optimization cells
            ////	cout<<"sigma_e_z = "<<sigma_e_z[getCell(i,j,size)]<<" (i,j) = ("<<i<<","<<j<<")"<<endl;
            //}

        }
    }
}

void Hy_init(float*Hy)
{
    int size = nx;
    for(int j=0;j<ny;j++)
    {
        for(int i = 0;i<size;i++)
        {
            Hy[getCell(i,j,size)] = 0;
        }
    }
}

void Hx_init(float*Hx)
{
    int size = nx;
    for(int j=0;j<ny;j++)
    {
        for(int i = 0;i<size;i++)
        {
            Hx[getCell(i,j,size)] = 0;
        }
    }
}

void My_init(float*My)
{
    int size = nx;
    for(int j=0;j<ny;j++)
    {
        for(int i = 0;i<size;i++)
        {
            My[getCell(i,j,size)] = 0;
        }
    }
}

void Mx_init(float*Mx)
{
    int size = nx;
    for(int j=0;j<ny;j++)
    {
        for(int i = 0;i<size;i++)
        {
            Mx[getCell(i,j,size)] = 0;
        }
    }
}

void mu_r_y_init(float*mu_r_y)
{
    int size = nx;
    for(int j=0;j<ny;j++)
    {
        for(int i = 0;i<size;i++)
        {
            mu_r_y[getCell(i,j,size)] =1.000;
        }
    }

}

void mu_r_x_init(float*mu_r_x)
{
    int size = nx;
    for(int j=0;j<ny;j++)
    {
        for(int i = 0;i<size;i++)
        {
            mu_r_x[getCell(i,j,size)]=1.000;
        }
    }

}

void sigma_m_y_init(float*sigma_m_y)
{
    int size = nx;
    for(int j=0;j<ny;j++)
    {
        for(int i = 0;i<size;i++)
        {
            sigma_m_y[getCell(i,j,size)] = 0;
        }
    }
}

void sigma_m_x_init(float*sigma_m_x)
{
    int size = nx;
    for(int j=0;j<ny;j++)
    {
        for(int i = 0;i<size;i++)
        {
            sigma_m_x[getCell(i,j,size)] = 0;
        }
    }
}

void C_Psi_ezy_init(float *C_Psi_ezy,float*Cezhx)
{
    int size = 20;
    for(int j = 0;j<ny;j++)
        for( int i =0;i<size;i++)
        {
            if(i<10)
            {
                C_Psi_ezy[getCell(i,j,size)]=dy*Cezhx[getCell(i,j,nx)];
            }
            else
            {
                C_Psi_ezy[getCell(i,j,size)]=dy*Cezhx[getCell(nx-20+i,j,nx)];
            }
        }
}

void C_Psi_ezx_init(float* C_Psi_ezx,float*Cezhy)
{
    int size_y=20;
    for(int j=0;j<size_y;j++)
    {
        for(int i=0;i<nx;i++)
        {
            if(j<10)
            {
                C_Psi_ezx[getCell(i,j,nx)] = dx*Cezhy[getCell(i,j,nx)];
            }
            else
            {
                C_Psi_ezx[getCell(i,j,nx)] = dx*Cezhy[getCell(i,ny-20+j,nx)];
            }
        }
    }
}

void C_Psi_hyx_init(float*C_Psi_hyx,float*Chyez)
{
    int size_x=20;
    for(int j=0;j<ny;j++)
    {
        for(int i=0;i<size_x;i++)
        {
            if(i<10)
            {
                C_Psi_hyx[getCell(i,j,size_x)]=dx*Chyez[getCell(i,j,nx)];
            }
            else
            {
                C_Psi_hyx[getCell(i,j,size_x)]=dx*Chyez[getCell(nx-20+i,j,nx)];
            }
        }
    }
}

void C_psi_hxy_init(float *C_Psi_hxy,float*Chxez)
{
    int size_y=20;
    for(int j=0;j<size_y;j++)
    {
        for(int i=0;i<nx;i++)
        {
            if(j<11)
            {
                C_Psi_hxy[getCell(i,j,nx)]=dy*Chxez[getCell(i,j,nx)];
            }
            else
            {
                C_Psi_hxy[getCell(i,j,nx)]=dy*Chxez[getCell(i,ny-20+j,nx)];
            }
        }
    }
}

void aex_init(float*aex,float*sigma_e_pml,float*kex,float*alpha_e_x,float*bex)
{
    int size=ncells;
    //aex[0]=0.0;
    //cout<<"aex[0] = "<<aex[0]<<endl;
    for(int i=0;i<size;i++)
    {
        aex[i]=((bex[i]-1)*sigma_e_pml[i])/(dx*(sigma_e_pml[i]*kex[i]+alpha_e_x[i]*kex[i]*kex[i]));
        //cout<<"aex["<<i<<"] = "<<aex[i]<<endl;
    }
}

void bex_init(float*bex ,float*sigma_e_pml,float*kex,float*alpha_e_x)
{
    int size=ncells;
    for(int i=0;i<size;i++)
    {
        bex[i]=exp(-1*(dt/eps0)*(sigma_e_pml[i]/kex[i]+alpha_e_x[i]));
    }
}

void aey_init(float*aey,float*sigma_e_pml,float*key,float*alpha_e_y,float*bey)
{
    for(int i=0;i>ncells;i++)
    {
        aey[i]=(bey[i]-1)*sigma_e_pml[i]/(dy*(sigma_e_pml[i]*key[i]+alpha_e_y[i]*key[i]*key[i]));
    }
}

void bey_init(float*bey,float*sigma_e_pml,float*key,float*alpha_e_y)
{
    int size=ncells;
    for(int i=0;i<size;i++)
    {
        bey[i]=exp(-1*(dt/eps0)*(sigma_e_pml[i]/key[i]+alpha_e_y[i]));
    }
}

void amy_init(float*amy,float*sigma_m_pml,float*kmy,float*alpha_m_y,float*bmy)
{
    int size=ncells;
    for(int i=0;i<size;i++)
    {
        amy[i]=(bmy[i]-1)*sigma_m_pml[i]/(dx*(sigma_m_pml[i]*kmy[i]+alpha_m_y[i]*kmy[i]*kmy[i]));
    }
}

void bmy_init(float*bmy,float*sigma_m_pml,float*kmy,float*alpha_m_y)
{
    int size=ncells;
    for(int i=0;i<size;i++)
    {
        bmy[i]=exp(-1*(dt/mu0)*(sigma_m_pml[i]/kmy[i]+alpha_m_y[i]));
    }
}

void amx_init(float*amx,float*sigma_m_pml,float*kmx,float*alpha_m_x,float*bmx)
{
    int size=ncells;

    //cout<<" amx = "<<amx[0]<<endl;
    //amx[0]=0.0;
    //cout<<" amx = "<<amx[0]<<endl;
    for(int i=0;i<size;i++)
    {
        amx[i]=(bmx[i]-1)*sigma_m_pml[i]/(dx*(sigma_m_pml[i]*kmx[i]+alpha_m_x[i]*kmx[i]*kmx[i]));
        //	cout<<" amx = "<<amx[i]<<endl;
    }
}

void bmx_init(float*bmx,float*sigma_m_pml,float*kmx,float*alpha_m_x)
{
    int size=10;
    float argument;
    //float constant;
    for(int i=0;i<size;i++)
    {
        //constant = dt/mu0;
        //cout<< "dt/mu0 = "<<constant<<endl;
        argument = -1*(dt/mu0)*((sigma_m_pml[i]/kmx[i])+alpha_m_x[i]);
        bmx[i]=exp(argument);
        //cout<<"argument of bmx = "<<argument<<endl;
        //cout<<"bmx = "<<bmx[i]<<endl;
    }
}

void alpha_e_init(float*alpha_e)
{
    float rho;
    int size=ncells;
    for(int i=0;i<ncells;i++)
    {
        rho = ((float)i+0.25)/ncells;
        alpha_e[i]=alpha_min+(alpha_max-alpha_min)*rho;
        //	cout<<"alpha_e = "<<alpha_e[i]<<endl;
    }
}

void alpha_m_init(float*alpha_e,float*alpha_m)
{
    int size=ncells;
    float rho;
    for(int i=0;i<size;i++)
    {
        rho = ((float)i+0.75)/ncells;
        alpha_m[i]=(mu0/eps0)*(alpha_min+(alpha_max-alpha_min)*rho);
        //cout<<"alpha_m = "<<alpha_m[i]<<endl;
    }
}

void k_e_init(float*k)
{
    int size=ncells;
    float rho;
    for(int i=0;i<size;i++)
    {
        rho = ((float)i+0.25)/ncells;
        k[i]=pow(rho,npml)*(kmax-1)+1;
        //cout<<"k ["<<i<<"]= "<<k[i]<<endl;

    }
}

void k_m_init(float*k)
{
    int size=ncells;
    float rho;
    for(int i=0;i<size;i++)
    {
        rho = ((float)i+0.75)/ncells;
        k[i]=pow(rho,npml)*(kmax-1)+1;
        //cout<<"k ["<<i<<"]= "<<k[i]<<endl;

    }
}

void sigma_e_pml_init(float* sigma_e_pml)  
{
    float sigma_max = (npml+1)/(150*PI*dx);
    int size = 10;
    float rho;
    for(int i=0;i<size;i++)
    {
        rho = ((float)i+0.25)/ncells;
        sigma_e_pml[i]=sigma_max*sigma_factor*pow(rho,npml);
        //cout<<"sigma_e_pml = "<<sigma_e_pml[i]<<endl;
    }
}

void sigma_m_pml_init(float*sigma_m_pml,float*sigma_e_pml)
{
    float rho;
    int size = 10;
    float sigma_max = (npml+1)/(150*PI*dx);
    for(int i=0;i<size;i++)
    {
        rho = ((float)i+0.75)/ncells;
        sigma_m_pml[i]=(mu0/eps0)*sigma_max*sigma_factor*pow(rho,npml);
        //cout<<"sigma_m_pml "<<sigma_m_pml[i]<<endl;
    }
}

void Psi_ezy_init(float*Psi_ezy)
{  
    int size=nx*20;
    for(int i=0;i<size;i++)
    {
        Psi_ezy[i]=0.0;
    }
}

void Psi_ezx_init(float*Psi_ezx)
{
    int size=ny*20;
    for(int i=0;i<size;i++)
    {
        Psi_ezx[i]=0.0;
    }
}

void Psi_hyx_init(float*Psi_hyx)
{
    int size=ny*20;
    for(int i=0;i<size;i++)
    {
        Psi_hyx[i]=0.0;
    }
}

void Psi_hxy_init(float*Psi_hxy)
{
    int size=nx*20;  
    for(int i=0;i<size;i++)
    {
        Psi_hxy[i]=0.0;
    }
}

void CJ_Init(cuComplex * cjzyn,int size)
{
    cuComplex nullComplex(0,0);
    for( int i =0; i<size;i++)
    {
        cjzyn[i] = nullComplex;
    }
}
