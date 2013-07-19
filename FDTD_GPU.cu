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

#include "FDTD_common.hxx"

void __cudaCheck(cudaError err, const char* file, const int line);
#define cudaCheck(err) __cudaCheck (err, __FILE__, __LINE__)

void __cudaCheckLastError(const char* errorMessage, const char* file, const int line);
#define cudaCheckLastError(msg) __cudaCheckLastError (msg, __FILE__, __LINE__)

void __cudaCheck(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

void __cudaCheckLastError(const char *errorMessage, const char *file, const int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

//#include <unistd.h>
//const cuComplex jcmpx (0.0, 1.0);
/*static void HandleError( cudaError_t err, const char *file,  int line ) {
  if (err != cudaSuccess) {
  printf( "%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );
  exit( EXIT_FAILURE );
  }
  }*/

/*
__device__ int dgetCell(int x, int y, int size) {
    return x +y*size;
}

int getCell(int x, int y, int size) {
    return x +y*size;
}
*/


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

            cjzyn[index] += Hx*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt);  //cjzyn and cmxyn need to have nx-2*NF2FFbound-2 elements
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


__global__ void E_field_update(int *i,float*dev_Ez,float*dev_Hy,float*dev_Hx,float*dev_Psi_ezx,float*dev_aex,float*dev_aey,float*dev_bex,float*dev_bey,float*dev_Psi_ezy,float*kex,float *Ceze,float*Cezhy,float*Cezhx,float*Cezeip,float*Cezeic,float*Phi,float *dev_Jp,float *dev_Cezjp,float *dev_Beta_p)
{
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    int y=threadIdx.y+blockDim.y*blockIdx.y;
    //  int offset = x+y*blockDim.x*gridDim.x;
    float buffer_Ez;
    //float Ceh = (dt/dx)/(eps0);
    float Cezj = -dt/eps0;
    float length_offset;
    if(x<=nx&&y<=ny)
    {

        //if(x==0||x==nx||y==0||y==ny)
        if(x==nx||y==ny||x==0||y==0)
        {
            buffer_Ez=0.0;
        }
        else
        {
                
                buffer_Ez = Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])
                    -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)])
                    +Cezeic[dgetCell(x,y,nx+1)]*fwf((float)(*i)+0.5,x-nx/2,y-ny/2,*Phi,-breast_radius)
                    +Cezeip[dgetCell(x,y,nx+1)]*fwf((float)(*i)-0.5,x-nx/2,y-ny/2,*Phi,-breast_radius)
                    -dev_Cezjp[dgetCell(x,y,nx+1)]*dev_Jp[dgetCell(x,y,nx+1)];
                
                dev_Jp[dgetCell(x,y,nx+1)] = Kp*dev_Jp[dgetCell(x,y,nx+1)] + (dev_Beta_p[dgetCell(x,y,nx+1)]/dt)*(buffer_Ez - dev_Ez[dgetCell(x,y,nx+1)]
                        + fwf((float)(*i)+0.5,x-nx/2,y-ny/2,*Phi,-breast_radius) - fwf((float)(*i)-0.5,x-nx/2,y-ny/2,*Phi,-breast_radius));//buffer_Ez is Ezs(n+1) dev_Ez is Ezs(n)



           if(x<=ncells&&x!=0) //This is pml stuff.
            {
                buffer_Ez =  Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])/kex[ncells-x]
                    -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)])/kex[ncells-x];
                dev_Psi_ezx[dgetCell(x-1,y-1,20)] = dev_bex[ncells-x]*dev_Psi_ezx[dgetCell(x-1,y-1,20)]+dev_aex[ncells-x]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)]);
                buffer_Ez += Cezhy[dgetCell(x,y,nx+1)]*dx*dev_Psi_ezx[dgetCell(x-1,y-1,2*ncells)];
            }
            if(x>=(nx-ncells)&&x!=nx)
            {
                buffer_Ez =  Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])/kex[x-nx+ncells]
                    -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)])/kex[x-nx+ncells];
                dev_Psi_ezx[dgetCell(x-nx+20,y-1,20)]=dev_bex[x-nx+ncells]*dev_Psi_ezx[dgetCell(x-nx+20,y-1,20)]+dev_aex[x-nx+ncells]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)]);
                buffer_Ez+=Cezhy[dgetCell(x,y,nx+1)]*dx*dev_Psi_ezx[dgetCell(x-nx+20,y-1,2*ncells)];
            }
            if(y<=ncells&&y!=0)
            {
                buffer_Ez =  Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])/kex[ncells-y]
                    -Cezhx[dgetCell(x,y,nx+1)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)])/kex[ncells-y];
                dev_Psi_ezy[dgetCell(x-1,y-1,nx)]=dev_bey[(ncells-y)]*dev_Psi_ezy[dgetCell(x-1,y-1,nx)]+dev_aey[(ncells-y)]*(dev_Hx[dgetCell(x,y,nx)]-dev_Hx[dgetCell(x,y-1,nx)]);
                buffer_Ez-=Cezhx[dgetCell(x,y,nx+1)]*dy*dev_Psi_ezy[dgetCell(x-1,y-1,nx)];
            }
            if(y>=(ny-ncells)&&y!=ny)
            {
                buffer_Ez =  Ceze[dgetCell(x,y,nx+1)]*dev_Ez[dgetCell(x,y,nx+1)]+Cezhy[dgetCell(x,y,nx+1)]*(dev_Hy[dgetCell(x,y,nx)]-dev_Hy[dgetCell(x-1,y,nx)])/kex[y-ny+ncells]
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

__global__ void Field_reset(float* Ez, float* Hy, float* Hx, float* Psi_ezy,float* Psi_ezx,float* Psi_hyx,float* Psi_hxy,cuComplex*cjzyn,cuComplex*cjzxp,cuComplex*cjzyp,cuComplex*cjzxn,cuComplex*cmxyn,cuComplex*cmyxp,cuComplex*cmxyp,cuComplex*cmyxn,float *dev_Jp)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    int index = x + y*blockDim.x*gridDim.x;
    if(x<=ncells&&x!=0)
    {
        Psi_ezx[dgetCell(x-1,y-1,20)] =0;
    }
    if(x>=(nx-ncells)&&x!=nx)
    {
        Psi_ezx[dgetCell(x-nx+20,y-1,20)]=0;
    }
    if(y<=ncells&&y!=0)
    {
        Psi_ezy[dgetCell(x-1,y-1,nx)]=0;
    }
    if(y>=(ny-ncells)&&y!=ny)
    {
        Psi_ezy[dgetCell(x-1,y-ny+20,nx)]=0;
    }
    if(x<ncells)
    {

        Psi_hyx[dgetCell(x,y,20)]=0;
    } 
    if(x>=(nx-ncells))
    {
        Psi_hyx[dgetCell(x-nx+20,y,2*ncells)]=0.0;
    }
    if(y<ncells)
    {
        Psi_hxy[dgetCell(x,y,nx)]=0.0;
    }
    if(y>=(ny-ncells))
    {
        Psi_hxy[dgetCell(x,y-ny+20,nx)]=0.0;
    }
    if(x<=nx&&y<=ny)
    {
        Ez[dgetCell(x,y,nx+1)] = 0.0;
        dev_Jp[dgetCell(x,y,nx+1)] = 0.0;
    }
    if(x<nx&&y<ny)
    {
        Hy[dgetCell(x,y,nx)] = 0.0;
        Hx[dgetCell(x,y,nx)] = 0.0;
    }

        if(index<size_cjzy)
        {
            cjzyp[index] = cuComplex(0,0);//cjzyp and cmxyp have nx - 2*NF2FFBoundary -2 elements
            cjzyn[index] = cuComplex(0,0);
            cmxyp[index] = cuComplex(0,0);
            cmxyn[index] = cuComplex(0,0);
        }
        if(index<size_cjzx)
        {
            cjzxp[index] = cuComplex(0,0);
            cjzxn[index] = cuComplex(0,0);
            cmyxp[index] = cuComplex(0,0);
            cmyxn[index] = cuComplex(0,0);
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


__host__ __device__ int getOptimizationCell(int x, int y)
{
    int x_coord,y_coord;
    x_coord = (x-(nx/2-(int)(breast_radius/dx)))/(2*breast_radius/(numpatches*dx));
    y_coord = (y-(ny/2-breast_radius/dy))/(2*breast_radius/(numpatches*dy));//the optimization space is 216 FDTD cells wide and high. //The optimization space is split into 25 by 25 optimization cells. 
    //each optimization cell has 24 by 24 FDTD cells within it. That's what the 108, 24 and 25 are about.  
    return x_coord+numpatches*y_coord;//The max return should be, numpatches*numpatches-1, hopefully.
}

void N2FPostProcess (float* D,float* P,float f,cuComplex *cjzxp,cuComplex *cjzyp,cuComplex *cjzxn,cuComplex *cjzyn,cuComplex *cmxyp,cuComplex *cmyxp,cuComplex *cmxyn,cuComplex *cmyxn)
{
    int indexofleg1 = nx-2*NF2FFdistfromboundary-2;
    int indexofleg2 = nx+ny-4*NF2FFdistfromboundary-2;
    int indexofleg3 = 2*nx+ny-6*NF2FFdistfromboundary-4;
    int maxindex = 2*nx-8*NF2FFdistfromboundary+2*ny-4;
    int x,y;
	cuComplex L(0,0);
	cuComplex N(0,0);
    float rhoprime;
    float Psi;
    cuComplex E(0,0);
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
            N+=-1.0*cjzyn[index]*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*dx;
            L+=-1.0*sin(Phi)*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*cmxyn[index]*dx;//Lphi = 

        }
        for(index = indexofleg1;index<indexofleg2;index++)
        {

            x = CPUgetxfromthreadIdNF2FF(index);
            y = CPUgetyfromthreadIdNF2FF(index);
            flx = (float)x+0.5;
            fly = (float)y;
            rhoprime = sqrt(pow((dx*(((float)nx/2)-1-flx)),2)+pow((dy*(((float)ny/2)-1-fly)),2)); 
            Psi = atan2(-1*((float)ny/2)+1+fly,(-1*((float)nx/2)+1+flx))-Phi;
            N+=-1.0*cjzxp[index-indexofleg1]*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*dy;
            L+=cos(Phi)*cmyxp[index-indexofleg1]*cuexp(1.0*jcmpx*k*rhoprime*cos(Psi))*dy;//L_phi = -Lxsin(phi)+Lycos(Phi) here we only have Ly
        }
        for(index=indexofleg2;index<indexofleg3;index++)
        {
            x = CPUgetxfromthreadIdNF2FF(index);
            y = CPUgetyfromthreadIdNF2FF(index);
            flx = (float)x;
            fly = (float)y + 0.5;
            rhoprime = sqrt(pow((dx*(((float)nx/2)-1-flx)),2)+pow((dy*(((float)ny/2)-1-fly)),2)); 
            Psi = atan2((-1*(float)ny/2+1+fly),(-1*((float)nx/2)+1+flx))-Phi;
            N+=-1.0*cjzyp[index-indexofleg2]*cuexp(jcmpx*k*rhoprime*cos(Psi))*dx;
            L+=-1.0*sin(Phi)*cmxyp[index-indexofleg2]*cuexp(jcmpx*k*rhoprime*cos(Psi))*dx;//
        }
        for(index = indexofleg3;index<maxindex;index++)
        {
            x = CPUgetxfromthreadIdNF2FF(index);
            y = CPUgetyfromthreadIdNF2FF(index);
            flx = (float)x+0.5;
            fly = (float)y;
            rhoprime = sqrt(pow(dx*(((float)nx/2)-1-flx),2)+pow((dy*(((float)ny/2)-1-fly)),2)); 
            Psi = atan2(-1*((float)ny/2)+1+fly,-1*(float)nx/2+1+flx)-Phi;
            N+=-1.0*cjzxn[index-indexofleg3]*cuexp(jcmpx*k*rhoprime*cos(Psi))*dy;
            L+= cos(Phi)*cmyxn[index-indexofleg3]*cuexp(jcmpx*k*rhoprime*cos(Psi))*dy;
        }
        D[Phi_index] = (k*k*cuabs(L+(float)eta0*N)*cuabs(L+(float)eta0*N)/((float)8*(float)PI*(float)eta0*Prad*33.329));//why 33.329?  I dunno, something is probably wrong with Prad.
        E = L+(float)eta0*N;
        P[Phi_index] = atan2(E.i,E.r);
        L = cuComplex(0,0);
        N = cuComplex(0,0);

    }

}

__host__ __device__ float del_X(float del_eps, int k)
{
    return del_eps*exp(-1*(float)k*dt/relaxation_time)*(1-exp(-dt/relaxation_time))*(1-exp(-dt/relaxation_time));

}
//static void draw_func(void){
//	glDrawPixels(nx,ny,GL_RGBA,GL_UNSIGNED_BYTE,0);
//	glutSwapBuffers;
//}

using namespace std;

__global__ void scattered_parameter_init(float*eps_infinity,float *sigma_e_z,float*Cezeic,float*Cezeip,float *Cezjp, float* dev_Beta_p);

double FDTD_GPU(const vector<double> &arguments) {
//    cout << "calculating FDTD GPU" << endl;

//    cudaSetDevice(0);

    vector<float> image;
    //This is setting the material parameters of the optimization cells.
    for (int lerp = 0; lerp < numpatches*numpatches; lerp++) {  //numpatches is the amount of optimization patches across.
       image.push_back((float)arguments.at(lerp));
    /*	if(lerp == numpatches/2){
        image.push_back(6.75);
    	}
       else
         {
         image.push_back(3.14);  //eps_infinity of fat
         }*/
    }

    for (int lerp = numpatches*numpatches; lerp < numpatches*numpatches* 2; lerp++) {
        image.push_back((float)arguments.at(lerp));
	
      /* if(lerp == numpatches/2 + numpatches*numpatches){
       image.push_back(40);// del_eps 
	
        }
       else
       {
           image.push_back(1.61);
       }*/

    }

    for (int lerp = numpatches*numpatches*2 ; lerp<numpatches*numpatches*3;lerp++){
        image.push_back((float)arguments.at(lerp));
	
    /*   if(lerp == numpatches/2 + numpatches*numpatches){
       image.push_back(0);// sigma 
	
       }
       else
       {
           image.push_back(0);
       }*/
    }



    cudaError_t error;
	float freq;
    int grid_x = int(ceil((float)nx / 22));
    int grid_y = int(ceil((float)ny / 22));

    dim3 grid(grid_x, grid_y);
    dim3 block(22, 22);
    float *Ez = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    float *eps_infinity = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    float *Cezhy = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    float *Ceze = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    //Cezj later if using loop current source
    //float *Cezj = (float*)malloc(sizeof(float)*(1+nx)*(1+ny)); // if using loop current source
    float *del_eps = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    float *Beta_p = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    float radius;//tumor_radius,tumor_radius_2,tumor_radius_3;
    float *sigma_e_z = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));


    /********************  Setting Material Parameters*************************/

    for (int j = 0; j < ny + 1; j++) {
        for (int i = 0; i < nx + 1; i++) {
            Ez[getCell(i,j,nx+1)] = (float)0;
            eps_infinity[getCell(i,j,nx+1)] = 1;
            del_eps[getCell(i,j,nx+1)] = 0;
            sigma_e_z[getCell(i,j,nx+1)] = 0;
            radius = sqrt(pow( ((float)i-nx/2)*dx,2) + pow( ((float)j-ny/2)*dy,2));

            //tumor_radius = sqrt(pow( ((float)i - target_x)*dx,2) + pow( ((float)j-target_y)*dy,2));
            if (radius <= breast_radius) {

                eps_infinity[getCell(i,j,nx+1)] = (float)image.at(getOptimizationCell(i,j)); //This is the line that should be uncommented if using as forward solver
                del_eps[getCell(i,j,nx+1)] = (float)image.at(getOptimizationCell(i,j)+numpatches*numpatches);
                sigma_e_z[getCell(i,j,nx+1)] = (float)image.at(getOptimizationCell(i,j)+numpatches*numpatches*2);

                //eps_infinity[getCell(i,j,nx+1)] = 10;
                //if(tumor_radius <= tumor_nx+1)//delete this if using as forward solver
                //{
                //	eps_infinity[getCell(i,j,nx+1)] = 60;
                //}
            }
            Beta_p[getCell(i,j,nx+1)] = eps0*del_eps[getCell(i,j,nx+1)]*dt/(relaxation_time*(1+dt/(2*relaxation_time)));
            Cezhy[getCell(i,j,nx+1)] = (2*dt/dx)/(2*eps_infinity[getCell(i,j,nx+1)]*eps0 + sigma_e_z[getCell(i,j,nx+1)]*dt + Beta_p[getCell(i,j,nx+1)]);
            Ceze[getCell(i,j,nx+1)] = (2*eps_infinity[getCell(i,j,nx+1)]*eps0 - sigma_e_z[getCell(i,j,nx+1)]*dt + Beta_p[getCell(i,j,nx+1)])/(2*eps_infinity[getCell(i,j,nx+1)]*eps0 + sigma_e_z[getCell(i,j,nx+1)]*dt + Beta_p[getCell(i,j,nx+1)]);
        }
    }

    /*******************************************************************/

    /***********************************Setting up PML layer *************************/
    float *sigma_e_pml = (float*)malloc(sizeof(float)*ncells);
    float *sigma_m_pml = (float*)malloc(sizeof(float)*ncells);

    //initialize
    float sigma_max = (npml+1)/(150*PI*dx);
    float rho;
    for (int i = 0; i < ncells; i++) {
        rho = ((float)i+0.25)/ncells;
        sigma_e_pml[i] = sigma_max*sigma_factor*pow(rho,npml);

        rho = ((float)i+0.75)/ncells;
        sigma_m_pml[i] = (mu0/eps0)*sigma_max*sigma_factor*pow(rho,npml);

        /*
        cout<<"sigma_e_pml = "<<sigma_e_pml[i]<<endl;
        cout<<"sigma_m_pml "<<sigma_m_pml[i]<<endl;
        */
    }

    float *kex = (float*)malloc(sizeof(float)*ncells);
    float *kmx = (float*)malloc(sizeof(float)*ncells);
    float *aex = (float*)malloc(sizeof(float)*ncells);
    float *bex = (float*)malloc(sizeof(float)*ncells);
    float *amx = (float*)malloc(sizeof(float)*ncells);
    float *bmx = (float*)malloc(sizeof(float)*ncells);
    float *alpha_e = (float*)malloc(sizeof(float)*ncells);
    float *alpha_m = (float*)malloc(sizeof(float)*ncells);


    //Initialize kex and kmx (formerly k_e_init and k_m_init)
    //And alpha_e and alpha_m, and aex, bex, kex, amx, bmx, kmx
    for (int i = 0; i < ncells; i++) {
        rho = ((float)i+0.25)/ncells;
        kex[i]=pow(rho,npml)*(kmax-1)+1;
        alpha_e[i]=alpha_min+(alpha_max-alpha_min)*rho;

        rho = ((float)i+0.75)/ncells;
        kmx[i]=pow(rho,npml)*(kmax-1)+1;
        alpha_m[i]=(mu0/eps0)*(alpha_min+(alpha_max-alpha_min)*rho);

        bex[i]=exp(-1*(dt/eps0)*(sigma_e_pml[i]/kex[i]+alpha_e[i]));
        aex[i]=((bex[i]-1)*sigma_e_pml[i])/(dx*(sigma_e_pml[i]*kex[i]+alpha_e[i]*kex[i]*kex[i]));

        float argument = -1*(dt/mu0)*((sigma_m_pml[i]/kmx[i])+alpha_m[i]);
        bmx[i]=exp(argument);
        amx[i]=(bmx[i]-1)*sigma_m_pml[i]/(dx*(sigma_m_pml[i]*kmx[i]+alpha_m[i]*kmx[i]*kmx[i]));

        /*
        cout<<"kex["<<i<<"]= "<<kex[i]<<endl;
        cout<<"kmx["<<i<<"]= "<<kmx[i]<<endl;
        cout<<"aex["<<i<<"]= "<<aex[i]<<endl;
        cout<<"amx["<<i<<"]= "<<amx[i]<<endl;
        cout<<"bex["<<i<<"]= "<<bex[i]<<endl;
        cout<<"bmx["<<i<<"]= "<<bmx[i]<<endl;
        cout<<"alpha_e = "<<alpha_e[i]<<endl;
        cout<<"alpha_m = "<<alpha_m[i]<<endl;
        cout << endl;
        */
    }

    float *Psi_ezy = (float*)malloc(sizeof(float)*ny*20);
    float *Psi_ezx = (float*)malloc(sizeof(float)*nx*20);
    float *Psi_hyx = (float*)malloc(sizeof(float)*ny*20);
    float *Psi_hxy = (float*)malloc(sizeof(float)*nx*20);

    /*
    for (int i = 0; i < nx * 20; i++) {
        Psi_ezy[i] = 0.0;
        Psi_hxy[i] = 0.0;
    }

    for (int i = 0; i< ny * 20; i++) {
        Psi_ezx[i] = 0.0;
        Psi_hyx[i] = 0.0;
    }
    */
 
 
    
/**********************************************************************/

    float *D = (float*)malloc(sizeof(float)*numberofexcitationangles*numberofobservationangles*numberoffrequencies);//D = (float*)malloc(numberofobservationangles*sizeof(float));
    float *P = (float*)malloc(sizeof(float)*numberofexcitationangles*numberofobservationangles*numberoffrequencies);

    float *Hy = (float*)malloc(sizeof(float)*nx*ny);
    float *Hx = (float*)malloc(sizeof(float)*nx*ny);

    //This are output values from the device

    cuComplex *cjzxp, *cjzyp, *cjzxn, *cjzyn, *cmxyp, *cmyxp, *cmxyn, *cmyxn;
    cuComplex *hcjzyp = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzy); // hc**** are the host version of c****, Fourier transformed equivalent surface currents.
    cuComplex *hcjzyn = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzy);
    cuComplex *hcjzxp = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzx);
    cuComplex *hcjzxn = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzx);
    cuComplex *hcmxyn = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzy);
    cuComplex *hcmxyp = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzy);
    cuComplex *hcmyxp = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzx);
    cuComplex *hcmyxn = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzx);

    float *dev_freq, *dev_Phi;
    float *dev_Ceze,*dev_Cezhy, *dev_bex, *dev_aex, *dev_bmx, *dev_amx, *dev_kex, *dev_kmx;//dev_Cezj if using loop current source
    float *dev_Ez, *dev_Hy, *dev_Hx;

    float *dev_Psi_ezy, *dev_Psi_ezx, *dev_Psi_hyx, *dev_Psi_hxy;

    float *dev_Cezeic, *dev_Cezeip;
    float  *dev_eps_infinity,*dev_sigma_e_z;
    float *dev_Beta_p;
    float *dev_Cezjp,*dev_Jp;//dev_Jp is the polarization current term used in the Debye scattering (FD)2TD 

    cudaCheck( cudaMalloc(&dev_sigma_e_z,sizeof(float)*(nx+1)*(ny+1)) );
    cudaCheck( cudaMalloc(&dev_eps_infinity,sizeof(float)*(nx+1)*(ny+1)) );
    cudaCheck( cudaMalloc(&dev_Cezeic,sizeof(float)*(nx+1)*(ny+1)) );
    cudaCheck( cudaMalloc(&dev_Cezeip,sizeof(float)*(nx+1)*(ny+1)) );
    cudaCheck( cudaMalloc(&dev_Beta_p,sizeof(float)*(nx+1)*(ny+1)) );
    cudaCheck( cudaMalloc(&dev_Cezjp,sizeof(float)*(nx+1)*(ny+1)) );
    cudaCheck( cudaMemcpy(dev_eps_infinity,eps_infinity,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice) );
    cudaCheck( cudaMemcpy(dev_sigma_e_z,sigma_e_z,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice) );

    cudaCheck( cudaMemcpy(dev_Beta_p,Beta_p,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice) );
    
    scattered_parameter_init<<<grid,block>>>(dev_eps_infinity,dev_sigma_e_z,dev_Cezeic,dev_Cezeip,dev_Cezjp,dev_Beta_p);
    cudaCheckLastError("scattered_parameter_init kernel failed");

    //float *Cezeic = (float*)malloc((sizeof(float))*(nx+1)*(ny+1));
    // float *Cezeip = (float*)malloc((sizeof(float))*(nx+1)*(ny+1));
    //cudaMemcpy(Cezeic,dev_Cezeic,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyDeviceToHost);
    //cudaMemcpy(Cezeip,dev_Cezeip,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyDeviceToHost);

       cudaCheck(cudaMalloc(&dev_Phi,sizeof(float)));
    cudaCheck(cudaMalloc(&dev_kex,sizeof(float)*10));
    cudaCheck(cudaMalloc(&dev_kmx,sizeof(float)*10));
    cudaCheck(cudaMalloc(&dev_Ez,sizeof(float)*(nx+1)*(ny+1)));
    cudaCheck(cudaMalloc(&dev_Hy,sizeof(float)*nx*ny));
    cudaCheck(cudaMalloc(&dev_Jp,sizeof(float)*(1+nx)*(1+ny)));
    cudaCheck(cudaMalloc(&dev_freq ,sizeof(float)));
    cudaCheck(cudaMalloc(&dev_Hx,sizeof(float)*nx*ny));
    cudaCheck(cudaMalloc(&dev_Psi_ezy,sizeof(float)*20*(nx+1)));
    cudaCheck(cudaMalloc(&dev_Psi_ezx,sizeof(float)*20*(ny+1)));
    cudaCheck(cudaMalloc(&dev_Psi_hyx,sizeof(float)*20*(ny)));
    cudaCheck(cudaMalloc(&dev_Psi_hxy,sizeof(float)*20*(nx)));

    cudaCheck(cudaMalloc(&cjzxp,sizeof(cuComplex)*size_cjzx));
    cudaCheck(cudaMalloc(&cjzyp,sizeof(cuComplex)*size_cjzy));
    cudaCheck(cudaMalloc(&cjzxn,sizeof(cuComplex)*size_cjzx));
    cudaCheck(cudaMalloc(&cjzyn,sizeof(cuComplex)*size_cjzy));
    cudaCheck(cudaMalloc(&cmxyp,sizeof(cuComplex)*size_cjzy));
    cudaCheck(cudaMalloc(&cmxyn,sizeof(cuComplex)*size_cjzy));
    cudaCheck(cudaMalloc(&cmyxp,sizeof(cuComplex)*size_cjzx));
    cudaCheck(cudaMalloc(&cmyxn,sizeof(cuComplex)*size_cjzx));

    cudaCheck(cudaMemcpy(dev_freq,&freq,sizeof(float),cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&dev_bex,sizeof(float)*10));
    cudaCheck(cudaMalloc(&dev_bmx,sizeof(float)*10));
    cudaCheck(cudaMalloc(&dev_amx,sizeof(float)*10));
    cudaCheck(cudaMalloc(&dev_aex,sizeof(float)*10));
    cudaCheck(cudaMalloc(&dev_Cezhy,sizeof(float)*(nx+1)*(ny+1)));
    cudaCheck(cudaMalloc(&dev_Ceze,sizeof(float)*(nx+1)*(ny+1)));

    cudaCheck(cudaMemcpy(dev_amx,amx,sizeof(float)*10,cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_kex,kex,sizeof(float)*10,cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_kmx,kmx,sizeof(float)*10,cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_aex,aex,sizeof(float)*10,cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_bex,bex,sizeof(float)*10,cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_bmx,bmx,sizeof(float)*10,cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_amx,amx,sizeof(float)*10,cudaMemcpyHostToDevice));

    //cudaMalloc(&dev_Cezj,sizeof(float)*(nx+1)*(ny+1)); if using current source

    Field_reset<<<grid,block>>>(dev_Ez, dev_Hy, dev_Hx, dev_Psi_ezy, dev_Psi_ezx, dev_Psi_hyx, dev_Psi_hxy,cjzyn,cjzxp,cjzyp,cjzxn,cmxyn,cmyxp,cmxyp,cmyxn,dev_Jp);
    cudaCheckLastError("Field_reset kernel failed");
    //Field_reset is also good for making all these values zero.


    cudaCheck(cudaMemcpy(dev_Cezhy,Cezhy,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_Ceze,Ceze,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice));

    int *dev_i;
    cudaCheck( cudaMalloc(&dev_i,sizeof(int)) );
    float test_Ez;

    dim3 gridNF2FF((int)ceil(size_NF2FF_total/512.0));
    dim3 blockNF2FF(512);

    float test_Ez_2;
    float Phi;
//	ofstream measurement_data;
//	measurement_data.open("Measurement_Phase_tumor.txt");
   for(int Phi_index = 0; Phi_index < numberofexcitationangles; Phi_index++) {

			Phi = Phi_index*2*PI/numberofexcitationangles;
			cudaCheck( cudaMemcpy(dev_Phi,&Phi,sizeof(float),cudaMemcpyHostToDevice) );

			for (int i = 0; i < number_of_time_steps; i++) {
					cudaCheck( cudaMemcpy(dev_i,&i,sizeof(int),cudaMemcpyHostToDevice) );

					H_field_update<<<grid,block>>>(dev_Hy,dev_Hx,dev_Ez,dev_bmx,dev_Psi_hyx,dev_amx,dev_bmx,dev_amx,dev_Psi_hxy,dev_kmx);
					cudaCheckLastError("H_field_updated kernel failed");
					E_field_update<<<grid,block>>>(dev_i,dev_Ez,dev_Hy,dev_Hx,dev_Psi_ezx,dev_aex,dev_aex,dev_bex,dev_bex,dev_Psi_ezy,dev_kex,dev_Ceze,dev_Cezhy,dev_Cezhy,dev_Cezeip,dev_Cezeic,dev_Phi,dev_Jp,dev_Cezjp,dev_Beta_p);
					cudaCheckLastError("E_field_updated kernel failed");
					for(int freq_index = 0;freq_index<numberoffrequencies;freq_index++)
					{
					freq = startfrequency + deltafreq*freq_index;
					cudaCheck( cudaMemcpy(dev_freq,&freq,sizeof(float),cudaMemcpyHostToDevice) );
					calculate_JandM<<<gridNF2FF,blockNF2FF>>>(dev_freq, dev_i,dev_Ez,dev_Hy,dev_Hx, cjzxp+size_x_side*freq_index, cjzyp+size_y_side*freq_index, cjzxn+size_x_side*freq_index, cjzyn+size_y_side*freq_index, cmxyp+size_y_side*freq_index, cmyxp+size_x_side*freq_index, cmxyn+size_y_side*freq_index, cmyxn+size_x_side*freq_index);
					
					cudaCheckLastError("calculate_JandM kernel failed"  );
					}

				}

			cudaCheck( cudaMemcpy(hcjzyn,cjzyn,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost));
			cudaCheck( cudaMemcpy(hcjzxp,cjzxp,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost));
			cudaCheck( cudaMemcpy(hcjzyp,cjzyp,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost));
			cudaCheck( cudaMemcpy(hcjzxn,cjzxn,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost));
			cudaCheck( cudaMemcpy(hcmxyn,cmxyn,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost));
			cudaCheck( cudaMemcpy(hcmyxp,cmyxp,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost));
			cudaCheck( cudaMemcpy(hcmxyp,cmxyp,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost));
			cudaCheck( cudaMemcpy(hcmyxn,cmyxn,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost));

		

			for (int freq_index = 0; freq_index < numberoffrequencies; freq_index++)
			{
			freq = startfrequency + deltafreq*freq_index;
			
			N2FPostProcess(D + Phi_index*numberofobservationangles*numberoffrequencies+freq_index*numberofobservationangles, P + Phi_index*numberofobservationangles*numberoffrequencies+freq_index*numberofobservationangles, freq, hcjzxp+size_x_side*freq_index , hcjzyp+size_y_side*freq_index, hcjzxn+size_x_side*freq_index , hcjzyn+size_y_side*freq_index, hcmxyp+size_y_side*freq_index, hcmyxp+size_x_side*freq_index, hcmxyn+size_y_side*freq_index, hcmyxn+size_x_side*freq_index);
			
			}
			//D is a 3-dimensional array. The x axis is observation angles, z axis is Excitation angles, y axis is frequencies.
			//each N2FPostProcess  Fills D(:,freq_index,Phi_index) where ":" is, as per matlab notation, all the elements of the x row.

			

			Field_reset<<<grid,block>>>(dev_Ez, dev_Hy, dev_Hx, dev_Psi_ezy, dev_Psi_ezx, dev_Psi_hyx, dev_Psi_hxy,cjzyn,cjzxp,cjzyp,cjzxn,cmxyn,cmyxp,cmxyp,cmyxn,dev_Jp);
			cudaCheckLastError("Field_reset kernel failed");
		}
	

//	for(int index = 0;index < numberofobservationangles * numberoffrequencies * numberofexcitationangles ; index++)
//	{
  //          measurement_data<<D[index]<<" , ";
   // }		
   // for(int index = 0;index < numberofobservationangles * numberoffrequencies * numberofexcitationangles ; index++)
//	{
 //           measurement_data<<P[index]<<" , ";
   // }
   float measurement[] = { 0.00972085 , 0.00839843 , 0.00716477 , 0.00602837 , 0.00499639 , 0.00407456 , 0.00326719 , 0.00257694 , 0.00200486 , 0.00155032 , 0.001211 , 0.000983016 , 0.000860978 , 0.000838265 , 0.000907295 , 0.00105985 , 0.00128744 , 0.00158166 , 0.0019345 , 0.00233865 , 0.00278767 , 0.00327608 , 0.00379946 , 0.0043544 , 0.00493826 , 0.00554922 , 0.00618589 , 0.0068472 , 0.00753214 , 0.00823947 , 0.00896767 , 0.0097147 , 0.0104778 , 0.0112539 , 0.0120391 , 0.0128293 , 0.01362 , 0.014407 , 0.0151865 , 0.0159555 , 0.0167119 , 0.0174549 , 0.018185 , 0.018904 , 0.019615 , 0.0203218 , 0.021029 , 0.0217414 , 0.0224637 , 0.0231999 , 0.0239534 , 0.0247265 , 0.0255202 , 0.0263342 , 0.0271671 , 0.028016 , 0.028877 , 0.0297454 , 0.0306154 , 0.031481 , 0.0323355 , 0.0331721 , 0.0339839 , 0.0347637 , 0.0355045 , 0.0361991 , 0.0368407 , 0.0374222 , 0.0379367 , 0.0383776 , 0.0387384 , 0.0390131 , 0.039196 , 0.0392824 , 0.0392677 , 0.0391489 , 0.0389235 , 0.0385901 , 0.0381484 , 0.0375989 , 0.0369427 , 0.0361821 , 0.0353197 , 0.0343586 , 0.0333028 , 0.0321564 , 0.0309248 , 0.0296135 , 0.0282295 , 0.0267801 , 0.0252746 , 0.0237228 , 0.0221357 , 0.0205256 , 0.0189053 , 0.0172881 , 0.0156879 , 0.0141181 , 0.012592 , 0.0111223 , 0.261638 , 0.255231 , 0.2473 , 0.237988 , 0.227465 , 0.215925 , 0.20357 , 0.190602 , 0.177214 , 0.16358 , 0.149863 , 0.136222 , 0.122821 , 0.109841 , 0.0974773 , 0.0859362 , 0.0754125 , 0.0660691 , 0.0580148 , 0.0512888 , 0.0458543 , 0.0416051 , 0.0383781 , 0.035974 , 0.0341764 , 0.0327726 , 0.0315701 , 0.0304103 , 0.0291772 , 0.0278034 , 0.0262688 , 0.0245933 , 0.0228249 , 0.0210229 , 0.0192423 , 0.0175202 , 0.0158707 , 0.0142877 , 0.0127538 , 0.0112537 , 0.0097862 , 0.00837088 , 0.00704776 , 0.00587033 , 0.00489451 , 0.00416744 , 0.00371855 , 0.00355529 , 0.00366227 , 0.00400406 , 0.00452899 , 0.00517357 , 0.00586748 , 0.0065382 , 0.00711685 , 0.00754416 , 0.00777688 , 0.00779268 , 0.00759376 , 0.00720701 , 0.00668044 , 0.00607759 , 0.00546987 , 0.00492986 , 0.00452655 , 0.00432411 , 0.00438439 , 0.00477203 , 0.00556095 , 0.00683983 , 0.00871478 , 0.0113079 , 0.0147518 , 0.0191798 , 0.0247148 , 0.0314575 , 0.039476 , 0.0487979 , 0.0594055 , 0.071235 , 0.0841772 , 0.0980824 , 0.112765 , 0.128012 , 0.143584 , 0.159233 , 0.174698 , 0.189719 , 0.204041 , 0.217425 , 0.229649 , 0.240523 , 0.249884 , 0.257606 , 0.263597 , 0.267798 , 0.270175 , 0.270723 , 0.269455 , 0.266407 , 0.446609 , 0.438635 , 0.42443 , 0.405066 , 0.38168 , 0.355303 , 0.326747 , 0.29659 , 0.26527 , 0.233239 , 0.201115 , 0.169732 , 0.140076 , 0.113106 , 0.0895498 , 0.0697678 , 0.0537315 , 0.0411274 , 0.0315183 , 0.024485 , 0.019692 , 0.0168724 , 0.0157654 , 0.0160577 , 0.0173618 , 0.019243 , 0.0212767 , 0.0231105 , 0.0245041 , 0.0253345 , 0.0255688 , 0.0252239 , 0.0243377 , 0.0229675 , 0.0212116 , 0.019231 , 0.0172448 , 0.0154873 , 0.014144 , 0.0132983 , 0.0129179 , 0.0128862 , 0.0130604 , 0.013322 , 0.0135995 , 0.0138562 , 0.0140616 , 0.0141647 , 0.0140898 , 0.0137518 , 0.0130868 , 0.0120764 , 0.0107587 , 0.00921987 , 0.00757019 , 0.00592016 , 0.00436645 , 0.00299527 , 0.00189911 , 0.00119365 , 0.00102086 , 0.00152787 , 0.00282553 , 0.0049419 , 0.00779219 , 0.0111795 , 0.0148255 , 0.0184162 , 0.0216432 , 0.0242308 , 0.025949 , 0.0266206 , 0.0261338 , 0.0244599 , 0.0216733 , 0.0179658 , 0.0136502 , 0.0091522 , 0.0049948 , 0.00178127 , 0.000180541 , 0.000912612 , 0.00472428 , 0.0123427 , 0.0244001 , 0.0413388 , 0.0633204 , 0.090173 , 0.121394 , 0.156202 , 0.193623 , 0.23255 , 0.271788 , 0.310066 , 0.346044 , 0.378351 , 0.405658 , 0.426805 , 0.440916 , 0.44752 , 0.404564 , 0.39744 , 0.373997 , 0.337737 , 0.293123 , 0.244628 , 0.196029 , 0.150159 , 0.109058 , 0.074225 , 0.0466542 , 0.0266573 , 0.0136927 , 0.0064709 , 0.00334312 , 0.00274933 , 0.00348823 , 0.00474852 , 0.00600963 , 0.00695104 , 0.00742505 , 0.00745425 , 0.00719093 , 0.00683332 , 0.00654233 , 0.00640191 , 0.00642507 , 0.00657834 , 0.00680004 , 0.00701209 , 0.00714291 , 0.00717772 , 0.00722167 , 0.00752106 , 0.00838208 , 0.00999708 , 0.0122927 , 0.014931 , 0.0174743 , 0.0195843 , 0.0211151 , 0.022066 , 0.0224693 , 0.0223151 , 0.0215689 , 0.0202519 , 0.0185016 , 0.0165548 , 0.0146717 , 0.0130578 , 0.0118314 , 0.0110327 , 0.0106597 , 0.010702 , 0.0111683 , 0.0120925 , 0.0135123 , 0.015417 , 0.0176783 , 0.0200022 , 0.0219518 , 0.0230536 , 0.0229397 , 0.0214453 , 0.0186266 , 0.014732 , 0.0101922 , 0.0056518 , 0.00199815 , 0.000319927 , 0.00176459 , 0.00733048 , 0.0176658 , 0.0329299 , 0.0527261 , 0.076088 , 0.101502 , 0.12698 , 0.150198 , 0.168707 , 0.180224 , 0.182958 , 0.175943 , 0.159318 , 0.13446 , 0.103966 , 0.0714529 , 0.0411962 , 0.0176459 , 0.00483199 , 0.00574909 , 0.0218637 , 0.0528913 , 0.0968801 , 0.150496 , 0.209357 , 0.26836 , 0.322073 , 0.365278 , 0.393667 , 0.422784 , 0.406386 , 0.372498 , 0.325628 , 0.270897 , 0.213474 , 0.158816 , 0.112623 , 0.0794991 , 0.0608879 , 0.0543892 , 0.0556811 , 0.0613645 , 0.0697825 , 0.079341 , 0.0870114 , 0.0893075 , 0.084751 , 0.0749404 , 0.0630398 , 0.0514295 , 0.0408473 , 0.031291 , 0.0231384 , 0.0172474 , 0.0142128 , 0.013697 , 0.0144453 , 0.014932 , 0.0141512 , 0.0120336 , 0.00926462 , 0.00668832 , 0.00475632 , 0.0034331 , 0.00251304 , 0.00187932 , 0.00147493 , 0.00126595 , 0.00132322 , 0.00177826 , 0.00265189 , 0.00382128 , 0.00515312 , 0.00662513 , 0.00832133 , 0.0102996 , 0.0124577 , 0.01454 , 0.0162594 , 0.017392 , 0.0178002 , 0.0174447 , 0.0164283 , 0.01502 , 0.0135794 , 0.0123769 , 0.011411 , 0.0103731 , 0.00888401 , 0.00692846 , 0.00510977 , 0.00436348 , 0.00530634 , 0.00785434 , 0.011501 , 0.015969 , 0.0215096 , 0.0283961 , 0.0360465 , 0.0427579 , 0.0464597 , 0.0458499 , 0.0409957 , 0.0331072 , 0.0238822 , 0.014972 , 0.00781877 , 0.00375099 , 0.00405606 , 0.00980209 , 0.0214156 , 0.0382885 , 0.0587525 , 0.0804924 , 0.10114 , 0.118774 , 0.132302 , 0.141814 , 0.148717 , 0.155427 , 0.16471 , 0.179141 , 0.200859 , 0.231252 , 0.27013 , 0.31471 , 0.35933 , 0.39653 , 0.419203 , 1.13394 , 1.07799 , 0.973694 , 0.832076 , 0.668346 , 0.502542 , 0.355467 , 0.240634 , 0.15949 , 0.105393 , 0.071894 , 0.055831 , 0.0533194 , 0.0567706 , 0.0585716 , 0.056189 , 0.0516248 , 0.0467873 , 0.0413026 , 0.0346474 , 0.0283193 , 0.0246991 , 0.0246387 , 0.0270957 , 0.0306586 , 0.0345726 , 0.0382571 , 0.04059 , 0.0404263 , 0.0378638 , 0.0343662 , 0.0312455 , 0.0285113 , 0.0256278 , 0.0227419 , 0.0204159 , 0.0184563 , 0.0159264 , 0.0125547 , 0.009225 , 0.0068175 , 0.00539237 , 0.00451621 , 0.00386447 , 0.00347152 , 0.00355106 , 0.00424854 , 0.00551929 , 0.00709834 , 0.00864511 , 0.00993238 , 0.0108371 , 0.0112264 , 0.0110039 , 0.0103161 , 0.00962762 , 0.00946882 , 0.0100689 , 0.0112872 , 0.012936 , 0.015059 , 0.0177214 , 0.020566 , 0.0227422 , 0.0233508 , 0.0219852 , 0.0188388 , 0.0144308 , 0.0095339 , 0.00540002 , 0.0037401 , 0.00618722 , 0.0136448 , 0.0259949 , 0.0421073 , 0.0598612 , 0.0762882 , 0.0881787 , 0.0931109 , 0.0902997 , 0.0806667 , 0.0662581 , 0.049793 , 0.0345183 , 0.0235598 , 0.0185994 , 0.0193658 , 0.0251798 , 0.0372408 , 0.0592433 , 0.09626 , 0.153775 , 0.237174 , 0.350105 , 0.491406 , 0.65253 , 0.817425 , 0.965327 , 1.07578 , 1.13397 , 1.38045 , 1.31448 , 1.17341 , 0.973947 , 0.73953 , 0.504143 , 0.304962 , 0.163256 , 0.0761031 , 0.0297429 , 0.0129532 , 0.0145931 , 0.0209783 , 0.0241319 , 0.0251424 , 0.0257499 , 0.0245901 , 0.0221175 , 0.0211754 , 0.0220715 , 0.0221135 , 0.019957 , 0.0169357 , 0.0145629 , 0.0130498 , 0.0120129 , 0.0114754 , 0.0120707 , 0.0144463 , 0.0183589 , 0.0224064 , 0.0250635 , 0.0262135 , 0.0270416 , 0.027949 , 0.0280259 , 0.02726 , 0.027128 , 0.0280996 , 0.0284761 , 0.0271303 , 0.0252127 , 0.0238581 , 0.0224488 , 0.0199944 , 0.0169763 , 0.0148137 , 0.0142602 , 0.0149828 , 0.0160237 , 0.0164996 , 0.0161898 , 0.0155863 , 0.015464 , 0.0162727 , 0.017707 , 0.0188749 , 0.0191367 , 0.0187369 , 0.0181398 , 0.0170073 , 0.0147158 , 0.0116671 , 0.00917839 , 0.00798903 , 0.00792019 , 0.00925378 , 0.0130559 , 0.0195893 , 0.0272991 , 0.0338274 , 0.0373955 , 0.0370766 , 0.0327899 , 0.0256522 , 0.0179271 , 0.0123101 , 0.0112705 , 0.0167348 , 0.0297019 , 0.0498508 , 0.0755811 , 0.103937 , 0.129891 , 0.146734 , 0.148991 , 0.135024 , 0.106514 , 0.0679286 , 0.0288431 , 0.00453248 , 0.0119407 , 0.0663503 , 0.180239 , 0.357906 , 0.586797 , 0.836684 , 1.06988 , 1.2525 , 1.36012 , 1.49551 , 1.40156 , 1.169 , 0.857267 , 0.540647 , 0.283783 , 0.115646 , 0.0313153 , 0.0151852 , 0.043942 , 0.0817782 , 0.104169 , 0.111595 , 0.106066 , 0.0856349 , 0.0631255 , 0.0532492 , 0.0519845 , 0.0516625 , 0.053906 , 0.057205 , 0.0552244 , 0.049214 , 0.0450782 , 0.0423325 , 0.0364547 , 0.0284905 , 0.0243361 , 0.0251621 , 0.0257917 , 0.0232225 , 0.0202209 , 0.0193207 , 0.0189013 , 0.0174773 , 0.0166843 , 0.018016 , 0.0194815 , 0.0193379 , 0.0197244 , 0.0222712 , 0.0243829 , 0.0242384 , 0.024181 , 0.0255577 , 0.0261999 , 0.0244857 , 0.0220606 , 0.0211052 , 0.0218472 , 0.0229918 , 0.0234233 , 0.0235392 , 0.0247241 , 0.0273103 , 0.02949 , 0.0291112 , 0.0268451 , 0.0252802 , 0.0240777 , 0.0209146 , 0.0173862 , 0.0174616 , 0.0200025 , 0.0210661 , 0.0217574 , 0.0244113 , 0.0255105 , 0.021376 , 0.0153671 , 0.0119491 , 0.0107149 , 0.0115105 , 0.0185696 , 0.034425 , 0.054228 , 0.0697329 , 0.0761309 , 0.0732877 , 0.062832 , 0.0469428 , 0.0299995 , 0.0189548 , 0.020802 , 0.0401505 , 0.077434 , 0.126329 , 0.173178 , 0.200991 , 0.196269 , 0.156151 , 0.0930777 , 0.0356535 , 0.0253352 , 0.103157 , 0.288124 , 0.565494 , 0.891569 , 1.20207 , 1.42308 , 1.61256 , 1.49415 , 1.19138 , 0.806085 , 0.461849 , 0.224542 , 0.093107 , 0.0531827 , 0.0774068 , 0.110023 , 0.120062 , 0.11473 , 0.0932162 , 0.0610043 , 0.0408341 , 0.0339502 , 0.0311319 , 0.0354668 , 0.0415848 , 0.0395556 , 0.0352144 , 0.0355841 , 0.0374507 , 0.0388802 , 0.0406947 , 0.0407661 , 0.0388181 , 0.0383473 , 0.0394575 , 0.0396243 , 0.0395575 , 0.0391429 , 0.0362479 , 0.0333881 , 0.0329209 , 0.0309515 , 0.0272997 , 0.0264739 , 0.0268087 , 0.0243017 , 0.0218048 , 0.0222314 , 0.0228361 , 0.0214619 , 0.0203272 , 0.0206698 , 0.0210336 , 0.0206561 , 0.0207894 , 0.022562 , 0.0251517 , 0.0265775 , 0.0263843 , 0.0262723 , 0.0278907 , 0.0307833 , 0.0334602 , 0.0358437 , 0.038111 , 0.0390064 , 0.0385026 , 0.0379154 , 0.0358768 , 0.0307067 , 0.0247521 , 0.0199185 , 0.0155067 , 0.0133218 , 0.0160523 , 0.0225141 , 0.0301452 , 0.0366664 , 0.0387156 , 0.0349676 , 0.0281823 , 0.0217472 , 0.0185695 , 0.0228979 , 0.0374085 , 0.0595728 , 0.0840627 , 0.102956 , 0.104475 , 0.0843606 , 0.0544872 , 0.0310958 , 0.0277184 , 0.0545507 , 0.105034 , 0.155511 , 0.182423 , 0.170703 , 0.126758 , 0.0910723 , 0.121683 , 0.264366 , 0.53056 , 0.884287 , 1.24597 , 1.51682 , 2.25792 , 2.07928 , 1.62871 , 1.05898 , 0.567864 , 0.250713 , 0.0980921 , 0.0733079 , 0.101676 , 0.113807 , 0.102758 , 0.0720192 , 0.0382797 , 0.0284777 , 0.0331327 , 0.0436363 , 0.0551271 , 0.0531371 , 0.0448846 , 0.0379254 , 0.0292598 , 0.0260227 , 0.0307599 , 0.0356047 , 0.0372393 , 0.0367187 , 0.0330498 , 0.0265586 , 0.022678 , 0.0248917 , 0.0287412 , 0.0306922 , 0.0335239 , 0.0360117 , 0.0344917 , 0.0347116 , 0.0372255 , 0.0352284 , 0.0337768 , 0.0364122 , 0.0362618 , 0.0337921 , 0.0338134 , 0.0339148 , 0.0316858 , 0.0289836 , 0.0266213 , 0.0244373 , 0.0231738 , 0.0230268 , 0.0224334 , 0.020654 , 0.0192908 , 0.0203249 , 0.0233849 , 0.0273671 , 0.0325083 , 0.0376294 , 0.039753 , 0.0397883 , 0.0394188 , 0.0359855 , 0.0305299 , 0.0281213 , 0.027427 , 0.0266914 , 0.0290363 , 0.0319684 , 0.0303683 , 0.0259699 , 0.0212624 , 0.0165142 , 0.015533 , 0.0242925 , 0.0428821 , 0.0622257 , 0.0705937 , 0.0634982 , 0.0470994 , 0.0331974 , 0.0310116 , 0.04269 , 0.0662558 , 0.0944627 , 0.111714 , 0.108112 , 0.0874781 , 0.0624735 , 0.0547807 , 0.0744858 , 0.106452 , 0.127162 , 0.119097 , 0.102419 , 0.144773 , 0.321982 , 0.684079 , 1.20266 , 1.74061 , 2.12971 , 0.0385277 , 0.0388277 , 0.0391423 , 0.039468 , 0.0398011 , 0.0401376 , 0.040474 , 0.0408066 , 0.0411331 , 0.0414508 , 0.0417584 , 0.0420546 , 0.042339 , 0.0426108 , 0.0428699 , 0.0431155 , 0.0433466 , 0.043562 , 0.04376 , 0.0439387 , 0.0440958 , 0.0442294 , 0.0443372 , 0.0444179 , 0.0444702 , 0.0444938 , 0.0444887 , 0.0444556 , 0.0443958 , 0.0443109 , 0.0442024 , 0.0440722 , 0.0439218 , 0.0437528 , 0.0435659 , 0.0433618 , 0.043141 , 0.0429035 , 0.0426492 , 0.0423782 , 0.0420912 , 0.041789 , 0.041473 , 0.0411459 , 0.0408101 , 0.0404697 , 0.0401285 , 0.0397909 , 0.0394614 , 0.039144 , 0.0388424 , 0.0385593 , 0.0382971 , 0.0380566 , 0.037838 , 0.0376405 , 0.0374627 , 0.0373025 , 0.0371576 , 0.0370255 , 0.0369042 , 0.0367917 , 0.036687 , 0.0365891 , 0.0364976 , 0.0364129 , 0.0363352 , 0.0362651 , 0.036203 , 0.0361493 , 0.0361039 , 0.036067 , 0.036038 , 0.0360164 , 0.0360017 , 0.0359933 , 0.0359906 , 0.0359931 , 0.0360009 , 0.0360139 , 0.0360325 , 0.0360566 , 0.0360871 , 0.036124 , 0.036168 , 0.0362195 , 0.036279 , 0.0363467 , 0.0364238 , 0.0365108 , 0.0366089 , 0.0367197 , 0.0368444 , 0.0369852 , 0.0371437 , 0.0373216 , 0.0375202 , 0.0377403 , 0.0379821 , 0.038245 , 0.0314798 , 0.0364847 , 0.0414033 , 0.0461339 , 0.0505899 , 0.0547052 , 0.0584378 , 0.0617739 , 0.0647281 , 0.0673422 , 0.0696809 , 0.0718235 , 0.0738544 , 0.0758515 , 0.0778766 , 0.0799677 , 0.0821328 , 0.0843527 , 0.0865829 , 0.0887614 , 0.0908157 , 0.0926709 , 0.0942563 , 0.0955083 , 0.0963752 , 0.0968177 , 0.0968103 , 0.0963428 , 0.0954232 , 0.094078 , 0.0923541 , 0.0903172 , 0.0880481 , 0.0856364 , 0.0831693 , 0.0807216 , 0.0783456 , 0.0760629 , 0.0738612 , 0.0716959 , 0.0694953 , 0.0671712 , 0.0646298 , 0.0617851 , 0.0585706 , 0.0549481 , 0.0509137 , 0.0464991 , 0.0417694 , 0.0368171 , 0.0317558 , 0.026711 , 0.0218133 , 0.0171909 , 0.0129645 , 0.00924306 , 0.00612098 , 0.00367593 , 0.00196717 , 0.00103358 , 0.000891681 , 0.00153346 , 0.00292461 , 0.0050036 , 0.0076824 , 0.0108488 , 0.0143704 , 0.0181004 , 0.0218833 , 0.0255611 , 0.0289802 , 0.0319965 , 0.0344811 , 0.0363268 , 0.0374517 , 0.0378054 , 0.037371 , 0.0361681 , 0.0342513 , 0.0317079 , 0.0286521 , 0.0252181 , 0.0215527 , 0.0178081 , 0.0141351 , 0.0106788 , 0.00757471 , 0.00494541 , 0.00289809 , 0.00152096 , 0.000880114 , 0.0010164 , 0.0019435 , 0.00364741 , 0.00608774 , 0.0092006 , 0.0129023 , 0.017094 , 0.0216658 , 0.0265008 , 0.0177829 , 0.00954661 , 0.00369016 , 0.000784135 , 0.0011748 , 0.00497166 , 0.0120707 , 0.0222017 , 0.0349775 , 0.049926 , 0.0664986 , 0.0840697 , 0.101955 , 0.119468 , 0.136006 , 0.151146 , 0.164687 , 0.176646 , 0.187171 , 0.196453 , 0.204628 , 0.211742 , 0.217741 , 0.222493 , 0.225822 , 0.227533 , 0.227445 , 0.225421 , 0.221411 , 0.21548 , 0.207809 , 0.198655 , 0.188283 , 0.176876 , 0.164485 , 0.151037 , 0.136404 , 0.120533 , 0.103565 , 0.0858986 , 0.0681786 , 0.051194 , 0.0357496 , 0.0225557 , 0.012166 , 0.00496097 , 0.00115131 , 0.000776885 , 0.00369221 , 0.00954669 , 0.0177786 , 0.0276387 , 0.0382506 , 0.0486959 , 0.0581062 , 0.0657383 , 0.0710266 , 0.0736089 , 0.0733417 , 0.0703087 , 0.0648222 , 0.0574034 , 0.0487299 , 0.0395497 , 0.0305784 , 0.0224036 , 0.015426 , 0.00984473 , 0.00568446 , 0.00284081 , 0.0011247 , 0.000294674 , 8.01405e-05 , 0.000205253 , 0.000420415 , 0.00054045 , 0.000480968 , 0.000282192 , 0.000110736 , 0.000235922 , 0.000983101 , 0.00267426 , 0.00556958 , 0.00982424 , 0.0154658 , 0.0223908 , 0.0303678 , 0.0390419 , 0.0479327 , 0.0564456 , 0.0639032 , 0.0696123 , 0.0729619 , 0.0735232 , 0.0711284 , 0.0659011 , 0.0582406 , 0.0487718 , 0.0382812 , 0.0276492 , 0.0770128 , 0.0914558 , 0.101504 , 0.106188 , 0.105251 , 0.0989659 , 0.0879315 , 0.0730159 , 0.0554847 , 0.0371581 , 0.0203553 , 0.0075602 , 0.00102066 , 0.00259214 , 0.0138811 , 0.0363693 , 0.0711431 , 0.118224 , 0.17594 , 0.2408 , 0.308006 , 0.372264 , 0.428521 , 0.472432 , 0.500598 , 0.510754 , 0.501945 , 0.474671 , 0.430923 , 0.374117 , 0.308881 , 0.240686 , 0.175178 , 0.117299 , 0.070432 , 0.0360144 , 0.0138041 , 0.0026165 , 0.00102358 , 0.00758156 , 0.0205839 , 0.0377706 , 0.0564447 , 0.0740106 , 0.0885197 , 0.0988328 , 0.104395 , 0.104928 , 0.100327 , 0.0908042 , 0.077112 , 0.0606446 , 0.0433333 , 0.0273682 , 0.0148347 , 0.00733272 , 0.00564969 , 0.00958196 , 0.0179979 , 0.0291435 , 0.041047 , 0.0518284 , 0.0598519 , 0.0638496 , 0.0631672 , 0.0580554 , 0.0497454 , 0.0401151 , 0.0310621 , 0.0239125 , 0.0191593 , 0.0165717 , 0.015515 , 0.0152692 , 0.0152405 , 0.0150757 , 0.0147191 , 0.0144176 , 0.0146535 , 0.0160069 , 0.0189943 , 0.0239287 , 0.0308095 , 0.0392128 , 0.0481957 , 0.0563045 , 0.0618135 , 0.0632139 , 0.0597696 , 0.0518427 , 0.0408095 , 0.0286524 , 0.0174891 , 0.00922793 , 0.00537135 , 0.00687492 , 0.0140034 , 0.0262155 , 0.0421619 , 0.0598569 , 0.023687 , 0.00994099 , 0.00249563 , 0.00191465 , 0.00777438 , 0.0188709 , 0.0331364 , 0.0475907 , 0.0588685 , 0.064365 , 0.0632483 , 0.0565575 , 0.046617 , 0.0367493 , 0.0315232 , 0.0365752 , 0.0573378 , 0.0974713 , 0.158158 , 0.238206 , 0.333949 , 0.438576 , 0.541745 , 0.63056 , 0.691969 , 0.715709 , 0.696895 , 0.637711 , 0.547597 , 0.441145 , 0.333861 , 0.237696 , 0.158938 , 0.0993644 , 0.0588826 , 0.0368759 , 0.0312723 , 0.0373247 , 0.0485046 , 0.0589336 , 0.0649255 , 0.0649098 , 0.0588106 , 0.047672 , 0.0335391 , 0.0192429 , 0.00779916 , 0.00166969 , 0.00234265 , 0.0102125 , 0.0244564 , 0.0429067 , 0.062223 , 0.0785837 , 0.0887619 , 0.091127 , 0.0860361 , 0.0754344 , 0.0620073 , 0.0484133 , 0.0368269 , 0.0286922 , 0.024569 , 0.0241375 , 0.0264327 , 0.0302025 , 0.0342219 , 0.0375762 , 0.0399532 , 0.0417429 , 0.0437011 , 0.0463423 , 0.0495451 , 0.0526566 , 0.0549169 , 0.055826 , 0.0552474 , 0.0533252 , 0.0503945 , 0.046956 , 0.0436104 , 0.0408247 , 0.038619 , 0.0364865 , 0.0337597 , 0.0302256 , 0.0265215 , 0.0240032 , 0.0241978 , 0.0282069 , 0.036339 , 0.0480135 , 0.0618088 , 0.0755301 , 0.0863747 , 0.0914604 , 0.0887675 , 0.0780753 , 0.0612858 , 0.0418622 , 0.0605189 , 0.0714893 , 0.0776678 , 0.0777283 , 0.0711186 , 0.0585162 , 0.0424384 , 0.0267886 , 0.0151731 , 0.0095126 , 0.0100341 , 0.0157359 , 0.0245264 , 0.0341093 , 0.0439504 , 0.0565863 , 0.077483 , 0.114296 , 0.175358 , 0.266304 , 0.385824 , 0.523839 , 0.663522 , 0.785056 , 0.869019 , 0.899938 , 0.870756 , 0.786406 , 0.663258 , 0.523534 , 0.387787 , 0.270185 , 0.178343 , 0.114769 , 0.0768648 , 0.0570245 , 0.0455295 , 0.0353225 , 0.0247864 , 0.0159136 , 0.0106511 , 0.00991374 , 0.0146249 , 0.0255869 , 0.0416433 , 0.0587167 , 0.0718719 , 0.0782886 , 0.0778018 , 0.0714347 , 0.0604872 , 0.0469184 , 0.0336787 , 0.0239411 , 0.0198695 , 0.0220897 , 0.0298645 , 0.0411835 , 0.0528168 , 0.0612961 , 0.0647314 , 0.0636217 , 0.0597081 , 0.0544227 , 0.0485441 , 0.0427657 , 0.038042 , 0.0352704 , 0.0347905 , 0.0362403 , 0.0388195 , 0.0417552 , 0.0446233 , 0.0472565 , 0.0494305 , 0.0507043 , 0.0505724 , 0.0487906 , 0.0456309 , 0.0418554 , 0.0383833 , 0.0359163 , 0.0348532 , 0.0354349 , 0.0377696 , 0.0417533 , 0.0471017 , 0.0533246 , 0.0594381 , 0.0637739 , 0.0644364 , 0.0602904 , 0.0517028 , 0.0405452 , 0.0296248 , 0.0218956 , 0.0196462 , 0.0238193 , 0.0336678 , 0.0469278 , 0.0180486 , 0.00958071 , 0.00683165 , 0.0110559 , 0.0220562 , 0.0376693 , 0.0544577 , 0.0694081 , 0.0804977 , 0.0854975 , 0.0819731 , 0.0697604 , 0.0520435 , 0.0333868 , 0.0183851 , 0.0130053 , 0.0257054 , 0.0657721 , 0.14095 , 0.255925 , 0.409735 , 0.591453 , 0.778389 , 0.941102 , 1.05219 , 1.09236 , 1.0534 , 0.941367 , 0.777897 , 0.593432 , 0.415035 , 0.260463 , 0.141245 , 0.0639066 , 0.0253017 , 0.0136437 , 0.0183239 , 0.033688 , 0.0540204 , 0.0718401 , 0.0823364 , 0.0852014 , 0.0811041 , 0.0703296 , 0.0543886 , 0.0367765 , 0.021559 , 0.0113372 , 0.00715552 , 0.00923631 , 0.0170531 , 0.0291503 , 0.0430304 , 0.0552289 , 0.0620382 , 0.0613755 , 0.0546317 , 0.0459043 , 0.0389301 , 0.035315 , 0.0351876 , 0.0380474 , 0.0427107 , 0.0478394 , 0.052972 , 0.0579178 , 0.0616123 , 0.0628656 , 0.0620426 , 0.0610017 , 0.0612438 , 0.0628415 , 0.0650371 , 0.0671608 , 0.068866 , 0.0698163 , 0.0695299 , 0.0677959 , 0.0652007 , 0.0629449 , 0.0619455 , 0.0620625 , 0.0622102 , 0.0613722 , 0.0594408 , 0.0567379 , 0.0529944 , 0.0477072 , 0.0416111 , 0.0367005 , 0.0344509 , 0.0351435 , 0.0389821 , 0.0460534 , 0.0545527 , 0.0605581 , 0.0606472 , 0.0542211 , 0.0431237 , 0.0301131 , 0.0538413 , 0.0635858 , 0.0645978 , 0.0564933 , 0.0419338 , 0.0253747 , 0.0112695 , 0.00390251 , 0.00821699 , 0.0267072 , 0.0551344 , 0.0855254 , 0.110004 , 0.118261 , 0.101553 , 0.0650407 , 0.0282199 , 0.0157857 , 0.0552018 , 0.170876 , 0.36817 , 0.628721 , 0.920935 , 1.20183 , 1.41129 , 1.48686 , 1.40269 , 1.19217 , 0.920514 , 0.638502 , 0.376852 , 0.171425 , 0.0540276 , 0.0176989 , 0.0293127 , 0.0652994 , 0.105423 , 0.12285 , 0.110836 , 0.0858597 , 0.0577054 , 0.0284511 , 0.00781047 , 0.00365505 , 0.0114183 , 0.0250952 , 0.0424133 , 0.0584517 , 0.065714 , 0.0617706 , 0.0506266 , 0.0375948 , 0.026682 , 0.0207132 , 0.0208418 , 0.0264207 , 0.0362433 , 0.0486517 , 0.0599422 , 0.0655328 , 0.0647629 , 0.061191 , 0.0573689 , 0.0532408 , 0.0493981 , 0.0479907 , 0.0504151 , 0.0559631 , 0.0623396 , 0.0678759 , 0.0731985 , 0.0797784 , 0.0871375 , 0.0928896 , 0.0952983 , 0.094758 , 0.0928607 , 0.090487 , 0.0870905 , 0.0818327 , 0.0750416 , 0.0680175 , 0.061555 , 0.0556121 , 0.0505536 , 0.0476423 , 0.0480104 , 0.0516021 , 0.0567717 , 0.0612567 , 0.0638914 , 0.0638146 , 0.0589154 , 0.0485228 , 0.0362203 , 0.0264176 , 0.0207124 , 0.0200069 , 0.0262999 , 0.0392613 , 0.0211999 , 0.0141326 , 0.0152918 , 0.0255605 , 0.0421959 , 0.0610069 , 0.0758422 , 0.0762632 , 0.0576874 , 0.031144 , 0.011534 , 0.00986893 , 0.0337375 , 0.0740297 , 0.108048 , 0.119294 , 0.0978309 , 0.0521644 , 0.0278864 , 0.0849752 , 0.267275 , 0.588646 , 1.00944 , 1.43099 , 1.73574 , 1.83927 , 1.72036 , 1.4199 , 1.01514 , 0.599518 , 0.269864 , 0.0845312 , 0.0299559 , 0.0526097 , 0.0991099 , 0.124038 , 0.110295 , 0.0740705 , 0.0355555 , 0.0106563 , 0.0110907 , 0.0312346 , 0.0563304 , 0.0741283 , 0.076158 , 0.0625009 , 0.0425563 , 0.0252184 , 0.0151852 , 0.014081 , 0.0210088 , 0.0330562 , 0.0459699 , 0.0551088 , 0.0574066 , 0.0530628 , 0.0452822 , 0.0383755 , 0.0353436 , 0.0382065 , 0.0476899 , 0.0596343 , 0.0678742 , 0.0728251 , 0.0769184 , 0.0769505 , 0.0718215 , 0.0675984 , 0.0681318 , 0.0711313 , 0.074881 , 0.0817862 , 0.0934113 , 0.105899 , 0.113005 , 0.113121 , 0.109282 , 0.10332 , 0.0944868 , 0.0838015 , 0.0752618 , 0.0709703 , 0.0690729 , 0.0682383 , 0.0699191 , 0.0737617 , 0.07573 , 0.0729492 , 0.0662708 , 0.057142 , 0.0468197 , 0.0382124 , 0.0346533 , 0.0372482 , 0.0443485 , 0.0524458 , 0.0572411 , 0.0552447 , 0.0461267 , 0.033229 , 0.0613993 , 0.0640599 , 0.05242 , 0.0333649 , 0.0180925 , 0.0131123 , 0.0223166 , 0.0437449 , 0.0659797 , 0.0795064 , 0.076705 , 0.0546429 , 0.0311214 , 0.0303343 , 0.0601249 , 0.102386 , 0.120939 , 0.0980873 , 0.0554229 , 0.0596824 , 0.201403 , 0.535315 , 1.04839 , 1.64065 , 2.12028 , 2.29405 , 2.0988 , 1.63063 , 1.05952 , 0.544109 , 0.200566 , 0.0610857 , 0.0566615 , 0.0982783 , 0.124431 , 0.103117 , 0.0596175 , 0.0322744 , 0.0313428 , 0.0557658 , 0.0785788 , 0.0781168 , 0.0649608 , 0.0438442 , 0.021598 , 0.0135148 , 0.0184659 , 0.0316623 , 0.0511703 , 0.0654776 , 0.0623476 , 0.0463292 , 0.0315746 , 0.0264031 , 0.0302667 , 0.0391748 , 0.0499755 , 0.0596502 , 0.0650725 , 0.0648496 , 0.0607533 , 0.0573206 , 0.0589626 , 0.0653127 , 0.0724741 , 0.0807047 , 0.0900065 , 0.0944816 , 0.0937934 , 0.0941034 , 0.0958406 , 0.0962405 , 0.0975345 , 0.102599 , 0.109013 , 0.111989 , 0.109333 , 0.102831 , 0.0969387 , 0.094992 , 0.096257 , 0.097045 , 0.0955062 , 0.0923567 , 0.0875734 , 0.080553 , 0.07139 , 0.0623628 , 0.0577539 , 0.0576987 , 0.059277 , 0.06265 , 0.0646354 , 0.05958 , 0.0495481 , 0.0388589 , 0.0294978 , 0.0264498 , 0.0338415 , 0.0483221 , 0.0237439 , 0.0229905 , 0.0222569 , 0.02154 , 0.020836 , 0.02014 , 0.019447 , 0.018752 , 0.0180504 , 0.0173383 , 0.016613 , 0.0158732 , 0.015119 , 0.0143516 , 0.0135738 , 0.012789 , 0.0120016 , 0.0112162 , 0.0104373 , 0.00966925 , 0.00891598 , 0.0081808 , 0.00746643 , 0.00677515 , 0.00610872 , 0.00546876 , 0.00485678 , 0.00427454 , 0.00372403 , 0.00320784 , 0.00272915 , 0.00229199 , 0.00190111 , 0.00156212 , 0.00128142 , 0.00106602 , 0.00092347 , 0.000861558 , 0.000888087 , 0.00101055 , 0.00123583 , 0.00156986 , 0.00201738 , 0.00258165 , 0.00326431 , 0.0040653 , 0.00498275 , 0.00601309 , 0.00715104 , 0.00838977 , 0.00972098 , 0.011135 , 0.012621 , 0.014167 , 0.0157601 , 0.0173865 , 0.0190321 , 0.0206822 , 0.0223223 , 0.0239382 , 0.0255164 , 0.0270442 , 0.0285105 , 0.0299053 , 0.0312202 , 0.0324486 , 0.0335847 , 0.0346246 , 0.035565 , 0.0364037 , 0.0371388 , 0.0377691 , 0.0382938 , 0.0387122 , 0.0390242 , 0.0392302 , 0.0393312 , 0.039329 , 0.039226 , 0.0390262 , 0.0387338 , 0.0383545 , 0.0378944 , 0.0373604 , 0.0367597 , 0.0360999 , 0.0353885 , 0.034633 , 0.0338406 , 0.0330185 , 0.0321734 , 0.031312 , 0.0304405 , 0.0295654 , 0.028692 , 0.027826 , 0.0269721 , 0.0261344 , 0.025316 , 0.0245189 , 0.00450414 , 0.00400088 , 0.00366038 , 0.00353229 , 0.00365255 , 0.0040405 , 0.00469764 , 0.00560836 , 0.00674288 , 0.00806213 , 0.0095233 , 0.0110856 , 0.0127139 , 0.0143803 , 0.0160637 , 0.0177465 , 0.0194124 , 0.0210438 , 0.0226224 , 0.0241306 , 0.0255569 , 0.026901 , 0.0281799 , 0.0294331 , 0.0307272 , 0.0321579 , 0.0338504 , 0.035958 , 0.0386558 , 0.0421325 , 0.046574 , 0.0521456 , 0.0589696 , 0.0671048 , 0.0765335 , 0.087157 , 0.0988078 , 0.111271 , 0.124312 , 0.137708 , 0.15126 , 0.164807 , 0.178216 , 0.191363 , 0.204121 , 0.216343 , 0.227852 , 0.238449 , 0.247922 , 0.256062 , 0.262682 , 0.267627 , 0.270787 , 0.272097 , 0.271535 , 0.269116 , 0.264883 , 0.258901 , 0.251246 , 0.24201 , 0.231297 , 0.219233 , 0.205975 , 0.191709 , 0.176664 , 0.161096 , 0.145287 , 0.129527 , 0.114096 , 0.099249 , 0.0852095 , 0.072156 , 0.0602228 , 0.0495001 , 0.0400379 , 0.0318497 , 0.0249157 , 0.019187 , 0.0145885 , 0.0110232 , 0.00837632 , 0.0065222 , 0.00533086 , 0.00467528 , 0.00443728 , 0.00451163 , 0.00480747 , 0.00524716 , 0.00576378 , 0.00629788 , 0.00679544 , 0.00720734 , 0.00749111 , 0.00761406 , 0.00755727 , 0.00731858 , 0.00691408 , 0.00637736 , 0.00575667 , 0.00511087 , 0.0134553 , 0.0140988 , 0.0143442 , 0.0142472 , 0.0139074 , 0.0134492 , 0.0129991 , 0.0126635 , 0.0125154 , 0.0125915 , 0.0129039 , 0.0134571 , 0.0142646 , 0.0153512 , 0.0167375 , 0.0184102 , 0.0202902 , 0.0222172 , 0.0239618 , 0.0252667 , 0.0259047 , 0.0257353 , 0.0247427 , 0.0230483 , 0.0208986 , 0.0186381 , 0.016672 , 0.0154273 , 0.0153147 , 0.0166975 , 0.0198752 , 0.0250931 , 0.0325792 , 0.0425925 , 0.0554503 , 0.0715029 , 0.0910422 , 0.114175 , 0.140714 , 0.170131 , 0.201612 , 0.234177 , 0.266842 , 0.298741 , 0.329177 , 0.357566 , 0.383335 , 0.405809 , 0.424178 , 0.437541 , 0.445013 , 0.44587 , 0.439664 , 0.426299 , 0.406055 , 0.379572 , 0.347813 , 0.311998 , 0.273529 , 0.233892 , 0.194558 , 0.156884 , 0.122032 , 0.0909236 , 0.064215 , 0.0422901 , 0.0252634 , 0.0129923 , 0.00510533 , 0.00104827 , 0.000144943 , 0.0016631 , 0.00487298 , 0.00908976 , 0.0136984 , 0.0181649 , 0.0220419 , 0.0249766 , 0.0267242 , 0.0271635 , 0.0263081 , 0.0243034 , 0.0214053 , 0.0179407 , 0.0142595 , 0.0106866 , 0.00748827 , 0.00485214 , 0.00288546 , 0.00162432 , 0.0010492 , 0.00110222 , 0.00170238 , 0.00275639 , 0.00416358 , 0.00581576 , 0.00759489 , 0.00937317 , 0.0110194 , 0.0124116 , 0.0119559 , 0.0132221 , 0.0148018 , 0.0165802 , 0.0183968 , 0.0200593 , 0.0213775 , 0.0221995 , 0.0224315 , 0.0220297 , 0.020985 , 0.0193243 , 0.0171409 , 0.0146318 , 0.012103 , 0.0099043 , 0.00830533 , 0.00738253 , 0.00700349 , 0.00692164 , 0.00690995 , 0.00684571 , 0.00671687 , 0.0065793 , 0.00650706 , 0.0065557 , 0.00673784 , 0.00700793 , 0.00726245 , 0.00736399 , 0.00718693 , 0.0066613 , 0.0057948 , 0.0046822 , 0.00354297 , 0.00282181 , 0.00334167 , 0.00640891 , 0.0137119 , 0.0269344 , 0.0472421 , 0.0749575 , 0.109611 , 0.150227 , 0.1955 , 0.243642 , 0.291998 , 0.3368 , 0.373412 , 0.397129 , 0.404276 , 0.393137 , 0.364387 , 0.320908 , 0.267156 , 0.20835 , 0.149793 , 0.0964313 , 0.0525815 , 0.0216465 , 0.00570347 , 0.00507993 , 0.0181993 , 0.0418926 , 0.0720793 , 0.10448 , 0.135089 , 0.160423 , 0.177716 , 0.185213 , 0.182438 , 0.170269 , 0.15071 , 0.126422 , 0.1002 , 0.0745509 , 0.0514367 , 0.0321766 , 0.0174672 , 0.00747477 , 0.00195112 , 0.000336459 , 0.00183167 , 0.00545892 , 0.0101473 , 0.0148653 , 0.0187803 , 0.0213842 , 0.0225291 , 0.0223604 , 0.0211945 , 0.0194031 , 0.0173381 , 0.0152942 , 0.0134916 , 0.0120727 , 0.0111108 , 0.0106294 , 0.0106236 , 0.0110751 , 0.0170052 , 0.0160132 , 0.0145054 , 0.0126631 , 0.010668 , 0.00866548 , 0.00676377 , 0.00505616 , 0.00363042 , 0.0025522 , 0.0018417 , 0.00146357 , 0.00134246 , 0.00141901 , 0.00171426 , 0.00232835 , 0.00337446 , 0.0049212 , 0.00695096 , 0.00929771 , 0.0116096 , 0.0134252 , 0.0143834 , 0.0144952 , 0.0143301 , 0.0149465 , 0.0174918 , 0.0226544 , 0.0303636 , 0.040031 , 0.0510907 , 0.0631259 , 0.0752043 , 0.0851605 , 0.0901085 , 0.088332 , 0.080886 , 0.0710661 , 0.0621771 , 0.0561794 , 0.0547543 , 0.0610983 , 0.0795802 , 0.112994 , 0.160105 , 0.215865 , 0.273731 , 0.327746 , 0.373014 , 0.405252 , 0.42081 , 0.41763 , 0.396425 , 0.361091 , 0.317896 , 0.273767 , 0.234428 , 0.203182 , 0.180748 , 0.165965 , 0.156686 , 0.150309 , 0.143995 , 0.135042 , 0.121618 , 0.103476 , 0.0820186 , 0.0596391 , 0.0388505 , 0.0217422 , 0.00978702 , 0.00374658 , 0.00354778 , 0.00823637 , 0.0161542 , 0.0253323 , 0.0339276 , 0.0405223 , 0.0442188 , 0.0446152 , 0.0417863 , 0.0363189 , 0.0292876 , 0.0220101 , 0.0156133 , 0.0106965 , 0.00734378 , 0.00540238 , 0.00472274 , 0.00516757 , 0.0064808 , 0.00824537 , 0.0100297 , 0.0115859 , 0.0129063 , 0.0141074 , 0.0152652 , 0.0163198 , 0.0170896 , 0.0173632 , 0.00979481 , 0.00862263 , 0.00729484 , 0.00591388 , 0.00466165 , 0.00375573 , 0.00334662 , 0.00349189 , 0.00419348 , 0.00540593 , 0.00709184 , 0.00929789 , 0.01206 , 0.0152241 , 0.0183744 , 0.0209884 , 0.0228878 , 0.0245839 , 0.0269767 , 0.0305424 , 0.0347179 , 0.0381719 , 0.0398179 , 0.0395245 , 0.0378524 , 0.0353024 , 0.0319946 , 0.0281632 , 0.0249047 , 0.0240852 , 0.02705 , 0.0333277 , 0.0408768 , 0.0478232 , 0.053659 , 0.058146 , 0.0597989 , 0.0574758 , 0.0539176 , 0.0563577 , 0.072548 , 0.106798 , 0.161699 , 0.242628 , 0.356333 , 0.502866 , 0.669697 , 0.834918 , 0.976557 , 1.07915 , 1.13348 , 1.13361 , 1.07683 , 0.967246 , 0.818759 , 0.652962 , 0.492104 , 0.352021 , 0.239735 , 0.155713 , 0.0971561 , 0.0596405 , 0.0376171 , 0.025568 , 0.0197778 , 0.0191196 , 0.0240035 , 0.034416 , 0.0488526 , 0.0647628 , 0.0793816 , 0.0899931 , 0.0940516 , 0.0898994 , 0.0778155 , 0.0603741 , 0.0415489 , 0.0250757 , 0.0131563 , 0.006242 , 0.00375938 , 0.00490577 , 0.0087951 , 0.0141267 , 0.0191371 , 0.0222644 , 0.023007 , 0.0219813 , 0.0200962 , 0.0178471 , 0.0153647 , 0.0128601 , 0.0107995 , 0.00964003 , 0.0095026 , 0.0100946 , 0.0108798 , 0.0113619 , 0.0113022 , 0.0107303 , 0.0157635 , 0.0153926 , 0.0149541 , 0.0148122 , 0.0154883 , 0.0172725 , 0.0198342 , 0.0222967 , 0.024018 , 0.0252214 , 0.0264374 , 0.027518 , 0.0277779 , 0.0272012 , 0.0267161 , 0.0269081 , 0.0272378 , 0.0268452 , 0.0255794 , 0.0237477 , 0.0213736 , 0.0183648 , 0.0151185 , 0.0125627 , 0.0115018 , 0.0119473 , 0.013257 , 0.0149483 , 0.0170861 , 0.0196302 , 0.0216047 , 0.0218857 , 0.0213607 , 0.0221704 , 0.0241527 , 0.0253456 , 0.0254579 , 0.0250292 , 0.0218325 , 0.0150346 , 0.013105 , 0.0295496 , 0.0753556 , 0.162423 , 0.305318 , 0.50655 , 0.742767 , 0.975384 , 1.17163 , 1.31065 , 1.37723 , 1.35962 , 1.25487 , 1.07348 , 0.839389 , 0.58738 , 0.356649 , 0.178436 , 0.0651654 , 0.01159 , 0.00463152 , 0.0294812 , 0.0695689 , 0.108658 , 0.136448 , 0.149723 , 0.148139 , 0.13217 , 0.105323 , 0.0747825 , 0.0477867 , 0.0282174 , 0.0166354 , 0.011986 , 0.0128199 , 0.017744 , 0.0252358 , 0.033019 , 0.0380545 , 0.0380122 , 0.0330677 , 0.0257589 , 0.0187748 , 0.0132709 , 0.00939964 , 0.00743392 , 0.00761593 , 0.00944178 , 0.0118562 , 0.0141664 , 0.0162759 , 0.0180789 , 0.019147 , 0.0191637 , 0.0183764 , 0.01739 , 0.0166497 , 0.0162537 , 0.0160861 , 0.0159715 , 0.0213251 , 0.0209567 , 0.0215303 , 0.0229953 , 0.0247652 , 0.0258188 , 0.0256449 , 0.0248442 , 0.0241041 , 0.0232067 , 0.0217608 , 0.020193 , 0.0190966 , 0.0184028 , 0.0177489 , 0.0171708 , 0.0169959 , 0.0174303 , 0.0185523 , 0.0204566 , 0.0228622 , 0.0246948 , 0.0253012 , 0.0261214 , 0.0295786 , 0.0354812 , 0.0409848 , 0.0452123 , 0.050221 , 0.0556936 , 0.0571039 , 0.0534212 , 0.0503133 , 0.050345 , 0.0530689 , 0.0646638 , 0.0878967 , 0.108259 , 0.113087 , 0.105221 , 0.0833635 , 0.0455262 , 0.0158109 , 0.0318623 , 0.116338 , 0.284299 , 0.543184 , 0.862483 , 1.16998 , 1.39094 , 1.47774 , 1.4112 , 1.20213 , 0.897422 , 0.569686 , 0.289943 , 0.104396 , 0.0263235 , 0.0373156 , 0.0960779 , 0.15896 , 0.198503 , 0.204375 , 0.177527 , 0.129755 , 0.0794091 , 0.0408354 , 0.0201582 , 0.017803 , 0.0295479 , 0.0475214 , 0.0642238 , 0.0749139 , 0.0767087 , 0.0684837 , 0.0525868 , 0.034586 , 0.0200114 , 0.0117035 , 0.00960288 , 0.0117894 , 0.0159136 , 0.0202488 , 0.0234483 , 0.0242674 , 0.0227199 , 0.0204161 , 0.018687 , 0.0177638 , 0.0180216 , 0.0199461 , 0.0229217 , 0.0257551 , 0.0278121 , 0.028877 , 0.0287506 , 0.0275781 , 0.0258324 , 0.0239907 , 0.0224141 , 0.0246586 , 0.0226485 , 0.0214091 , 0.0209863 , 0.0208006 , 0.0207264 , 0.0210211 , 0.0214801 , 0.0215319 , 0.0214686 , 0.0221418 , 0.0237324 , 0.0253648 , 0.0265491 , 0.0277354 , 0.0295636 , 0.0319661 , 0.0338775 , 0.0348085 , 0.0365033 , 0.0394872 , 0.0407892 , 0.0392034 , 0.0384408 , 0.0405242 , 0.0418323 , 0.0396371 , 0.0371129 , 0.0370285 , 0.0373246 , 0.0373398 , 0.0392197 , 0.0390851 , 0.0340123 , 0.0314141 , 0.0352939 , 0.0427685 , 0.0619231 , 0.0938409 , 0.117748 , 0.123003 , 0.110737 , 0.0778467 , 0.0536547 , 0.0946495 , 0.229213 , 0.464906 , 0.802824 , 1.18709 , 1.49625 , 1.61871 , 1.51855 , 1.2409 , 0.879617 , 0.532447 , 0.269029 , 0.123846 , 0.0917875 , 0.12733 , 0.172002 , 0.186052 , 0.158651 , 0.105643 , 0.0554482 , 0.0289841 , 0.0315469 , 0.0543414 , 0.0818587 , 0.100774 , 0.102975 , 0.0873758 , 0.0615809 , 0.0368695 , 0.0215627 , 0.0181088 , 0.0227519 , 0.029255 , 0.0347744 , 0.0386554 , 0.037772 , 0.0301464 , 0.0206536 , 0.0151667 , 0.0136358 , 0.0146806 , 0.0190737 , 0.0253029 , 0.0303093 , 0.0341849 , 0.0375755 , 0.0388526 , 0.0377714 , 0.0366236 , 0.0360629 , 0.034298 , 0.0308727 , 0.02788 , 0.0268486 , 0.0269557 , 0.0264125 , 0.021598 , 0.022241 , 0.0226435 , 0.0242181 , 0.026962 , 0.0299954 , 0.0324136 , 0.0333783 , 0.0333887 , 0.0339025 , 0.0349808 , 0.0348854 , 0.0337876 , 0.0343454 , 0.0358781 , 0.0350946 , 0.0337713 , 0.0342668 , 0.0332864 , 0.0297716 , 0.0270618 , 0.0254475 , 0.0242681 , 0.0266014 , 0.0328404 , 0.0372261 , 0.0370022 , 0.034314 , 0.0304205 , 0.0275706 , 0.0303783 , 0.0368595 , 0.0439475 , 0.0531907 , 0.0545437 , 0.0428279 , 0.0331249 , 0.0295077 , 0.0399689 , 0.0725462 , 0.101989 , 0.114471 , 0.103914 , 0.0746333 , 0.0983134 , 0.250518 , 0.567332 , 1.06077 , 1.63731 , 2.09583 , 2.27853 , 2.14748 , 1.75064 , 1.2052 , 0.683628 , 0.321633 , 0.145165 , 0.10421 , 0.12107 , 0.127204 , 0.105573 , 0.0750825 , 0.0569206 , 0.0641839 , 0.0877578 , 0.108327 , 0.112462 , 0.0940305 , 0.064055 , 0.0411864 , 0.0323789 , 0.0354354 , 0.046722 , 0.0609998 , 0.0698175 , 0.0639933 , 0.0437268 , 0.0238678 , 0.0164474 , 0.017048 , 0.0197213 , 0.0251961 , 0.0303488 , 0.0303565 , 0.0282782 , 0.027204 , 0.0263609 , 0.0272818 , 0.0308093 , 0.034726 , 0.0378988 , 0.0399136 , 0.0394767 , 0.0365587 , 0.0327533 , 0.0286526 , 0.0240452 , 0.0201748 , 0.0188457 , 0.0199933 , 0.00597483 , 0.00729258 , 0.00904695 , 0.0112189 , 0.0137817 , 0.0167017 , 0.0199388 , 0.0234475 , 0.0271779 , 0.0310766 , 0.035088 , 0.039155 , 0.043221 , 0.0472298 , 0.0511271 , 0.0548616 , 0.058385 , 0.0616531 , 0.0646261 , 0.0672686 , 0.0695507 , 0.0714472 , 0.0729381 , 0.0740087 , 0.07465 , 0.0748567 , 0.0746298 , 0.0739734 , 0.072897 , 0.0714132 , 0.0695386 , 0.0672937 , 0.0647017 , 0.0617899 , 0.0585891 , 0.0551341 , 0.0514638 , 0.0476215 , 0.0436546 , 0.0396146 , 0.035556 , 0.0315361 , 0.0276132 , 0.0238457 , 0.0202903 , 0.0170005 , 0.0140256 , 0.011409 , 0.00918735 , 0.00738948 , 0.00603622 , 0.00513971 , 0.00470342 , 0.00472219 , 0.00518252 , 0.00606299 , 0.00733493 , 0.00896317 , 0.0109069 , 0.0131208 , 0.0155559 , 0.0181608 , 0.0208825 , 0.023668 , 0.0264641 , 0.0292194 , 0.031884 , 0.0344108 , 0.0367553 , 0.0388774 , 0.0407399 , 0.0423107 , 0.0435626 , 0.0444732 , 0.0450262 , 0.0452112 , 0.0450242 , 0.0444676 , 0.0435509 , 0.042289 , 0.0407041 , 0.0388242 , 0.0366824 , 0.0343171 , 0.031771 , 0.02909 , 0.0263233 , 0.023522 , 0.0207383 , 0.0180252 , 0.015435 , 0.0130192 , 0.0108272 , 0.00890563 , 0.00729747 , 0.00604129 , 0.00517024 , 0.0047115 , 0.00468536 , 0.00510477 , 0.0324051 , 0.0324859 , 0.032291 , 0.0318262 , 0.0311217 , 0.0302261 , 0.0292015 , 0.0281166 , 0.0270407 , 0.0260378 , 0.0251607 , 0.0244479 , 0.0239197 , 0.0235787 , 0.0234107 , 0.0233887 , 0.0234782 , 0.0236416 , 0.0238444 , 0.0240578 , 0.0242605 , 0.0244396 , 0.0245881 , 0.0247043 , 0.024789 , 0.0248443 , 0.0248715 , 0.0248699 , 0.0248366 , 0.0247673 , 0.0246566 , 0.0245025 , 0.0243088 , 0.0240899 , 0.0238729 , 0.0236979 , 0.0236141 , 0.0236734 , 0.0239204 , 0.0243834 , 0.0250672 , 0.0259505 , 0.0269875 , 0.0281134 , 0.029253 , 0.0303285 , 0.0312674 , 0.0320089 , 0.0325077 , 0.0327378 , 0.0326946 , 0.0323972 , 0.0318898 , 0.0312431 , 0.0305555 , 0.0299495 , 0.0295687 , 0.0295681 , 0.0301018 , 0.0313084 , 0.0332945 , 0.0361211 , 0.0397936 , 0.0442583 , 0.0494056 , 0.0550784 , 0.0610856 , 0.0672162 , 0.0732524 , 0.0789811 , 0.0842022 , 0.0887345 , 0.0924198 , 0.0951264 , 0.0967551 , 0.0972424 , 0.0965658 , 0.094747 , 0.0918536 , 0.0879956 , 0.0833219 , 0.0780117 , 0.0722647 , 0.0662908 , 0.0602995 , 0.0544898 , 0.0490422 , 0.0441114 , 0.0398195 , 0.0362518 , 0.0334525 , 0.0314232 , 0.0301231 , 0.0294727 , 0.0293605 , 0.0296526 , 0.0302051 , 0.0308753 , 0.0315332 , 0.0320703 , 0.0190435 , 0.0208558 , 0.0218635 , 0.0219114 , 0.0209097 , 0.0188638 , 0.0159059 , 0.012313 , 0.00849254 , 0.00492968 , 0.00210496 , 0.000411158 , 0.000100609 , 0.00127937 , 0.0039403 , 0.00800791 , 0.0133666 , 0.0198589 , 0.0272586 , 0.0352412 , 0.0433713 , 0.0511216 , 0.0579249 , 0.0632423 , 0.0666346 , 0.0678162 , 0.0666888 , 0.0633493 , 0.0580776 , 0.051304 , 0.0435579 , 0.0354046 , 0.0273785 , 0.0199305 , 0.0134011 , 0.0080239 , 0.00395279 , 0.00129061 , 9.96712e-05 , 0.000382722 , 0.00204187 , 0.00484086 , 0.00840042 , 0.012242 , 0.0158699 , 0.0188602 , 0.0209246 , 0.0219297 , 0.0218782 , 0.0208695 , 0.0190627 , 0.0166552 , 0.0138803 , 0.011018 , 0.00840494 , 0.0064303 , 0.00551038 , 0.0060434 , 0.00836001 , 0.0126899 , 0.0191592 , 0.027817 , 0.0386709 , 0.0517046 , 0.066859 , 0.0839769 , 0.102735 , 0.122595 , 0.142808 , 0.162472 , 0.180644 , 0.196452 , 0.209186 , 0.218348 , 0.223654 , 0.225006 , 0.222458 , 0.216184 , 0.20646 , 0.193657 , 0.178241 , 0.160779 , 0.141929 , 0.122412 , 0.102971 , 0.0843031 , 0.067008 , 0.0515452 , 0.0382238 , 0.0272126 , 0.0185639 , 0.0122347 , 0.00809891 , 0.00595356 , 0.00552357 , 0.00647399 , 0.008432 , 0.0110151 , 0.0138576 , 0.0166294 , 0.00690205 , 0.00722718 , 0.00852989 , 0.0106291 , 0.0133539 , 0.0166002 , 0.0203299 , 0.0244975 , 0.0289432 , 0.0333209 , 0.0371371 , 0.0399143 , 0.0413986 , 0.0416741 , 0.0410965 , 0.0401005 , 0.0390146 , 0.0379819 , 0.0369911 , 0.0359694 , 0.0348802 , 0.0337774 , 0.0327923 , 0.032072 , 0.0317146 , 0.0317374 , 0.0320833 , 0.0326522 , 0.0333412 , 0.0340889 , 0.034906 , 0.0358673 , 0.0370455 , 0.0384215 , 0.0398469 , 0.0410887 , 0.0419011 , 0.0420646 , 0.0413929 , 0.0397511 , 0.0370954 , 0.033525 , 0.0293104 , 0.0248537 , 0.0205671 , 0.0167421 , 0.0135065 , 0.0108856 , 0.00889789 , 0.00761145 , 0.00714019 , 0.00760551 , 0.0090936 , 0.0116113 , 0.0150254 , 0.0189961 , 0.0229762 , 0.0263821 , 0.0289496 , 0.0311143 , 0.0341485 , 0.0398993 , 0.0502461 , 0.066595 , 0.0896923 , 0.119786 , 0.156878 , 0.200734 , 0.250524 , 0.304335 , 0.358995 , 0.410425 , 0.454411 , 0.487417 , 0.507117 , 0.512525 , 0.503822 , 0.482061 , 0.448901 , 0.406462 , 0.357305 , 0.304419 , 0.25108 , 0.200517 , 0.155436 , 0.117609 , 0.0877285 , 0.0655321 , 0.0500878 , 0.0400757 , 0.0339989 , 0.0303541 , 0.0278093 , 0.0253791 , 0.0225371 , 0.0192093 , 0.0156535 , 0.0122847 , 0.00951878 , 0.0076695 , 0.0155053 , 0.0195481 , 0.0240146 , 0.0280157 , 0.0308602 , 0.0320271 , 0.0311877 , 0.0283669 , 0.0241176 , 0.0194375 , 0.0153589 , 0.012522 , 0.0110942 , 0.0110438 , 0.0124479 , 0.0155221 , 0.0203656 , 0.0267505 , 0.0342257 , 0.0423881 , 0.0509925 , 0.0598004 , 0.0683177 , 0.0756428 , 0.0806075 , 0.0822142 , 0.0801279 , 0.0748713 , 0.0675643 , 0.0593717 , 0.0510435 , 0.0428298 , 0.0347768 , 0.0271346 , 0.0204914 , 0.0154873 , 0.0124033 , 0.0110741 , 0.0111875 , 0.0126163 , 0.0154169 , 0.0194697 , 0.0241367 , 0.028344 , 0.0310712 , 0.0318197 , 0.0306862 , 0.0280848 , 0.0244652 , 0.0202858 , 0.0161911 , 0.0131453 , 0.0122949 , 0.0144918 , 0.0196716 , 0.0265637 , 0.0331697 , 0.0378243 , 0.0399278 , 0.0396668 , 0.0372641 , 0.0330152 , 0.0284649 , 0.0274902 , 0.0356796 , 0.05799 , 0.096683 , 0.151398 , 0.220721 , 0.302922 , 0.394818 , 0.490113 , 0.579252 , 0.65139 , 0.697301 , 0.711573 , 0.693314 , 0.645639 , 0.574632 , 0.488198 , 0.39495 , 0.303113 , 0.219578 , 0.149289 , 0.0950602 , 0.0576643 , 0.0359863 , 0.0271864 , 0.02717 , 0.0315098 , 0.036493 , 0.0397454 , 0.0402138 , 0.0377709 , 0.0328895 , 0.0265495 , 0.0201465 , 0.0151561 , 0.0126389 , 0.0129017 , 0.036383 , 0.031369 , 0.0258404 , 0.0206764 , 0.0163738 , 0.0130964 , 0.0110855 , 0.0107627 , 0.0122194 , 0.0146645 , 0.016736 , 0.0176222 , 0.0176115 , 0.0173348 , 0.0168167 , 0.015508 , 0.0131313 , 0.0102467 , 0.00778467 , 0.00623985 , 0.00549736 , 0.00515999 , 0.00489423 , 0.00460576 , 0.00437253 , 0.00426176 , 0.00426782 , 0.00438679 , 0.00466643 , 0.00514932 , 0.00582718 , 0.00674607 , 0.00813436 , 0.0102912 , 0.0131422 , 0.0158667 , 0.0174514 , 0.0178053 , 0.0177088 , 0.0175954 , 0.0169394 , 0.0150901 , 0.0125705 , 0.0109235 , 0.0111927 , 0.0132479 , 0.016513 , 0.0207219 , 0.0258143 , 0.0313462 , 0.0363759 , 0.040073 , 0.0422239 , 0.0430161 , 0.0426068 , 0.0413154 , 0.0402032 , 0.040662 , 0.0427616 , 0.0445132 , 0.0435745 , 0.0394013 , 0.033135 , 0.0262013 , 0.021054 , 0.023772 , 0.0437555 , 0.089301 , 0.163992 , 0.266982 , 0.394051 , 0.536034 , 0.676842 , 0.795688 , 0.873545 , 0.898999 , 0.869863 , 0.791535 , 0.67508 , 0.536194 , 0.393491 , 0.264613 , 0.161593 , 0.0888067 , 0.0445612 , 0.0238608 , 0.0199505 , 0.025209 , 0.0329077 , 0.0390391 , 0.0424197 , 0.0433991 , 0.042841 , 0.0418064 , 0.0412373 , 0.0414773 , 0.0421641 , 0.0425917 , 0.0420654 , 0.0400605 , 0.010697 , 0.00890514 , 0.0104971 , 0.0143401 , 0.0188057 , 0.023123 , 0.0273922 , 0.0311041 , 0.0328121 , 0.0319717 , 0.029896 , 0.0280525 , 0.0262651 , 0.0238539 , 0.0218294 , 0.0221567 , 0.025272 , 0.0296535 , 0.0341437 , 0.0395137 , 0.0470005 , 0.0562543 , 0.0651948 , 0.0716783 , 0.0750421 , 0.0758991 , 0.0748176 , 0.0714974 , 0.0651921 , 0.0561734 , 0.046584 , 0.0389997 , 0.0340901 , 0.0303 , 0.0261695 , 0.0226735 , 0.0218614 , 0.0238847 , 0.0267805 , 0.0288023 , 0.0303108 , 0.0320263 , 0.032933 , 0.0313839 , 0.027447 , 0.0229428 , 0.0188328 , 0.0146752 , 0.0106679 , 0.00842965 , 0.00976222 , 0.0150562 , 0.0225996 , 0.0293192 , 0.0330685 , 0.0349307 , 0.0380142 , 0.0431381 , 0.0485396 , 0.0545823 , 0.063512 , 0.0726757 , 0.0741307 , 0.0643604 , 0.048251 , 0.0323506 , 0.0241402 , 0.0390459 , 0.0970509 , 0.208146 , 0.366112 , 0.554745 , 0.751725 , 0.927304 , 1.04869 , 1.09151 , 1.04822 , 0.928012 , 0.752669 , 0.552398 , 0.360225 , 0.203557 , 0.0968518 , 0.0402158 , 0.0233363 , 0.0310327 , 0.0483209 , 0.0639126 , 0.0715809 , 0.0702759 , 0.0631951 , 0.0548784 , 0.0479957 , 0.0428608 , 0.0390427 , 0.0360439 , 0.0328098 , 0.0281698 , 0.0220162 , 0.0155728 , 0.034114 , 0.0376118 , 0.0397246 , 0.0387067 , 0.0351593 , 0.0307225 , 0.0255572 , 0.0196044 , 0.0148956 , 0.0132343 , 0.0133964 , 0.0142424 , 0.0168304 , 0.0206025 , 0.0223845 , 0.0216466 , 0.0207094 , 0.0196818 , 0.0167327 , 0.0124966 , 0.0094187 , 0.00808259 , 0.00755352 , 0.00752123 , 0.0082847 , 0.00913449 , 0.00897084 , 0.00814959 , 0.00757295 , 0.0076035 , 0.00893888 , 0.0124663 , 0.0172103 , 0.0201108 , 0.0207585 , 0.0218104 , 0.0231427 , 0.0214173 , 0.0170566 , 0.0142506 , 0.0137426 , 0.0137635 , 0.0152213 , 0.0198209 , 0.0256772 , 0.0306665 , 0.0352197 , 0.039213 , 0.039915 , 0.0361986 , 0.031453 , 0.0291487 , 0.0285007 , 0.0268923 , 0.0247809 , 0.0248321 , 0.026026 , 0.0256249 , 0.027662 , 0.037699 , 0.0526177 , 0.0683187 , 0.0862161 , 0.099306 , 0.0935731 , 0.0709035 , 0.046715 , 0.0356596 , 0.0628853 , 0.1643 , 0.353932 , 0.611628 , 0.902346 , 1.1822 , 1.39192 , 1.47275 , 1.3985 , 1.18988 , 0.902605 , 0.603165 , 0.346354 , 0.163817 , 0.0636292 , 0.0332413 , 0.0453528 , 0.0709527 , 0.0906402 , 0.0955693 , 0.085597 , 0.0678404 , 0.0503182 , 0.0369059 , 0.0290228 , 0.0263822 , 0.0259619 , 0.0251256 , 0.0246041 , 0.0257709 , 0.0281712 , 0.0309313 , 0.0400537 , 0.0352 , 0.028666 , 0.0241108 , 0.0224079 , 0.0216454 , 0.0229264 , 0.0284604 , 0.034862 , 0.0379464 , 0.039405 , 0.0400519 , 0.0368515 , 0.0316 , 0.0285942 , 0.0263485 , 0.0234928 , 0.0235915 , 0.0272524 , 0.0300369 , 0.0310211 , 0.0333273 , 0.0369942 , 0.0387223 , 0.0379874 , 0.0378773 , 0.039496 , 0.0398497 , 0.0365019 , 0.032274 , 0.0307689 , 0.0302782 , 0.0270457 , 0.0234241 , 0.0242363 , 0.0277501 , 0.029268 , 0.0315787 , 0.0377316 , 0.0417664 , 0.0401132 , 0.0381328 , 0.0359225 , 0.0294641 , 0.0229114 , 0.0215144 , 0.0225309 , 0.0240458 , 0.028451 , 0.0350052 , 0.0397447 , 0.0413559 , 0.0409791 , 0.0406693 , 0.0428945 , 0.0456742 , 0.0433646 , 0.0378816 , 0.0340705 , 0.0291637 , 0.0269023 , 0.0382903 , 0.0598005 , 0.0844617 , 0.108876 , 0.113329 , 0.0869845 , 0.0535852 , 0.0476204 , 0.114957 , 0.308484 , 0.632953 , 1.03077 , 1.42242 , 1.72185 , 1.84323 , 1.73899 , 1.43507 , 1.02409 , 0.619498 , 0.30505 , 0.115809 , 0.0449827 , 0.0527941 , 0.0863167 , 0.108442 , 0.106356 , 0.0844912 , 0.0570033 , 0.0363398 , 0.0273148 , 0.0287438 , 0.0347834 , 0.0400015 , 0.0430371 , 0.0438002 , 0.042635 , 0.0414093 , 0.0413753 , 0.0416337 , 0.0378865 , 0.0328069 , 0.0319588 , 0.0362343 , 0.0390831 , 0.0390451 , 0.0389198 , 0.036119 , 0.029513 , 0.0247229 , 0.0223743 , 0.0208099 , 0.0226596 , 0.0261062 , 0.0287103 , 0.0321387 , 0.0340234 , 0.0327825 , 0.0323281 , 0.0325638 , 0.0301172 , 0.0268445 , 0.025754 , 0.0266599 , 0.0283083 , 0.0292481 , 0.0284802 , 0.0267658 , 0.0257585 , 0.0270666 , 0.0300086 , 0.0314336 , 0.0316227 , 0.033763 , 0.0352344 , 0.0322452 , 0.0291868 , 0.0275036 , 0.0232652 , 0.0206768 , 0.0230991 , 0.0258213 , 0.029732 , 0.0361466 , 0.0390911 , 0.0390263 , 0.0396463 , 0.0367959 , 0.0305777 , 0.0304879 , 0.0376052 , 0.0409835 , 0.0371181 , 0.0354983 , 0.0413177 , 0.0474305 , 0.0517091 , 0.0588593 , 0.0604925 , 0.0539479 , 0.0461463 , 0.0344753 , 0.0323134 , 0.053648 , 0.0863636 , 0.118974 , 0.125717 , 0.0893852 , 0.0506844 , 0.0739862 , 0.234912 , 0.592167 , 1.1086 , 1.66403 , 2.11101 , 2.29808 , 2.13456 , 1.67467 , 1.09723 , 0.584036 , 0.235383 , 0.0714702 , 0.0493868 , 0.0885961 , 0.121229 , 0.118191 , 0.0861656 , 0.050836 , 0.0320632 , 0.0335487 , 0.0452563 , 0.0559034 , 0.0610742 , 0.0590319 , 0.0525333 , 0.0461761 , 0.0404824 , 0.0370562 , 0.038053 , 0.0401169 , 1.70752 , 1.6933 , 1.67472 , 1.65069 , 1.61977 , 1.58002 , 1.52877 , 1.46237 , 1.37595 , 1.26337 , 1.11825 , 0.937154 , 0.72565 , 0.502582 , 0.293739 , 0.117829 , -0.0197296 , -0.122495 , -0.196958 , -0.249331 , -0.284613 , -0.306579 , -0.31802 , -0.321039 , -0.317212 , -0.307765 , -0.293659 , -0.275654 , -0.254374 , -0.230306 , -0.203856 , -0.175339 , -0.145002 , -0.113041 , -0.0795937 , -0.0447563 , -0.00859721 , 0.028839 , 0.0675293 , 0.107451 , 0.148591 , 0.190915 , 0.234383 , 0.278939 , 0.324502 , 0.370972 , 0.418222 , 0.466114 , 0.51449 , 0.563182 , 0.612018 , 0.660825 , 0.709433 , 0.757685 , 0.80543 , 0.852533 , 0.898873 , 0.944339 , 0.988838 , 1.03228 , 1.0746 , 1.11573 , 1.15561 , 1.19419 , 1.23143 , 1.26729 , 1.30175 , 1.33477 , 1.36635 , 1.39646 , 1.42511 , 1.45229 , 1.47801 , 1.50228 , 1.52511 , 1.54654 , 1.56658 , 1.58527 , 1.60266 , 1.61877 , 1.63367 , 1.64739 , 1.66 , 1.67153 , 1.68203 , 1.69156 , 1.70016 , 1.70785 , 1.71467 , 1.72062 , 1.7257 , 1.72988 , 1.73313 , 1.73535 , 1.73645 , 1.73628 , 1.73465 , 1.73131 , 1.72596 , 1.71819 , 0.00927192 , 0.0356029 , 0.062041 , 0.0888246 , 0.116215 , 0.144478 , 0.173876 , 0.204665 , 0.237105 , 0.271495 , 0.308206 , 0.347733 , 0.39073 , 0.438031 , 0.490627 , 0.549604 , 0.616021 , 0.690712 , 0.774043 , 0.865664 , 0.964321 , 1.06786 , 1.17349 , 1.27823 , 1.37945 , 1.47528 , 1.56473 , 1.64769 , 1.72468 , 1.79667 , 1.8648 , 1.93012 , 1.99344 , 2.05502 , 2.11449 , 2.17072 , 2.22191 , 2.26576 , 2.29967 , 2.32098 , 2.32687 , 2.31435 , 2.27998 , 2.21997 , 2.13081 , 2.01132 , 1.86595 , 1.70723 , 1.55302 , 1.4189 , 1.31247 , 1.23381 , 1.17893 , 1.14261 , 1.11977 , 1.10584 , 1.09671 , 1.08856 , 1.07762 , 1.05998 , 1.03144 , 0.987533 , 0.923645 , 0.835678 , 0.721335 , 0.582127 , 0.42507 , 0.262159 , 0.106639 , -0.0314806 , -0.147382 , -0.240531 , -0.312799 , -0.366984 , -0.405999 , -0.432524 , -0.448897 , -0.457094 , -0.458763 , -0.45525 , -0.447643 , -0.436805 , -0.423398 , -0.407915 , -0.390705 , -0.371998 , -0.351934 , -0.330598 , -0.308047 , -0.284342 , -0.259569 , -0.233857 , -0.207375 , -0.180327 , -0.152937 , -0.125422 , -0.0979759 , -0.0707404 , -0.0437961 , -0.0171482 , -2.11324 , -2.10575 , -2.09627 , -2.08463 , -2.07088 , -2.05525 , -2.03807 , -2.01958 , -1.99972 , -1.97783 , -1.95255 , -1.92181 , -1.88293 , -1.83294 , -1.76888 , -1.68805 , -1.58809 , -1.4666 , -1.32063 , -1.14681 , -0.943439 , -0.715328 , -0.477746 , -0.25214 , -0.0546052 , 0.110506 , 0.247611 , 0.364331 , 0.467983 , 0.564255 , 0.657037 , 0.748834 , 0.84147 , 0.936889 , 1.03774 , 1.14745 , 1.26948 , 1.40574 , 1.55466 , 1.7104 , 1.86473 , 2.0106 , 2.14457 , 2.26642 , 2.37721 , 2.47756 , 2.56703 , 2.64441 , 2.70844 , 2.75829 , 2.79368 , 2.81455 , 2.82079 , 2.81185 , 2.78636 , 2.74123 , 2.66967 , 2.55611 , 2.36501 , 2.02675 , 1.50906 , 1.02606 , 0.741233 , 0.595032 , 0.521063 , 0.485234 , 0.470744 , 0.468885 , 0.474956 , 0.486355 , 0.501562 , 0.519546 , 0.539376 , 0.560003 , 0.580101 , 0.597898 , 0.610769 , 0.613996 , 0.596025 , 0.511494 , -0.139254 , -1.97372 , -2.16434 , -2.20877 , -2.22099 , -2.22073 , -2.21423 , -2.20436 , -2.19286 , -2.181 , -2.16974 , -2.15973 , -2.15128 , -2.14444 , -2.13902 , -2.13469 , -2.131 , -2.12748 , -2.12367 , -2.11908 , 2.67249 , 2.67942 , 2.68682 , 2.69438 , 2.70131 , 2.70606 , 2.70603 , 2.69767 , 2.67664 , 2.63789 , 2.57466 , 2.47567 , 2.31787 , 2.05278 , 1.61216 , 1.06289 , 0.642886 , 0.387316 , 0.22005 , 0.090006 , -0.0281709 , -0.146101 , -0.267235 , -0.388799 , -0.503088 , -0.600197 , -0.671836 , -0.713681 , -0.724708 , -0.704458 , -0.650351 , -0.556951 , -0.418993 , -0.239657 , -0.0392302 , 0.150762 , 0.30806 , 0.429503 , 0.523979 , 0.603066 , 0.6759 , 0.747268 , 0.81773 , 0.885266 , 0.947514 , 1.00341 , 1.05373 , 1.10065 , 1.14694 , 1.1948 , 1.24485 , 1.2955 , 1.34326 , 1.38395 , 1.41444 , 1.43417 , 1.44558 , 1.45314 , 1.46146 , 1.47363 , 1.49084 , 1.51302 , 1.53985 , 1.57161 , 1.60979 , 1.65757 , 1.72158 , 1.81949 , 2.02588 , 3.02082 , -1.86658 , -1.6046 , -1.49421 , -1.42379 , -1.36979 , -1.324 , -1.28276 , -1.24432 , -1.20798 , -1.17363 , -1.14148 , -1.11188 , -1.0852 , -1.06206 , -1.0438 , -1.03348 , -1.03831 , -1.07659 , -1.20779 , -1.72117 , -2.95626 , 2.87609 , 2.74615 , 2.69449 , 2.67092 , 2.6603 , 2.65677 , 2.65755 , 2.66105 , 2.66624 , 1.39664 , 1.39561 , 1.40922 , 1.43809 , 1.48436 , 1.55327 , 1.65566 , 1.80944 , 2.03448 , 2.33145 , 2.6601 , 2.96849 , -3.04769 , -2.82392 , -2.64383 , -2.50297 , -2.38774 , -2.27974 , -2.1627 , -2.02814 , -1.87717 , -1.71464 , -1.53847 , -1.33391 , -1.08194 , -0.789293 , -0.511825 , -0.306563 , -0.180248 , -0.107957 , -0.0597958 , -0.0107578 , 0.0514702 , 0.114317 , 0.139877 , 0.0955203 , -0.00596936 , -0.0926043 , -0.0726194 , 0.0734262 , 0.241499 , 0.343472 , 0.39251 , 0.430244 , 0.481634 , 0.548528 , 0.618661 , 0.680013 , 0.728929 , 0.769082 , 0.806767 , 0.84678 , 0.890103 , 0.933329 , 0.970032 , 0.994089 , 1.00424 , 1.00526 , 1.00084 , 0.981606 , 0.915542 , 0.745837 , 0.43975 , 0.110198 , -0.113131 , -0.25059 , -0.354269 , -0.443341 , -0.512008 , -0.552175 , -0.565712 , -0.562474 , -0.554751 , -0.554262 , -0.572013 , -0.620967 , -0.724564 , -0.944258 , -1.45407 , -2.2841 , -2.7978 , -3.02887 , 3.12893 , 3.04464 , 2.97606 , 2.91039 , 2.83942 , 2.75674 , 2.6573 , 2.53825 , 2.40021 , 2.2479 , 2.08927 , 1.93386 , 1.79111 , 1.66858 , 1.57037 , 1.49668 , 1.44517 , 1.41268 , -0.142289 , -0.129866 , -0.110716 , -0.084556 , -0.0481578 , 0.00567606 , 0.0872834 , 0.20656 , 0.369285 , 0.581186 , 0.856226 , 1.19253 , 1.52213 , 1.77352 , 1.95695 , 2.11891 , 2.2891 , 2.46874 , 2.65239 , 2.85408 , 3.10229 , -2.87941 , -2.56624 , -2.28688 , -2.04716 , -1.83669 , -1.6508 , -1.4853 , -1.32798 , -1.1622 , -0.980097 , -0.792433 , -0.616931 , -0.455156 , -0.294018 , -0.132281 , 0.00730353 , 0.103042 , 0.156433 , 0.186487 , 0.214404 , 0.247393 , 0.277019 , 0.297736 , 0.316487 , 0.343585 , 0.379076 , 0.415108 , 0.447312 , 0.475517 , 0.499718 , 0.518078 , 0.525852 , 0.51394 , 0.469209 , 0.38149 , 0.257864 , 0.124897 , 0.00523732 , -0.0952509 , -0.173036 , -0.219035 , -0.230193 , -0.218038 , -0.201742 , -0.197327 , -0.214229 , -0.262898 , -0.375196 , -0.645492 , -1.22766 , -1.84864 , -2.16605 , -2.31325 , -2.38963 , -2.43286 , -2.46059 , -2.48516 , -2.51707 , -2.56472 , -2.63426 , -2.73209 , -2.87137 , -3.07997 , 2.88488 , 2.44647 , 1.96329 , 1.51821 , 1.13311 , 0.813742 , 0.560083 , 0.362385 , 0.209478 , 0.0931683 , 0.00696195 , -0.0551397 , -0.0983327 , -0.126463 , -0.14206 , -0.146846 , -1.94055 , -1.92601 , -1.90296 , -1.87279 , -1.83332 , -1.77584 , -1.68565 , -1.54658 , -1.3381 , -0.987437 , -0.302941 , 0.457095 , 0.880307 , 1.18352 , 1.47809 , 1.74563 , 1.98525 , 2.24726 , 2.53685 , 2.78427 , 2.96182 , 3.10632 , -3.02941 , -2.874 , -2.72082 , -2.56201 , -2.37312 , -2.15113 , -1.93856 , -1.77785 , -1.66199 , -1.55136 , -1.41136 , -1.24276 , -1.07677 , -0.925516 , -0.765613 , -0.583248 , -0.406577 , -0.264089 , -0.150255 , -0.0505722 , 0.0334884 , 0.0939526 , 0.133196 , 0.158508 , 0.17205 , 0.173963 , 0.170836 , 0.171219 , 0.17559 , 0.174836 , 0.157471 , 0.123267 , 0.0897777 , 0.0734367 , 0.067448 , 0.0498835 , 0.0103399 , -0.0355409 , -0.0698335 , -0.109257 , -0.203033 , -0.389995 , -0.650832 , -0.944778 , -1.25309 , -1.51836 , -1.69232 , -1.79585 , -1.86661 , -1.92683 , -1.98806 , -2.06386 , -2.1791 , -2.37822 , -2.72998 , 3.03307 , 2.5397 , 2.21082 , 1.99909 , 1.8498 , 1.73741 , 1.65012 , 1.57791 , 1.51002 , 1.43829 , 1.35871 , 1.26307 , 1.10627 , 0.497176 , -1.31537 , -1.64659 , -1.76342 , -1.82975 , -1.87322 , -1.9037 , -1.92552 , -1.93966 , -1.94509 , 2.69897 , 2.71648 , 2.74424 , 2.78216 , 2.83909 , 2.9356 , 3.11325 , -2.74705 , -1.56012 , -0.851334 , -0.614938 , -0.444509 , -0.266278 , -0.0889315 , 0.112483 , 0.407768 , 0.784792 , 1.13924 , 1.47222 , 1.79618 , 2.06533 , 2.29297 , 2.53635 , 2.80071 , 3.04127 , -3.01808 , -2.7435 , -2.39886 , -2.10418 , -1.92505 , -1.78735 , -1.62359 , -1.47648 , -1.40707 , -1.3602 , -1.25365 , -1.12436 , -1.04205 , -0.96224 , -0.822877 , -0.668462 , -0.555153 , -0.458443 , -0.354547 , -0.270905 , -0.23081 , -0.222023 , -0.223529 , -0.219947 , -0.205295 , -0.185541 , -0.175615 , -0.186804 , -0.205448 , -0.198559 , -0.162481 , -0.138447 , -0.170378 , -0.244535 , -0.305915 , -0.374209 , -0.538252 , -0.774749 , -0.943355 , -1.06716 , -1.22348 , -1.36693 , -1.44803 , -1.53477 , -1.73206 , -2.05983 , -2.45193 , -2.93 , 2.90215 , 2.65048 , 2.53706 , 2.47153 , 2.40573 , 2.3196 , 2.20416 , 2.03965 , 1.76729 , 1.26947 , 0.579846 , 0.0755694 , -0.200647 , -0.364531 , -0.478896 , -0.57154 , -0.658915 , -0.762573 , -0.930275 , -1.3453 , -2.51366 , 3.08661 , 2.88256 , 2.79273 , 2.74071 , 2.70916 , 2.69553 , 1.24635 , 1.25823 , 1.28927 , 1.35488 , 1.48062 , 1.6994 , 2.10035 , 2.84842 , -2.79184 , -2.47409 , -2.23889 , -2.0239 , -1.83053 , -1.5676 , -1.15339 , -0.710996 , -0.276463 , 0.129249 , 0.405638 , 0.651719 , 0.971559 , 1.30582 , 1.59722 , 1.88324 , 2.1643 , 2.42981 , 2.70942 , 3.0065 , -2.99469 , -2.71896 , -2.44385 , -2.21091 , -2.00818 , -1.77572 , -1.54403 , -1.37205 , -1.22706 , -1.0863 , -0.981488 , -0.904304 , -0.831022 , -0.766026 , -0.701356 , -0.624779 , -0.558978 , -0.519819 , -0.486343 , -0.462569 , -0.481494 , -0.533169 , -0.566245 , -0.566159 , -0.563258 , -0.583678 , -0.616333 , -0.645817 , -0.686878 , -0.744581 , -0.79136 , -0.823319 , -0.874049 , -0.944603 , -1.00438 , -1.07348 , -1.20299 , -1.39426 , -1.65178 , -2.02633 , -2.41684 , -2.69779 , -2.90177 , -3.05652 , 3.1006 , 2.95755 , 2.74737 , 2.43039 , 1.97123 , 1.44887 , 1.05973 , 0.816288 , 0.648296 , 0.519042 , 0.404197 , 0.261902 , 0.023334 , -0.432155 , -1.18808 , -1.78938 , -2.08998 , -2.27674 , -2.44147 , -2.62963 , -2.91525 , 2.84348 , 2.15337 , 1.69912 , 1.46631 , 1.34294 , 1.27876 , 1.25094 , -0.293364 , -0.281479 , -0.253479 , -0.186806 , -0.0450072 , 0.220383 , 0.740745 , 1.54469 , 2.06627 , 2.38562 , 2.67031 , 2.94789 , -2.90321 , -2.2302 , -1.64183 , -1.17023 , -0.864678 , -0.625927 , -0.323164 , 0.00815171 , 0.376543 , 0.829313 , 1.1919 , 1.42932 , 1.6372 , 1.83756 , 2.03389 , 2.2846 , 2.63615 , 2.95935 , -3.12294 , -2.92462 , -2.67174 , -2.45702 , -2.27059 , -2.04112 , -1.82052 , -1.62682 , -1.43503 , -1.28063 , -1.1419 , -1.00552 , -0.918836 , -0.841941 , -0.733376 , -0.661103 , -0.646074 , -0.600762 , -0.506389 , -0.450698 , -0.482458 , -0.588653 , -0.745391 , -0.890457 , -0.974458 , -1.03761 , -1.10918 , -1.15342 , -1.1707 , -1.20929 , -1.2695 , -1.33491 , -1.45662 , -1.6477 , -1.84648 , -2.06686 , -2.2885 , -2.45635 , -2.62197 , -2.8353 , -3.09077 , 2.8448 , 2.32027 , 1.81579 , 1.52453 , 1.36307 , 1.2393 , 1.09763 , 0.881598 , 0.507882 , -0.0278341 , -0.537364 , -0.910833 , -1.16166 , -1.35471 , -1.56368 , -1.83476 , -2.23532 , -2.81816 , 2.91824 , 2.52889 , 2.21515 , 1.86987 , 1.3363 , 0.653885 , 0.172244 , -0.0855192 , -0.215182 , -0.273848 , -0.293715 , 1.55936 , 1.56483 , 1.56998 , 1.57479 , 1.57931 , 1.58355 , 1.58755 , 1.59136 , 1.59502 , 1.59856 , 1.60203 , 1.60544 , 1.60881 , 1.61214 , 1.61541 , 1.6186 , 1.62167 , 1.62459 , 1.62729 , 1.62975 , 1.6319 , 1.63372 , 1.63517 , 1.63621 , 1.63683 , 1.63699 , 1.63668 , 1.63589 , 1.63462 , 1.63286 , 1.63062 , 1.6279 , 1.62473 , 1.62114 , 1.61716 , 1.61284 , 1.60824 , 1.6034 , 1.59838 , 1.59324 , 1.58801 , 1.58273 , 1.57743 , 1.57211 , 1.56677 , 1.56141 , 1.55602 , 1.55056 , 1.54504 , 1.53945 , 1.53378 , 1.52805 , 1.52228 , 1.51651 , 1.51077 , 1.50513 , 1.49962 , 1.49429 , 1.48919 , 1.48436 , 1.47983 , 1.47561 , 1.47172 , 1.46817 , 1.46495 , 1.46206 , 1.45949 , 1.45723 , 1.45527 , 1.45359 , 1.45219 , 1.45106 , 1.45018 , 1.44956 , 1.4492 , 1.4491 , 1.44928 , 1.44976 , 1.45056 , 1.45172 , 1.45326 , 1.45522 , 1.45763 , 1.46052 , 1.46391 , 1.46781 , 1.47222 , 1.47711 , 1.48246 , 1.48822 , 1.49434 , 1.50076 , 1.50739 , 1.51416 , 1.521 , 1.52782 , 1.53455 , 1.54112 , 1.54748 , 1.55357 , -0.406221 , -0.375517 , -0.345526 , -0.315529 , -0.284969 , -0.253402 , -0.22046 , -0.185858 , -0.14939 , -0.110959 , -0.0706064 , -0.0285383 , 0.0148579 , 0.0590156 , 0.10322 , 0.146656 , 0.188479 , 0.227872 , 0.264096 , 0.296521 , 0.32463 , 0.348012 , 0.366346 , 0.379378 , 0.386912 , 0.388802 , 0.384957 , 0.375347 , 0.360019 , 0.339114 , 0.312885 , 0.281715 , 0.246125 , 0.206781 , 0.164471 , 0.120085 , 0.0745491 , 0.0287689 , -0.0164507 , -0.060466 , -0.102834 , -0.14332 , -0.181881 , -0.218619 , -0.253737 , -0.287502 , -0.320215 , -0.352211 , -0.383871 , -0.415661 , -0.448193 , -0.48232 , -0.519294 , -0.56104 , -0.610679 , -0.673642 , -0.760202 , -0.891943 , -1.11887 , -1.54852 , -2.20774 , -2.73205 , -3.01164 , 3.11846 , 3.02471 , 2.96185 , 2.91691 , 2.8834 , 2.85777 , 2.83796 , 2.8227 , 2.81122 , 2.803 , 2.7977 , 2.79513 , 2.79517 , 2.79777 , 2.80296 , 2.81085 , 2.82165 , 2.83567 , 2.85345 , 2.87577 , 2.90391 , 2.93992 , 2.98732 , 3.05255 , -3.13479 , -2.98014 , -2.69986 , -2.17425 , -1.50898 , -1.07447 , -0.845367 , -0.712694 , -0.625927 , -0.563328 , -0.514534 , -0.474032 , -0.438604 , 0.642702 , 0.587367 , 0.445941 , -0.122729 , -1.54367 , -1.93561 , -2.04229 , -2.08162 , -2.09515 , -2.0959 , -2.0893 , -2.078 , -2.06332 , -2.04593 , -2.02618 , -2.00433 , -1.98086 , -1.95645 , -1.9321 , -1.90895 , -1.88813 , -1.87062 , -1.85715 , -1.84819 , -1.84395 , -1.84453 , -1.84987 , -1.85981 , -1.87406 , -1.89214 , -1.91339 , -1.93695 , -1.96185 , -1.98716 , -2.01206 , -2.03604 , -2.05885 , -2.08042 , -2.1007 , -2.11931 , -2.13539 , -2.14721 , -2.1517 , -2.143 , -2.10777 , -2.0043 , -1.61259 , -0.17703 , 0.385015 , 0.524815 , 0.580087 , 0.606383 , 0.619373 , 0.625244 , 0.627164 , 0.627165 , 0.626768 , 0.627151 , 0.629132 , 0.6331 , 0.638967 , 0.646194 , 0.65392 , 0.661197 , 0.667296 , 0.672102 , 0.676558 , 0.683285 , 0.697821 , 0.732156 , 0.81839 , 1.08298 , 2.09477 , 2.9803 , -3.08676 , -3.03632 , -3.06998 , 3.04804 , 2.41279 , 1.28165 , 0.902335 , 0.78256 , 0.733993 , 0.712389 , 0.702682 , 0.698288 , 0.695951 , 0.694088 , 0.692118 , 0.690064 , 0.688249 , 0.687044 , 0.68668 , 0.687148 , 0.68816 , 0.689149 , 0.689268 , 0.687314 , 0.681488 , 0.668716 , -1.24915 , -1.23479 , -1.20871 , -1.17397 , -1.1331 , -1.08867 , -1.04345 , -1.00018 , -0.961646 , -0.93181 , -0.921016 , -0.972274 , -1.46981 , 3.05477 , 2.83669 , 2.81762 , 2.83319 , 2.8553 , 2.87617 , 2.89389 , 2.90853 , 2.92054 , 2.9301 , 2.93699 , 2.9407 , 2.94066 , 2.93651 , 2.92821 , 2.91609 , 2.90072 , 2.88273 , 2.86257 , 2.8405 , 2.81691 , 2.79367 , 2.77798 , 2.79701 , 3.01387 , -1.53197 , -1.03046 , -0.983408 , -0.999499 , -1.03413 , -1.07643 , -1.12248 , -1.16979 , -1.21587 , -1.25807 , -1.29369 , -1.32001 , -1.33375 , -1.32995 , -1.29947 , -1.22299 , -1.05391 , -0.678958 , -0.0214751 , 0.513327 , 0.765106 , 0.875034 , 0.919872 , 0.931547 , 0.92512 , 0.907894 , 0.88201 , 0.845301 , 0.791894 , 0.713129 , 0.599551 , 0.44542 , 0.256285 , 0.0541539 , -0.129905 , -0.271137 , -0.357087 , -0.3828 , -0.345766 , -0.245748 , -0.0897345 , 0.102198 , 0.300091 , 0.478215 , 0.624712 , 0.739078 , 0.82577 , 0.890167 , 0.936902 , 0.969176 , 0.988273 , 0.992893 , 0.977785 , 0.930326 , 0.820755 , 0.574248 , 0.0493423 , -0.605346 , -0.979822 , -1.14597 , -1.21913 , -1.24697 , -0.537401 , -0.635694 , -1.03978 , -2.33573 , -2.85697 , -2.98326 , -3.02829 , -3.05053 , -3.06617 , -3.08182 , -3.10494 , 3.13275 , 3.03703 , 2.84718 , 2.52263 , 2.11484 , 1.77467 , 1.55842 , 1.43283 , 1.36141 , 1.3211 , 1.29812 , 1.28433 , 1.27519 , 1.26834 , 1.26279 , 1.25845 , 1.25618 , 1.25803 , 1.26749 , 1.28972 , 1.33173 , 1.40374 , 1.52353 , 1.7249 , 2.05148 , 2.46455 , 2.79361 , 2.97196 , 3.05098 , 3.08411 , 3.1025 , 3.1194 , 3.13739 , -3.12579 , -3.09232 , -2.99119 , -2.4917 , -1.08049 , -0.718394 , -0.642728 , -0.623566 , -0.618718 , -0.615891 , -0.610058 , -0.597558 , -0.573796 , -0.53221 , -0.464021 , -0.358696 , -0.205862 , -0.00198001 , 0.235225 , 0.460582 , 0.629417 , 0.72721 , 0.762744 , 0.749746 , 0.699152 , 0.620705 , 0.527112 , 0.434482 , 0.357397 , 0.304048 , 0.276179 , 0.272181 , 0.289941 , 0.328247 , 0.386821 , 0.465009 , 0.559018 , 0.658897 , 0.748437 , 0.80967 , 0.827788 , 0.791284 , 0.689691 , 0.518065 , 0.294108 , 0.0648177 , -0.126893 , -0.269317 , -0.369603 , -0.437155 , -0.479042 , -0.500974 , -0.508583 , -0.507862 , -0.505426 , -0.509887 , -2.30257 , -2.35645 , -2.38717 , -2.41099 , -2.44227 , -2.49302 , -2.57561 , -2.7133 , -2.96168 , 2.87732 , 2.31812 , 1.8908 , 1.59332 , 1.34039 , 1.08854 , 0.828149 , 0.571585 , 0.340001 , 0.148334 , 1.95969e-05 , -0.10896 , -0.185413 , -0.236698 , -0.26968 , -0.289813 , -0.300268 , -0.301415 , -0.290874 , -0.264273 , -0.216678 , -0.143968 , -0.0424087 , 0.0946145 , 0.278393 , 0.513484 , 0.775155 , 1.02381 , 1.25388 , 1.49807 , 1.80454 , 2.22312 , 2.73936 , -3.10028 , -2.83147 , -2.68687 , -2.60772 , -2.56233 , -2.5343 , -2.51154 , -2.48099 , -2.42704 , -2.33038 , -2.16473 , -1.89809 , -1.52843 , -1.14416 , -0.846058 , -0.643775 , -0.506875 , -0.410088 , -0.336438 , -0.271643 , -0.202367 , -0.119878 , -0.0240963 , 0.0773797 , 0.173938 , 0.25616 , 0.318393 , 0.359464 , 0.381467 , 0.388531 , 0.386377 , 0.381214 , 0.377544 , 0.377094 , 0.380181 , 0.387751 , 0.401295 , 0.419458 , 0.433743 , 0.428927 , 0.391861 , 0.32217 , 0.232833 , 0.139246 , 0.0499537 , -0.0327095 , -0.108365 , -0.177752 , -0.24441 , -0.315768 , -0.403794 , -0.527366 , -0.716692 , -1.00916 , -1.39851 , -1.77437 , -2.04182 , -2.20646 , -2.224 , -2.53986 , 3.13617 , 2.54162 , 2.21897 , 2.04294 , 1.93295 , 1.85833 , 1.79722 , 1.72691 , 1.63173 , 1.50776 , 1.352 , 1.13477 , 0.747544 , -0.00495631 , -0.780006 , -1.19941 , -1.43191 , -1.57651 , -1.67206 , -1.73896 , -1.78839 , -1.82512 , -1.85028 , -1.86345 , -1.86348 , -1.84883 , -1.81825 , -1.77186 , -1.70967 , -1.6252 , -1.49484 , -1.26498 , -0.830478 , -0.0818619 , 0.65238 , 1.05142 , 1.25462 , 1.38892 , 1.5048 , 1.60241 , 1.66992 , 1.7183 , 1.78208 , 1.89801 , 2.09761 , 2.43639 , 3.00198 , -2.68141 , -2.34711 , -2.19395 , -2.11225 , -2.04845 , -1.97575 , -1.87933 , -1.75108 , -1.58769 , -1.38867 , -1.15518 , -0.897491 , -0.646088 , -0.438092 , -0.288109 , -0.181652 , -0.0940303 , -0.0136457 , 0.0539368 , 0.100906 , 0.130996 , 0.157773 , 0.190106 , 0.224165 , 0.25018 , 0.262362 , 0.262406 , 0.256536 , 0.25028 , 0.244669 , 0.236018 , 0.219937 , 0.195704 , 0.164565 , 0.124503 , 0.070561 , 0.000464207 , -0.0858013 , -0.195371 , -0.344862 , -0.544438 , -0.780954 , -1.02879 , -1.26551 , -1.46792 , -1.62182 , -1.73382 , -1.82097 , -1.89798 , -1.97585 , -2.07084 , 2.72662 , 2.66826 , 2.62878 , 2.58271 , 2.51899 , 2.4207 , 2.20511 , 1.48232 , 0.381479 , -0.00548378 , -0.159544 , -0.241388 , -0.302915 , -0.376527 , -0.481657 , -0.646945 , -1.00377 , -2.00644 , -2.83544 , -3.11384 , 3.04211 , 2.96932 , 2.92283 , 2.8902 , 2.86622 , 2.85124 , 2.84882 , 2.86234 , 2.89144 , 2.93308 , 2.99277 , 3.10744 , -2.88006 , -2.0433 , -1.11165 , -0.732958 , -0.576056 , -0.497293 , -0.433856 , -0.363934 , -0.288705 , -0.17532 , 0.175374 , 1.37692 , 2.05505 , 2.26584 , 2.36751 , 2.4205 , 2.45106 , 2.48371 , 2.54786 , 2.68201 , 2.93125 , -2.96738 , -2.52531 , -2.15656 , -1.90109 , -1.72741 , -1.59358 , -1.4732 , -1.3576 , -1.24071 , -1.10549 , -0.935972 , -0.741536 , -0.555454 , -0.399965 , -0.268018 , -0.153401 , -0.0676271 , -0.0182738 , 0.00909095 , 0.0359722 , 0.0670905 , 0.0915486 , 0.0993576 , 0.0918871 , 0.0789478 , 0.0676231 , 0.0555343 , 0.0340228 , -0.00564649 , -0.0704311 , -0.166876 , -0.298959 , -0.462381 , -0.642834 , -0.820269 , -0.97818 , -1.11345 , -1.23263 , -1.34078 , -1.44484 , -1.56665 , -1.74153 , -1.99899 , -2.35134 , -2.78335 , 3.10138 , 2.85023 , 2.65843 , 2.19336 , 1.58912 , 1.15739 , 0.939503 , 0.833151 , 0.761526 , 0.68553 , 0.592264 , 0.44251 , 0.016116 , -1.10411 , -1.74223 , -1.97394 , -2.09177 , -2.1917 , -2.34588 , -2.67638 , 2.69565 , 1.81834 , 1.52941 , 1.40618 , 1.33493 , 1.29069 , 1.26502 , 1.25266 , 1.25242 , 1.26692 , 1.29894 , 1.35438 , 1.46561 , 1.76508 , 2.63402 , -2.77353 , -2.43605 , -2.29856 , -2.21641 , -2.09449 , -1.86306 , -1.29196 , -0.145794 , 0.273307 , 0.413581 , 0.521567 , 0.605105 , 0.661836 , 0.748451 , 0.963608 , 1.40118 , 2.00122 , 2.46724 , 2.73232 , 2.88885 , 2.99791 , 3.09751 , -3.05188 , -2.84276 , -2.55744 , -2.24142 , -1.94228 , -1.68851 , -1.48889 , -1.33932 , -1.22874 , -1.12968 , -1.01654 , -0.890131 , -0.763051 , -0.633804 , -0.495253 , -0.361933 , -0.260267 , -0.194154 , -0.145916 , -0.107148 , -0.0849538 , -0.0878105 , -0.115604 , -0.163177 , -0.228345 , -0.311697 , -0.412884 , -0.532355 , -0.666247 , -0.795523 , -0.904445 , -1.00286 , -1.10518 , -1.21615 , -1.34926 , -1.53268 , -1.78683 , -2.09625 , -2.40344 , -2.65922 , -2.85367 , -2.99539 , -3.10366 , 3.07219 , 2.92056 , 1.58803 , 1.45751 , 1.33823 , 1.18771 , 0.885791 , 0.248437 , -0.426582 , -0.798098 , -0.980195 , -1.10334 , -1.2695 , -1.5344 , -2.00491 , -2.74272 , 2.99452 , 2.67873 , 2.45318 , 2.21984 , 1.74462 , 0.741905 , 0.104839 , -0.138017 , -0.244463 , -0.298317 , -0.330165 , -0.3467 , -0.345679 , -0.32551 , -0.28284 , -0.191634 , 0.0426488 , 0.685468 , 1.6479 , 2.13274 , 2.35406 , 2.55435 , 2.87772 , -2.87389 , -2.16781 , -1.69049 , -1.45196 , -1.29018 , -1.15855 , -0.990003 , -0.613743 , 0.089306 , 0.670729 , 0.968041 , 1.12897 , 1.24236 , 1.3617 , 1.53691 , 1.82562 , 2.22731 , 2.62418 , 2.95274 , -3.0729 , -2.88846 , -2.73372 , -2.55292 , -2.34584 , -2.13092 , -1.89601 , -1.65768 , -1.46224 , -1.3164 , -1.19102 , -1.06966 , -0.954751 , -0.843591 , -0.72895 , -0.613501 , -0.505141 , -0.407509 , -0.326409 , -0.280766 , -0.294392 , -0.371357 , -0.480704 , -0.579025 , -0.6608 , -0.754799 , -0.863806 , -0.963846 , -1.06269 , -1.18783 , -1.33841 , -1.51826 , -1.74065 , -1.97519 , -2.19439 , -2.38794 , -2.54693 , -2.70144 , -2.88844 , -3.11731 , 2.85845 , 2.43279 , 2.02666 , 1.76003 , 0.635156 , 0.586482 , 0.537727 , 0.489067 , 0.440676 , 0.392725 , 0.345374 , 0.298764 , 0.253023 , 0.208253 , 0.164541 , 0.121957 , 0.0805548 , 0.0403826 , 0.00148448 , -0.0360868 , -0.0722665 , -0.106967 , -0.140073 , -0.171427 , -0.200826 , -0.228017 , -0.252664 , -0.274366 , -0.292622 , -0.306814 , -0.316169 , -0.319725 , -0.316252 , -0.304167 , -0.281387 , -0.245157 , -0.191796 , -0.116553 , -0.0136707 , 0.122524 , 0.294559 , 0.496519 , 0.710922 , 0.914483 , 1.09011 , 1.23231 , 1.34375 , 1.43006 , 1.49685 , 1.54874 , 1.5892 , 1.62085 , 1.64561 , 1.66492 , 1.67987 , 1.6913 , 1.69988 , 1.70612 , 1.71043 , 1.71314 , 1.71451 , 1.71473 , 1.71394 , 1.71225 , 1.7097 , 1.70631 , 1.7021 , 1.69702 , 1.69103 , 1.68407 , 1.6761 , 1.66703 , 1.65682 , 1.6454 , 1.63271 , 1.61872 , 1.60339 , 1.58668 , 1.56856 , 1.54903 , 1.52805 , 1.50564 , 1.48177 , 1.45646 , 1.42971 , 1.40153 , 1.37194 , 1.34096 , 1.30861 , 1.27492 , 1.23992 , 1.20365 , 1.16614 , 1.12742 , 1.08753 , 1.0465 , 1.00438 , 0.961193 , 0.917004 , 0.871868 , 0.82586 , 0.779068 , 0.7316 , 0.683581 , 1.38043 , 1.48248 , 1.60883 , 1.75325 , 1.90234 , 2.03986 , 2.15347 , 2.23808 , 2.29434 , 2.32569 , 2.33623 , 2.32974 , 2.30948 , 2.27811 , 2.23784 , 2.19046 , 2.13735 , 2.07951 , 2.01756 , 1.95178 , 1.88216 , 1.80844 , 1.73025 , 1.6472 , 1.559 , 1.46563 , 1.36745 , 1.26537 , 1.16087 , 1.05596 , 0.952933 , 0.854078 , 0.761296 , 0.675889 , 0.598458 , 0.528966 , 0.466899 , 0.411437 , 0.361624 , 0.316499 , 0.275185 , 0.236935 , 0.201152 , 0.16738 , 0.135282 , 0.104597 , 0.0751166 , 0.0466502 , 0.0190124 , -0.00798382 , -0.0345241 , -0.0607814 , -0.0869049 , -0.113002 , -0.139128 , -0.165277 , -0.191376 , -0.21729 , -0.242832 , -0.267782 , -0.291908 , -0.314992 , -0.33685 , -0.357345 , -0.376389 , -0.393934 , -0.40995 , -0.424396 , -0.437178 , -0.448117 , -0.456898 , -0.463025 , -0.465777 , -0.464145 , -0.456761 , -0.441806 , -0.416906 , -0.379015 , -0.324359 , -0.248603 , -0.147578 , -0.0190566 , 0.134315 , 0.302651 , 0.470404 , 0.622625 , 0.750614 , 0.852622 , 0.931195 , 0.990472 , 1.0347 , 1.06769 , 1.09279 , 1.11302 , 1.1312 , 1.15022 , 1.17313 , 1.20341 , 1.24503 , 1.30248 , 2.83999 , 2.80913 , 2.76574 , 2.70835 , 2.63549 , 2.5462 , 2.44063 , 2.32034 , 2.18824 , 2.04788 , 1.90257 , 1.75492 , 1.60702 , 1.46109 , 1.31981 , 1.18607 , 1.06226 , 0.949423 , 0.847029 , 0.753055 , 0.664319 , 0.576735 , 0.485384 , 0.384417 , 0.266895 , 0.124981 , -0.0486919 , -0.257252 , -0.493841 , -0.739613 , -0.972083 , -1.17703 , -1.35129 , -1.49777 , -1.6205 , -1.72262 , -1.80637 , -1.87374 , -1.92695 , -1.96865 , -2.00171 , -2.02887 , -2.05241 , -2.0739 , -2.09416 , -2.11333 , -2.13108 , -2.1469 , -2.1603 , -2.17103 , -2.17907 , -2.1847 , -2.18838 , -2.19072 , -2.19239 , -2.19409 , -2.19644 , -2.19995 , -2.20493 , -2.21144 , -2.21926 , -2.22787 , -2.2365 , -2.24418 , -2.24975 , -2.25171 , -2.24767 , -2.2322 , -2.18805 , -2.01888 , -0.250903 , 0.513869 , 0.605217 , 0.626284 , 0.62512 , 0.613358 , 0.595596 , 0.574423 , 0.551782 , 0.529446 , 0.509233 , 0.493104 , 0.483258 , 0.48224 , 0.493214 , 0.520581 , 0.571399 , 0.658652 , 0.808581 , 1.07332 , 1.51417 , 2.03535 , 2.41108 , 2.62411 , 2.74263 , 2.80977 , 2.84684 , 2.86443 , 2.86781 , 2.85942 , 1.32955 , 1.28346 , 1.2375 , 1.19062 , 1.14105 , 1.08707 , 1.02767 , 0.962942 , 0.894001 , 0.822313 , 0.748469 , 0.670836 , 0.584672 , 0.482184 , 0.353999 , 0.193045 , 0.00166133 , -0.202105 , -0.38978 , -0.538686 , -0.640648 , -0.697588 , -0.714636 , -0.696845 , -0.649028 , -0.576781 , -0.486909 , -0.3864 , -0.280592 , -0.17198 , -0.0601342 , 0.0584352 , 0.192338 , 0.361564 , 0.610385 , 1.01218 , 1.55205 , 1.9991 , 2.26488 , 2.41714 , 2.50918 , 2.56659 , 2.60135 , 2.61985 , 2.62649 , 2.62509 , 2.61902 , 2.61096 , 2.60272 , 2.59521 , 2.58876 , 2.58342 , 2.57928 , 2.57673 , 2.57669 , 2.58093 , 2.59269 , 2.61861 , 2.67481 , 2.81349 , -2.99787 , -1.76406 , -1.25949 , -1.1256 , -1.08467 , -1.07832 , -1.08761 , -1.10437 , -1.12472 , -1.1471 , -1.17115 , -1.19711 , -1.22527 , -1.25579 , -1.28874 , -1.32443 , -1.36413 , -1.41117 , -1.47382 , -1.57557 , -1.81952 , -3.05877 , 2.12111 , 1.89002 , 1.78204 , 1.7109 , 1.65831 , 1.61875 , 1.5899 , 1.57007 , 1.55744 , 1.54981 , 1.54467 , 1.53917 , 1.53027 , 1.51507 , 1.49153 , 1.45924 , 1.41968 , 1.37553 , 0.92315 , 0.881084 , 0.833195 , 0.778638 , 0.718639 , 0.6566 , 0.596213 , 0.537687 , 0.473956 , 0.390682 , 0.276583 , 0.14635 , 0.0492564 , 0.0280604 , 0.0729931 , 0.131033 , 0.156891 , 0.142035 , 0.100585 , 0.0475767 , -0.0115022 , -0.0818713 , -0.177023 , -0.315503 , -0.514525 , -0.774185 , -1.06119 , -1.32873 , -1.55628 , -1.7503 , -1.92223 , -2.07687 , -2.21374 , -2.3342 , -2.44657 , -2.5659 , -2.70991 , -2.89287 , -3.12053 , 2.89122 , 2.57976 , 2.24879 , 1.94775 , 1.71765 , 1.55988 , 1.45487 , 1.38386 , 1.33529 , 1.30377 , 1.28763 , 1.28695 , 1.30265 , 1.33627 , 1.39011 , 1.46688 , 1.56864 , 1.6951 , 1.84205 , 2.00138 , 2.16313 , 2.31815 , 2.45972 , 2.58385 , 2.68929 , 2.77737 , 2.85184 , 2.91872 , 2.98661 , 3.06854 , -3.09405 , -2.87075 , -2.34369 , -1.43312 , -0.919101 , -0.716846 , -0.625008 , -0.579142 , -0.55514 , -0.541484 , -0.531361 , -0.519585 , -0.500882 , -0.468822 , -0.415403 , -0.331247 , -0.205711 , -0.0261359 , 0.218297 , 0.511091 , 0.784227 , 0.974494 , 1.07771 , 1.11984 , 1.12592 , 1.11267 , 1.08959 , 1.06134 , 1.03001 , 0.996536 , 0.961134 , 0.626025 , 0.594942 , 0.55963 , 0.528783 , 0.508285 , 0.493424 , 0.46671 , 0.417555 , 0.363871 , 0.331112 , 0.319878 , 0.305384 , 0.260563 , 0.177276 , 0.0652203 , -0.0647241 , -0.211706 , -0.380803 , -0.569697 , -0.763559 , -0.947614 , -1.11949 , -1.28714 , -1.46047 , -1.64552 , -1.8439 , -2.05824 , -2.29937 , -2.5838 , -2.90826 , 3.05843 , 2.79761 , 2.58843 , 2.40107 , 2.21813 , 2.04183 , 1.87255 , 1.68577 , 1.43675 , 1.10899 , 0.766833 , 0.48263 , 0.263605 , 0.0957145 , -0.0279174 , -0.113188 , -0.169861 , -0.208372 , -0.236045 , -0.256096 , -0.268719 , -0.272774 , -0.266972 , -0.250087 , -0.220433 , -0.175172 , -0.110024 , -0.0196375 , 0.101831 , 0.260934 , 0.465935 , 0.725955 , 1.04618 , 1.42663 , 1.86886 , 2.35298 , 2.79732 , 3.13176 , -2.92285 , -2.76874 , -2.66363 , -2.59113 , -2.54047 , -2.5035 , -2.47226 , -2.43692 , -2.38365 , -2.29123 , -2.12201 , -1.79656 , -1.214 , -0.614919 , -0.301005 , -0.176576 , -0.137481 , -0.132512 , -0.136517 , -0.137585 , -0.130455 , -0.111125 , -0.0726544 , -0.00441324 , 0.102623 , 0.243027 , 0.388979 , 0.508357 , 0.588203 , 0.632495 , 0.649532 , 0.645841 , 0.332585 , 0.326544 , 0.321274 , 0.321533 , 0.321499 , 0.309297 , 0.278446 , 0.230573 , 0.167291 , 0.084916 , -0.0218008 , -0.153768 , -0.308585 , -0.482349 , -0.665106 , -0.842791 , -1.01122 , -1.17729 , -1.34234 , -1.49477 , -1.62556 , -1.74742 , -1.89327 , -2.09309 , -2.33797 , -2.57547 , -2.76839 , -2.92154 , -3.05452 , 3.09737 , 2.94463 , 2.73901 , 2.46295 , 2.15825 , 1.89135 , 1.65271 , 1.38851 , 1.09175 , 0.775935 , 0.338402 , -0.426317 , -1.11112 , -1.4617 , -1.67344 , -1.81942 , -1.91638 , -1.97754 , -2.01752 , -2.04674 , -2.069 , -2.08358 , -2.08884 , -2.08427 , -2.07016 , -2.04633 , -2.01135 , -1.96214 , -1.8915 , -1.77311 , -1.43767 , 0.387861 , 0.993088 , 1.16013 , 1.26457 , 1.34835 , 1.42233 , 1.49469 , 1.57444 , 1.67002 , 1.79055 , 1.95116 , 2.18118 , 2.52716 , 3.00819 , -2.77369 , -2.39743 , -2.15964 , -2.02117 , -1.94428 , -1.89463 , -1.83961 , -1.75222 , -1.6193 , -1.43961 , -1.2003 , -0.881143 , -0.537658 , -0.275661 , -0.111638 , -0.0106157 , 0.0561454 , 0.106269 , 0.147637 , 0.179604 , 0.200786 , 0.217554 , 0.241213 , 0.274906 , 0.308481 , 0.328942 , -0.0248021 , -0.0430258 , -0.0580597 , -0.0585578 , -0.0568933 , -0.0752171 , -0.124068 , -0.198514 , -0.291344 , -0.405661 , -0.544591 , -0.691069 , -0.818282 , -0.92699 , -1.03916 , -1.15269 , -1.24481 , -1.3227 , -1.42396 , -1.56121 , -1.71356 , -1.87809 , -2.08977 , -2.37808 , -2.70783 , -3.01033 , 3.00982 , 2.75789 , 2.50733 , 2.26913 , 2.02682 , 1.73662 , 1.39301 , 1.0419 , 0.680461 , 0.307221 , 0.00610793 , -0.207415 , -0.3902 , -0.569347 , -0.744209 , -0.988218 , -1.69949 , -2.89289 , 2.96122 , 2.77832 , 2.67672 , 2.61624 , 2.57639 , 2.54753 , 2.52929 , 2.52582 , 2.54059 , 2.57463 , 2.63043 , 2.72524 , 2.9361 , -2.65651 , -1.49336 , -1.06945 , -0.893239 , -0.785244 , -0.694759 , -0.59422 , -0.467745 , -0.299253 , -0.0372476 , 0.453666 , 1.18476 , 1.72161 , 2.01222 , 2.17991 , 2.28458 , 2.36143 , 2.43915 , 2.54282 , 2.69946 , 2.95048 , -2.92881 , -2.40488 , -1.96653 , -1.68169 , -1.4926 , -1.35804 , -1.2496 , -1.13496 , -0.990856 , -0.818044 , -0.624655 , -0.423006 , -0.253231 , -0.14332 , -0.0738339 , -0.0221197 , 0.00585613 , -0.000307878 , -0.0259251 , -0.0414208 , -0.0347531 , -0.0221192 , -0.362897 , -0.338127 , -0.300752 , -0.283133 , -0.307264 , -0.349524 , -0.381346 , -0.42557 , -0.513495 , -0.612763 , -0.677415 , -0.741228 , -0.851326 , -0.980016 , -1.10071 , -1.25409 , -1.46265 , -1.687 , -1.90818 , -2.14471 , -2.39568 , -2.65282 , -2.93687 , 3.02607 , 2.71233 , 2.43149 , 2.15303 , 1.85359 , 1.55681 , 1.26134 , 0.925793 , 0.590117 , 0.318771 , 0.0306699 , -0.381161 , -0.838189 , -1.29093 , -1.69612 , -1.95871 , -2.15996 , -2.38717 , -2.63406 , -2.95355 , 2.6904 , 1.93124 , 1.52262 , 1.29777 , 1.16698 , 1.09945 , 1.06879 , 1.05763 , 1.06212 , 1.08946 , 1.15484 , 1.28264 , 1.52142 , 1.98138 , 2.67937 , -3.07891 , -2.79077 , -2.59147 , -2.41419 , -2.22063 , -1.92123 , -1.33488 , -0.565841 , -0.0788659 , 0.173426 , 0.325414 , 0.455432 , 0.599701 , 0.77314 , 1.01422 , 1.4129 , 1.9632 , 2.42474 , 2.74128 , 2.9812 , -3.12173 , -2.99152 , -2.85868 , -2.64243 , -2.31449 , -1.94711 , -1.57748 , -1.26915 , -1.08299 , -0.966798 , -0.867606 , -0.78576 , -0.724565 , -0.667511 , -0.607999 , -0.555303 , -0.511929 , -0.472503 , -0.435676 , -0.404891 , -0.38394 , -0.372979 , -0.269386 , -0.239366 , -0.296733 , -0.395579 , -0.457269 , -0.479094 , -0.532162 , -0.635562 , -0.733936 , -0.822951 , -0.958733 , -1.12874 , -1.29336 , -1.4799 , -1.70427 , -1.93312 , -2.15074 , -2.36341 , -2.58777 , -2.82728 , -3.05145 , 3.0104 , 2.70666 , 2.32842 , 2.01862 , 1.81114 , 1.637 , 1.43378 , 1.15251 , 0.750056 , 0.305472 , -0.0578407 , -0.395342 , -0.720332 , -0.994633 , -1.31332 , -1.7722 , -2.36525 , -3.05253 , 2.78426 , 2.5044 , 2.21624 , 1.8835 , 1.35213 , 0.544354 , 0.0242696 , -0.245159 , -0.394373 , -0.463762 , -0.489568 , -0.499998 , -0.502399 , -0.484289 , -0.42214 , -0.285519 , -0.0241718 , 0.459205 , 1.14707 , 1.69041 , 2.04738 , 2.36344 , 2.75697 , -2.96654 , -2.37186 , -1.9671 , -1.70145 , -1.48013 , -1.26023 , -0.995309 , -0.610777 , -0.0899858 , 0.43879 , 0.838744 , 1.08735 , 1.23494 , 1.3495 , 1.51453 , 1.85652 , 2.40746 , 2.89444 , -3.01752 , -2.73148 , -2.53859 , -2.3658 , -2.16572 , -1.95848 , -1.73496 , -1.49793 , -1.31339 , -1.18609 , -1.0882 , -1.02503 , -0.988994 , -0.950407 , -0.899329 , -0.847641 , -0.789624 , -0.69082 , -0.537165 , -0.378018 , -0.299403 , -0.465411 , -0.601421 , -0.712018 , -0.802401 , -0.877096 , -0.939666 , -0.992803 , -1.03851 , -1.07827 , -1.1132 , -1.1441 , -1.17159 , -1.19614 , -1.21806 , -1.23763 , -1.25502 , -1.27037 , -1.28377 , -1.29531 , -1.30503 , -1.31297 , -1.31916 , -1.32361 , -1.32634 , -1.32737 , -1.32671 , -1.32435 , -1.32032 , -1.3146 , -1.3072 , -1.29809 , -1.28727 , -1.27468 , -1.26026 , -1.24391 , -1.2255 , -1.20483 , -1.18164 , -1.15557 , -1.12615 , -1.09275 , -1.05453 , -1.01036 , -0.958741 , -0.897649 , -0.824371 , -0.735307 , -0.625864 , -0.490711 , -0.325051 , -0.127729 , 0.0942168 , 0.323816 , 0.540335 , 0.729289 , 0.88629 , 1.0139 , 1.11723 , 1.20143 , 1.27077 , 1.32851 , 1.37711 , 1.41837 , 1.45362 , 1.48385 , 1.50979 , 1.532 , 1.55091 , 1.56683 , 1.58001 , 1.59063 , 1.59884 , 1.60475 , 1.60841 , 1.60989 , 1.6092 , 1.60632 , 1.60123 , 1.59387 , 1.58412 , 1.57186 , 1.55689 , 1.53897 , 1.51775 , 1.4928 , 1.46355 , 1.42925 , 1.38889 , 1.34114 , 1.28418 , 1.21559 , 1.13212 , 1.02959 , 0.902969 , 0.747282 , 0.560086 , 0.34567 , 0.118081 , -0.102517 , 1.46061 , 1.47227 , 1.47515 , 1.46972 , 1.45639 , 1.43545 , 1.40715 , 1.37178 , 1.32972 , 1.28159 , 1.22832 , 1.17118 , 1.11176 , 1.05184 , 0.993233 , 0.937586 , 0.886239 , 0.84015 , 0.799889 , 0.765689 , 0.737529 , 0.715214 , 0.698443 , 0.686889 , 0.680216 , 0.678143 , 0.680445 , 0.686993 , 0.697766 , 0.712872 , 0.732544 , 0.757136 , 0.787061 , 0.822702 , 0.864281 , 0.911696 , 0.964359 , 1.02109 , 1.08014 , 1.13934 , 1.19641 , 1.24926 , 1.29628 , 1.33641 , 1.36913 , 1.39428 , 1.41192 , 1.42214 , 1.425 , 1.42037 , 1.408 , 1.38743 , 1.3581 , 1.31935 , 1.27065 , 1.21173 , 1.14297 , 1.06562 , 0.982094 , 0.89581 , 0.810728 , 0.730584 , 0.658204 , 0.595161 , 0.541822 , 0.497669 , 0.461669 , 0.432599 , 0.409268 , 0.390645 , 0.375907 , 0.364455 , 0.355884 , 0.349955 , 0.346557 , 0.34568 , 0.347385 , 0.351796 , 0.359096 , 0.369527 , 0.383402 , 0.401128 , 0.423224 , 0.450332 , 0.483225 , 0.522778 , 0.569888 , 0.62532 , 0.689488 , 0.762154 , 0.842149 , 0.927246 , 1.0143 , 1.09969 , 1.17998 , 1.25239 , 1.31512 , 1.36727 , 1.40867 , 1.43961 , 0.101842 , 0.114796 , 0.117655 , 0.11364 , 0.105141 , 0.0941444 , 0.0825498 , 0.0725227 , 0.0671468 , 0.0723489 , 0.105428 , 0.270336 , 2.40887 , 2.90029 , 2.96866 , 2.99109 , 2.99908 , 3.00011 , 2.99694 , 2.99112 , 2.98381 , 2.97605 , 2.96868 , 2.9624 , 2.95766 , 2.95472 , 2.95365 , 2.95433 , 2.95649 , 2.9597 , 2.96336 , 2.96676 , 2.96905 , 2.96927 , 2.96607 , 2.95702 , 2.93548 , 2.87359 , 2.43747 , 0.199497 , 0.0433352 , 0.0127057 , 0.00811071 , 0.0134155 , 0.0230736 , 0.0341476 , 0.0445304 , 0.0523571 , 0.0556973 , 0.0522576 , 0.03896 , 0.0112416 , -0.0381442 , -0.12113 , -0.257011 , -0.471948 , -0.778589 , -1.12671 , -1.42163 , -1.62503 , -1.75444 , -1.83543 , -1.88594 , -1.91686 , -1.93477 , -1.94377 , -1.94662 , -1.94534 , -1.94149 , -1.93628 , -1.93065 , -1.92525 , -1.92048 , -1.91651 , -1.91336 , -1.91096 , -1.90925 , -1.90818 , -1.90779 , -1.90814 , -1.90926 , -1.91106 , -1.91318 , -1.91489 , -1.91498 , -1.91166 , -1.90239 , -1.88369 , -1.8506 , -1.79581 , -1.70792 , -1.56907 , -1.35528 , -1.05384 , -0.707727 , -0.406725 , -0.194763 , -0.059243 , 0.0242721 , 0.0740917 , -0.558911 , -0.277672 , -0.0400642 , 0.14076 , 0.272645 , 0.367637 , 0.434561 , 0.478685 , 0.503639 , 0.513075 , 0.511133 , 0.501898 , 0.488464 , 0.472225 , 0.452763 , 0.428606 , 0.398704 , 0.363776 , 0.326652 , 0.291358 , 0.261589 , 0.239419 , 0.224747 , 0.21555 , 0.208966 , 0.20274 , 0.196492 , 0.192194 , 0.193528 , 0.204387 , 0.22688 , 0.259728 , 0.297997 , 0.334877 , 0.364803 , 0.385942 , 0.40049 , 0.412713 , 0.425984 , 0.440465 , 0.452521 , 0.455898 , 0.443785 , 0.410696 , 0.353448 , 0.270809 , 0.161637 , 0.0223287 , -0.154082 , -0.373806 , -0.631656 , -0.90291 , -1.1547 , -1.36764 , -1.54068 , -1.68274 , -1.80619 , -1.92485 , -2.05312 , -2.20336 , -2.37933 , -2.56924 , -2.74997 , -2.90351 , -3.02614 , -3.12322 , 3.0818 , 3.01822 , 2.96719 , 2.92769 , 2.89866 , 2.87861 , 2.86583 , 2.85876 , 2.85622 , 2.85758 , 2.86268 , 2.87171 , 2.88511 , 2.90351 , 2.92776 , 2.95908 , 2.99915 , 3.05034 , 3.11577 , -3.0839 , -2.978 , -2.84615 , -2.6879 , -2.50921 , -2.32331 , -2.1456 , -1.98581 , -1.84452 , -1.71536 , -1.58827 , -1.45117 , -1.29023 , -1.09067 , -0.843097 , -0.775719 , -0.579019 , -0.443258 , -0.345064 , -0.268329 , -0.203344 , -0.143543 , -0.0822519 , -0.010345 , 0.0842602 , 0.213941 , 0.384972 , 0.591545 , 0.816283 , 1.03368 , 1.21508 , 1.34182 , 1.41418 , 1.4459 , 1.45428 , 1.45359 , 1.45215 , 1.45246 , 1.4534 , 1.45296 , 1.4499 , 1.4443 , 1.43734 , 1.4307 , 1.42569 , 1.42185 , 1.41541 , 1.39838 , 1.35858 , 1.281 , 1.15267 , 0.972522 , 0.757595 , 0.532966 , 0.319911 , 0.136773 , -0.00431803 , -0.104786 , -0.177237 , -0.237376 , -0.298266 , -0.368331 , -0.452441 , -0.55561 , -0.688473 , -0.872028 , -1.13361 , -1.47283 , -1.8177 , -2.08961 , -2.28324 , -2.4305 , -2.56076 , -2.69191 , -2.8325 , -2.98977 , 3.10049 , 2.83295 , 2.46135 , 2.05703 , 1.74519 , 1.54412 , 1.41393 , 1.32385 , 1.25895 , 1.21294 , 1.18239 , 1.16421 , 1.15533 , 1.15314 , 1.15587 , 1.16268 , 1.1736 , 1.18946 , 1.21207 , 1.24442 , 1.29095 , 1.35795 , 1.45471 , 1.59622 , 1.80714 , 2.11675 , 2.51297 , 2.89641 , -3.09674 , -2.89079 , -2.73271 , -2.59605 , -2.4652 , -2.32877 , -2.17295 , -1.97721 , -1.71755 , -1.39136 , -1.05305 , -1.83529 , -1.7019 , -1.55268 , -1.38256 , -1.19071 , -0.973135 , -0.721308 , -0.449455 , -0.213273 , -0.0531903 , 0.04099 , 0.0965565 , 0.129183 , 0.137534 , 0.114436 , 0.0597056 , -0.021024 , -0.124426 , -0.251244 , -0.398738 , -0.554993 , -0.708304 , -0.855488 , -0.99278 , -1.10374 , -1.16109 , -1.14332 , -1.04972 , -0.903431 , -0.741278 , -0.590911 , -0.455306 , -0.322924 , -0.19222 , -0.0789652 , 0.000711378 , 0.0419622 , 0.0494281 , 0.0334576 , 0.00253603 , -0.0487715 , -0.14356 , -0.31322 , -0.567596 , -0.854914 , -1.10791 , -1.31835 , -1.50525 , -1.67507 , -1.82541 , -1.95907 , -2.08619 , -2.21822 , -2.36346 , -2.52796 , -2.7185 , -2.938 , 3.11098 , 2.8913 , 2.70088 , 2.52469 , 2.33294 , 2.09823 , 1.78991 , 1.34794 , 0.768105 , 0.28318 , 0.00177029 , -0.156965 , -0.257208 , -0.326723 , -0.375364 , -0.406973 , -0.424501 , -0.430979 , -0.428957 , -0.419885 , -0.403847 , -0.379488 , -0.343951 , -0.292598 , -0.218163 , -0.108197 , 0.061865 , 0.343392 , 0.812715 , 1.40987 , 1.88341 , 2.19545 , 2.42511 , 2.62361 , 2.81731 , 3.0206 , -3.0451 , -2.82063 , -2.60452 , -2.40989 , -2.24099 , -2.09439 , -1.96246 , -2.55873 , -2.13 , -1.71618 , -1.4266 , -1.22596 , -1.0634 , -0.929013 , -0.834902 , -0.777332 , -0.727883 , -0.657357 , -0.565256 , -0.471571 , -0.374175 , -0.246286 , -0.0883696 , 0.054769 , 0.159129 , 0.236696 , 0.297078 , 0.337704 , 0.359789 , 0.373302 , 0.386799 , 0.400461 , 0.407721 , 0.401781 , 0.381311 , 0.350793 , 0.315865 , 0.278142 , 0.233078 , 0.172608 , 0.0884188 , -0.0292422 , -0.183095 , -0.343177 , -0.467813 , -0.565638 , -0.669009 , -0.775756 , -0.854772 , -0.902491 , -0.957819 , -1.06107 , -1.21662 , -1.39487 , -1.58523 , -1.83826 , -2.22871 , -2.68561 , -3.01924 , 3.06168 , 2.91027 , 2.74211 , 2.51902 , 2.25476 , 1.99628 , 1.7555 , 1.51671 , 1.29242 , 1.10808 , 0.950149 , 0.774322 , 0.534612 , 0.172066 , -0.436585 , -1.1529 , -1.56356 , -1.75816 , -1.86535 , -1.93332 , -1.9768 , -2.00088 , -2.00955 , -2.00677 , -1.99535 , -1.97597 , -1.9464 , -1.90078 , -1.82822 , -1.70891 , -1.5003 , -1.09809 , -0.389966 , 0.26185 , 0.629407 , 0.86142 , 1.04818 , 1.22829 , 1.42261 , 1.64175 , 1.88475 , 2.14339 , 2.40576 , 2.6522 , 2.86311 , 3.03885 , -3.07691 , -2.86917 , -2.99942 , -2.78074 , -2.59047 , -2.4249 , -2.27128 , -2.12736 , -1.98853 , -1.81538 , -1.55537 , -1.24659 , -0.980486 , -0.740185 , -0.517874 , -0.370666 , -0.289312 , -0.215647 , -0.127801 , -0.0491237 , 0.0226203 , 0.128231 , 0.288992 , 0.472322 , 0.668456 , 0.900054 , 1.107 , 1.18337 , 1.09674 , 0.88445 , 0.635603 , 0.419452 , 0.23094 , 0.0644143 , -0.061253 , -0.151797 , -0.230349 , -0.309117 , -0.388206 , -0.487591 , -0.647009 , -0.867564 , -1.09854 , -1.3661 , -1.69756 , -1.9764 , -2.14792 , -2.28518 , -2.44048 , -2.60327 , -2.76149 , -2.93924 , 3.11936 , 2.86945 , 2.64003 , 2.41401 , 2.1304 , 1.79167 , 1.46056 , 1.10897 , 0.687897 , 0.306722 , 0.0235483 , -0.228782 , -0.457936 , -0.634651 , -0.79451 , -1.02103 , -1.40666 , -2.06709 , -2.81223 , 3.0708 , 2.89374 , 2.79979 , 2.73896 , 2.70003 , 2.682 , 2.68272 , 2.69844 , 2.72696 , 2.77038 , 2.83678 , 2.94379 , 3.13307 , -2.76351 , -2.01084 , -1.30811 , -0.933402 , -0.702036 , -0.516019 , -0.327281 , -0.10564 , 0.160664 , 0.477519 , 0.851904 , 1.24832 , 1.61265 , 1.95001 , 2.27731 , 2.57263 , 2.82296 , 3.05341 , 2.44571 , 2.67669 , 2.94448 , -3.01857 , -2.68873 , -2.36638 , -2.03759 , -1.78297 , -1.63864 , -1.52151 , -1.39253 , -1.30118 , -1.2429 , -1.14026 , -0.995739 , -0.883017 , -0.771981 , -0.62147 , -0.51169 , -0.479076 , -0.474528 , -0.462095 , -0.444716 , -0.427548 , -0.409285 , -0.401828 , -0.419349 , -0.451446 , -0.478105 , -0.497448 , -0.524369 , -0.563307 , -0.618848 , -0.72105 , -0.864879 , -0.99518 , -1.12383 , -1.26465 , -1.36703 , -1.44104 , -1.5473 , -1.67819 , -1.79144 , -1.94356 , -2.2248 , -2.57226 , -2.88134 , 3.08391 , 2.75827 , 2.48543 , 2.25383 , 2.02883 , 1.79462 , 1.53586 , 1.25873 , 0.996613 , 0.730694 , 0.40973 , 0.0601888 , -0.315648 , -0.841975 , -1.3798 , -1.74937 , -2.03393 , -2.25344 , -2.43306 , -2.67366 , -3.12607 , 2.35686 , 1.66206 , 1.34907 , 1.20892 , 1.13253 , 1.08755 , 1.06679 , 1.06597 , 1.08099 , 1.11258 , 1.16788 , 1.25915 , 1.41238 , 1.71625 , 2.41321 , -3.02891 , -2.58818 , -2.33107 , -2.12826 , -1.91368 , -1.62485 , -1.21133 , -0.674055 , -0.14957 , 0.249741 , 0.574623 , 0.87923 , 1.17156 , 1.45269 , 1.72404 , 1.98008 , 2.21884 , 1.78838 , 2.15483 , 2.58209 , 2.93109 , -3.11766 , -2.92498 , -2.74376 , -2.57604 , -2.37807 , -2.13523 , -1.88998 , -1.6009 , -1.31208 , -1.13065 , -0.981245 , -0.836013 , -0.750251 , -0.68243 , -0.57905 , -0.486569 , -0.434868 , -0.376475 , -0.259293 , -0.0945544 , 0.048005 , 0.0894652 , 0.0110409 , -0.13312 , -0.27994 , -0.406081 , -0.504005 , -0.580112 , -0.672077 , -0.786606 , -0.87828 , -0.965451 , -1.10471 , -1.27115 , -1.46897 , -1.75697 , -2.04003 , -2.29422 , -2.5612 , -2.76236 , -2.92578 , -3.13074 , 2.93832 , 2.72256 , 2.39298 , 1.93357 , 1.54772 , 1.27325 , 0.994161 , 0.645944 , 0.312394 , 0.0129396 , -0.327149 , -0.649723 , -0.910275 , -1.20644 , -1.56512 , -2.01454 , -2.68034 , 3.05432 , 2.7005 , 2.42817 , 2.2144 , 1.93725 , 1.32578 , 0.367235 , -0.172794 , -0.391567 , -0.487371 , -0.533445 , -0.553402 , -0.554231 , -0.539162 , -0.507063 , -0.447399 , -0.335265 , -0.111015 , 0.425307 , 1.42563 , 2.02277 , 2.31828 , 2.55425 , 2.81821 , -3.09381 , -2.51606 , -1.85798 , -1.37238 , -1.02306 , -0.736781 , -0.457903 , -0.142708 , 0.194118 , 0.525948 , 0.866911 , 1.19985 , 1.49411 }; 

/*
    for (int i = 0; i < numberofexcitationangles*numberofobservationangles; i++) {
        cout << "D[" << i << " ]: " << D[i] << endl;
    }
    */

    float del_Phi = 0;
    float fit = 0;
	for(int i = 0; i < numberofobservationangles*numberofexcitationangles*numberoffrequencies; i++)
	{
		fit -= pow(D[i]-measurement[i],2)/(numberofexcitationangles*numberoffrequencies);
  /*      del_Phi = P[i]-measurement[i+numberofobservationangles*numberofexcitationangles*numberoffrequencies];
        if(del_Phi>PI)
        {
            del_Phi -= 2*PI;
        }
        else if(del_Phi<-1*PI)
        {
            del_Phi += 2*PI;
        }
        fit -= del_Phi*del_Phi/(10*PI*PI*numberofexcitationangles*numberoffrequencies);
*/
	//	if(abs(D[index]-measurement[index])>0.01)
		//	cout<<index<<" D = "<<D[index]<<" measurement = "<<measurement[index]<<endl;
	}


    error = cudaGetLastError();
    free(Ceze);
    free(Cezhy);
    free(Ez);
    free(eps_infinity);
    free(del_eps);
    free(sigma_e_z);
    free(Beta_p);
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
    free(P);

    free(hcjzxp);
    free(hcjzyp);
    free(hcjzxn);
    free(hcjzyn);
    free(hcmxyp);
    free(hcmyxp);
    free(hcmxyn);
    free(hcmyxn);

    cudaCheck( cudaFree(dev_Cezeic));
    cudaCheck( cudaFree(dev_Cezeip));
    cudaCheck( cudaFree(dev_eps_infinity));
    cudaCheck( cudaFree(dev_sigma_e_z));
    cudaCheck( cudaFree(dev_freq));
    cudaCheck( cudaFree(dev_Phi));
    cudaCheck( cudaFree(dev_i));
    cudaCheck( cudaFree(dev_Beta_p));
    cudaCheck( cudaFree(dev_Cezjp));

    cudaCheck( cudaFree(cjzxp));
    cudaCheck( cudaFree(cjzyp));
    cudaCheck( cudaFree(cjzxn));
    cudaCheck( cudaFree(cjzyn));
    cudaCheck( cudaFree(cmxyp));
    cudaCheck( cudaFree(cmyxp));
    cudaCheck( cudaFree(cmxyn));
    cudaCheck( cudaFree(cmyxn));
/* float *dev_freq, *dev_Phi;
    float *dev_Ceze,*dev_Cezhy, *dev_bex, *dev_aex, *dev_bmx, *dev_amx, *dev_kex, *dev_kmx;//dev_Cezj if using loop current source
    float *dev_Ez, *dev_Hy, *dev_Hx;

    float *dev_Psi_ezy, *dev_Psi_ezx, *dev_Psi_hyx, *dev_Psi_hxy;

    float *dev_Cezeic, *dev_Cezeip;
    float  *dev_eps_infinity,*dev_sigma_e_z;
    float *dev_Beta_p;
    float *dev_Cezjp,*dev_Jp;//dev_Jp is the polarization current term used in the Debye scattering (FD)2TD*/  

    cudaCheck( cudaFree(dev_Cezhy));
    cudaCheck( cudaFree(dev_Ceze));

    cudaCheck( cudaFree(dev_bex));
    cudaCheck( cudaFree(dev_aex));
    cudaCheck( cudaFree(dev_bmx));
    cudaCheck( cudaFree(dev_amx));
    cudaCheck( cudaFree(dev_kex));
    cudaCheck( cudaFree(dev_kmx));
    cudaCheck( cudaFree(dev_Ez));
    cudaCheck( cudaFree(dev_Jp));
    cudaCheck( cudaFree(dev_Hy));
    cudaCheck( cudaFree(dev_Hx));
    cudaCheck( cudaFree(dev_Psi_ezy));
    cudaCheck( cudaFree(dev_Psi_ezx));
    cudaCheck( cudaFree(dev_Psi_hyx));
    cudaCheck( cudaFree(dev_Psi_hxy));

    cout << "fitness is: " << fit << endl;
    return (double)fit;
}

__global__ void scattered_parameter_init(float *eps_infinity,float *sigma_e_z ,float *Cezeic, float *Cezeip, float *Cezjp,float *dev_Beta_p)
{
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    int y=threadIdx.y+blockDim.y*blockIdx.y;
    if(x<(nx+1)&&y<(ny+1))
    {
        Cezeic[dgetCell(x,y,nx+1)] = (-2*eps0*eps_infinity[dgetCell(x,y,nx+1)]+2*eps0 - sigma_e_z[dgetCell(x,y,nx+1)]*dt - dev_Beta_p[dgetCell(x,y,nx+1)])/(2*eps0*eps_infinity[dgetCell(x,y,nx+1)] + sigma_e_z[dgetCell(x,y,nx+1)]*dt + dev_Beta_p[dgetCell(x,y,nx+1)]);//I think it should be sigma_e_z*dt not *dx
        Cezeip[dgetCell(x,y,nx+1)] = (2*eps0*eps_infinity[dgetCell(x,y,nx+1)] - 2*eps0 - sigma_e_z[dgetCell(x,y,nx+1)]*dt + dev_Beta_p[dgetCell(x,y,nx+1)])/(2*eps0*eps_infinity[dgetCell(x,y,nx+1)] + sigma_e_z[dgetCell(x,y,nx+1)]*dt + dev_Beta_p[dgetCell(x,y,nx+1)]); 
        Cezjp[dgetCell(x,y,nx+1)] = ((1+Kp)*dt)/(2*eps0*eps_infinity[dgetCell(x,y,nx+1)] + sigma_e_z[dgetCell(x,y,nx+1)]*dt + dev_Beta_p[dgetCell(x,y,nx+1)]);

    }
}
