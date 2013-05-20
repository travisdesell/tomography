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
    /*	if(lerp == numpatches*numpatches/2){
        image.push_back(10);
    	}
       else
         {
         image.push_back(10);  //eps_infinity of fat
         }*/
    }

    for (int lerp = numpatches*numpatches; lerp < numpatches*numpatches* 2; lerp++) {
        image.push_back((float)arguments.at(lerp));
	
      /* if(lerp == numpatches*numpatches/2 + numpatches*numpatches){
       image.push_back(10);// del_eps 
	
        }
       else
       {
           image.push_back(10);
       }
*/
    }

    for (int lerp = numpatches*numpatches*2 ; lerp<numpatches*numpatches*3;lerp++){
        image.push_back((float)arguments.at(lerp));
	
      /* if(lerp == numpatches*numpatches/2 + numpatches*numpatches){
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
	ofstream measurement_data;
	measurement_data.open("Measurement_Phase.txt");
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
	

	for(int index = 0;index < numberofobservationangles * numberoffrequencies * numberofexcitationangles ; index++)
	{
            measurement_data<<D[index]<<" , ";
    }		
    for(int index = 0;index < numberofobservationangles * numberoffrequencies * numberofexcitationangles ; index++)
	{
            measurement_data<<P[index]<<" , ";
    }
  float measurement[] = {   0.104819 , 0.104614 , 0.103996 , 0.102968 , 0.101538 , 0.0997189 , 0.0975263 , 0.0949827 , 0.0921143 , 0.0889521 , 0.0855306 , 0.0818875 , 0.0780628 , 0.0740974 , 0.0700327 , 0.0659094 , 0.061767 , 0.0576431 , 0.0535728 , 0.0495891 , 0.0457224 , 0.0420006 , 0.0384487 , 0.0350892 , 0.0319415 , 0.0290221 , 0.0263436 , 0.0239152 , 0.0217419 , 0.0198251 , 0.0181617 , 0.0167451 , 0.0155651 , 0.0146086 , 0.0138599 , 0.0133018 , 0.0129153 , 0.0126809 , 0.0125784 , 0.0125874 , 0.0126875 , 0.0128583 , 0.0130799 , 0.0133328 , 0.0135982 , 0.0138586 , 0.014098 , 0.0143024 , 0.0144605 , 0.0145637 , 0.0146067 , 0.0145876 , 0.0145082 , 0.0143731 , 0.0141906 , 0.0139713 , 0.0137284 , 0.0134772 , 0.0132344 , 0.0130181 , 0.0128476 , 0.012743 , 0.0127253 , 0.0128161 , 0.013037 , 0.0134102 , 0.0139571 , 0.0146982 , 0.0156527 , 0.0168375 , 0.0182664 , 0.0199504 , 0.0218961 , 0.0241062 , 0.0265788 , 0.0293078 , 0.0322827 , 0.0354889 , 0.0389081 , 0.0425181 , 0.0462939 , 0.0502072 , 0.0542271 , 0.0583206 , 0.0624523 , 0.0665855 , 0.0706822 , 0.0747036 , 0.078611 , 0.0823664 , 0.0859326 , 0.0892746 , 0.0923592 , 0.0951565 , 0.0976391 , 0.0997834 , 0.101568 , 0.102977 , 0.103995 , 0.104612 , 0.238212 , 0.236988 , 0.234077 , 0.22957 , 0.223611 , 0.216394 , 0.208142 , 0.199089 , 0.189455 , 0.179426 , 0.169143 , 0.158695 , 0.148136 , 0.1375 , 0.126825 , 0.116174 , 0.105642 , 0.0953539 , 0.0854546 , 0.076088 , 0.0673798 , 0.0594212 , 0.0522627 , 0.0459132 , 0.0403457 , 0.035508 , 0.0313335 , 0.0277532 , 0.0247053 , 0.0221433 , 0.0200406 , 0.018391 , 0.0172046 , 0.0164998 , 0.0162913 , 0.0165779 , 0.0173331 , 0.0185014 , 0.0200003 , 0.0217301 , 0.0235865 , 0.0254719 , 0.0273053 , 0.0290254 , 0.0305901 , 0.0319718 , 0.0331521 , 0.0341168 , 0.0348535 , 0.0353496 , 0.0355931 , 0.0355733 , 0.0352815 , 0.034712 , 0.0338627 , 0.0327366 , 0.0313452 , 0.029713 , 0.0278817 , 0.0259134 , 0.023891 , 0.0219137 , 0.0200895 , 0.0185238 , 0.0173086 , 0.0165139 , 0.016184 , 0.01634 , 0.016986 , 0.018119 , 0.0197389 , 0.0218546 , 0.0244878 , 0.027672 , 0.031448 , 0.035858 , 0.0409392 , 0.0467174 , 0.0532025 , 0.0603869 , 0.0682443 , 0.0767322 , 0.0857963 , 0.0953742 , 0.105401 , 0.115812 , 0.126542 , 0.13752 , 0.148664 , 0.159872 , 0.171016 , 0.181935 , 0.192445 , 0.202341 , 0.21141 , 0.219442 , 0.226244 , 0.231643 , 0.2355 , 0.23771 , 0.202687 , 0.199828 , 0.193079 , 0.183013 , 0.170426 , 0.156218 , 0.141276 , 0.126357 , 0.112045 , 0.0987673 , 0.0868676 , 0.0766703 , 0.0684892 , 0.0625601 , 0.0589324 , 0.0573829 , 0.0574089 , 0.0583209 , 0.059398 , 0.0600408 , 0.0598646 , 0.0587086 , 0.0565807 , 0.0535821 , 0.0498489 , 0.0455297 , 0.0407933 , 0.0358502 , 0.0309581 , 0.0264036 , 0.0224576 , 0.0193189 , 0.0170745 , 0.0156922 , 0.015052 , 0.015 , 0.0153999 , 0.016161 , 0.0172356 , 0.018592 , 0.0201817 , 0.0219184 , 0.0236824 , 0.0253469 , 0.0268122 , 0.0280272 , 0.0289899 , 0.0297297 , 0.0302843 , 0.0306794 , 0.0309168 , 0.0309738 , 0.0308081 , 0.0303695 , 0.0296165 , 0.0285322 , 0.0271392 , 0.0255037 , 0.0237289 , 0.0219355 , 0.020236 , 0.0187131 , 0.017414 , 0.0163634 , 0.0155922 , 0.0151664 , 0.0152006 , 0.0158449 , 0.0172477 , 0.0195088 , 0.0226399 , 0.0265492 , 0.0310524 , 0.0359036 , 0.0408327 , 0.0455762 , 0.0498956 , 0.0535881 , 0.0564923 , 0.0585043 , 0.0596016 , 0.0598729 , 0.0595407 , 0.0589632 , 0.0586051 , 0.0589816 , 0.0605935 , 0.0638724 , 0.069148 , 0.0766347 , 0.0864161 , 0.0984189 , 0.11237 , 0.127765 , 0.143863 , 0.159732 , 0.174347 , 0.186702 , 0.195925 , 0.201373 , 0.463376 , 0.454541 , 0.431401 , 0.396207 , 0.351872 , 0.301524 , 0.248266 , 0.195186 , 0.145418 , 0.10198 , 0.0672921 , 0.042581 , 0.0275894 , 0.0208502 , 0.0203782 , 0.0243355 , 0.0312872 , 0.0400338 , 0.0493241 , 0.0577832 , 0.0641364 , 0.0675402 , 0.0677749 , 0.0651941 , 0.0605097 , 0.0545575 , 0.0481298 , 0.0418808 , 0.0362711 , 0.0315413 , 0.0277358 , 0.024785 , 0.0226166 , 0.0212288 , 0.0206727 , 0.0209498 , 0.0219066 , 0.0232334 , 0.0245957 , 0.0257943 , 0.0268181 , 0.0277613 , 0.0286925 , 0.0295765 , 0.0302899 , 0.0307057 , 0.0307839 , 0.0306056 , 0.0303322 , 0.030127 , 0.0300856 , 0.0302114 , 0.030429 , 0.0306237 , 0.03069 , 0.0305672 , 0.0302542 , 0.0297874 , 0.0291971 , 0.028467 , 0.0275305 , 0.0263207 , 0.0248487 , 0.023261 , 0.0218266 , 0.020853 , 0.0205838 , 0.0211432 , 0.0225605 , 0.0248392 , 0.0280049 , 0.0320894 , 0.0370627 , 0.0427616 , 0.0488578 , 0.0548744 , 0.0602297 , 0.0642897 , 0.0664352 , 0.0661546 , 0.0631728 , 0.0575737 , 0.0498683 , 0.0409675 , 0.0320908 , 0.0246781 , 0.0203521 , 0.0208848 , 0.0280816 , 0.0435229 , 0.0682226 , 0.102335 , 0.145027 , 0.194522 , 0.248234 , 0.302919 , 0.354863 , 0.400161 , 0.435128 , 0.456794 , 0.742575 , 0.725617 , 0.681557 , 0.615392 , 0.532975 , 0.440621 , 0.34554 , 0.255903 , 0.179349 , 0.120459 , 0.0793367 , 0.0528638 , 0.0375026 , 0.0309549 , 0.0314943 , 0.0365855 , 0.0429384 , 0.0480237 , 0.051227 , 0.0533598 , 0.0551943 , 0.0565886 , 0.0568429 , 0.0555887 , 0.0532276 , 0.0506225 , 0.048453 , 0.0467946 , 0.0451993 , 0.0431609 , 0.0406082 , 0.0380484 , 0.0362006 , 0.0353897 , 0.0353201 , 0.0355065 , 0.0358454 , 0.0366588 , 0.0382005 , 0.040159 , 0.0417167 , 0.0422245 , 0.0417859 , 0.0410879 , 0.0407412 , 0.0408401 , 0.0410284 , 0.0408895 , 0.0403102 , 0.0395426 , 0.0389718 , 0.0388415 , 0.0391384 , 0.0396772 , 0.0402802 , 0.0409036 , 0.0415974 , 0.0423221 , 0.0428073 , 0.0426508 , 0.0416456 , 0.0400371 , 0.0384056 , 0.0372234 , 0.0365193 , 0.0360073 , 0.0355473 , 0.0354235 , 0.0361316 , 0.0379044 , 0.0404666 , 0.0432372 , 0.0457439 , 0.0478653 , 0.0497484 , 0.0515478 , 0.053232 , 0.0545866 , 0.0553674 , 0.0554581 , 0.0548699 , 0.0535567 , 0.0512455 , 0.0475778 , 0.0425884 , 0.0371424 , 0.0329605 , 0.0323169 , 0.0378367 , 0.0525486 , 0.0797952 , 0.122532 , 0.182191 , 0.257811 , 0.345877 , 0.440692 , 0.534868 , 0.619922 , 0.687248 , 0.729586 , 0.90487 , 0.876516 , 0.802103 , 0.692223 , 0.559838 , 0.420847 , 0.293393 , 0.192518 , 0.123638 , 0.0823636 , 0.0613645 , 0.0556471 , 0.0605576 , 0.0685083 , 0.0720389 , 0.0694325 , 0.0642459 , 0.059601 , 0.0553977 , 0.0509176 , 0.0473689 , 0.0465873 , 0.0486184 , 0.0518087 , 0.0546823 , 0.0568105 , 0.0580655 , 0.0579149 , 0.0559489 , 0.0528544 , 0.0501661 , 0.0486987 , 0.047802 , 0.0466486 , 0.0456973 , 0.0462291 , 0.0483873 , 0.0504318 , 0.0508596 , 0.050367 , 0.050657 , 0.0520792 , 0.0531207 , 0.052264 , 0.0498947 , 0.047772 , 0.0471332 , 0.0478223 , 0.0488157 , 0.0492223 , 0.0488409 , 0.0479909 , 0.0471136 , 0.0466477 , 0.0470469 , 0.0485528 , 0.050746 , 0.0525295 , 0.0529774 , 0.0523324 , 0.0517818 , 0.0519549 , 0.0519997 , 0.0506856 , 0.0482408 , 0.0463634 , 0.0462647 , 0.0473944 , 0.048489 , 0.049318 , 0.0507417 , 0.0532407 , 0.056046 , 0.0578708 , 0.0580663 , 0.0568415 , 0.0546755 , 0.0519324 , 0.0491014 , 0.0471151 , 0.0470576 , 0.049461 , 0.054054 , 0.060103 , 0.0664943 , 0.0713795 , 0.0725451 , 0.0690582 , 0.0627529 , 0.0581715 , 0.0617456 , 0.0812249 , 0.124435 , 0.196589 , 0.29777 , 0.422321 , 0.559454 , 0.693798 , 0.806856 , 0.881001 , 1.16104 , 1.11758 , 0.993794 , 0.8102 , 0.597988 , 0.395838 , 0.236787 , 0.132675 , 0.0758291 , 0.0553774 , 0.0624303 , 0.0816766 , 0.0948041 , 0.0959068 , 0.0911254 , 0.0836515 , 0.0719715 , 0.0597661 , 0.0540568 , 0.0549146 , 0.0563728 , 0.0559155 , 0.0563442 , 0.059683 , 0.0639892 , 0.065822 , 0.0636437 , 0.0591387 , 0.0558897 , 0.0559023 , 0.0574897 , 0.0577719 , 0.0566768 , 0.0567072 , 0.0583218 , 0.0587979 , 0.0575179 , 0.057461 , 0.0600143 , 0.0620267 , 0.0606041 , 0.0580033 , 0.0578293 , 0.0597374 , 0.0603731 , 0.0583039 , 0.0557337 , 0.0552233 , 0.0569757 , 0.0592087 , 0.0600358 , 0.0589146 , 0.0567558 , 0.055186 , 0.0554637 , 0.0573619 , 0.0591586 , 0.0595206 , 0.0593 , 0.0602332 , 0.0618909 , 0.0619358 , 0.0599841 , 0.0586206 , 0.0592567 , 0.0598044 , 0.0584007 , 0.0566844 , 0.0569036 , 0.0581518 , 0.0581567 , 0.0570595 , 0.0570202 , 0.0590044 , 0.0617213 , 0.0633352 , 0.0630607 , 0.0613136 , 0.0590551 , 0.0568946 , 0.0547696 , 0.0531463 , 0.0542456 , 0.0605025 , 0.0715419 , 0.0840722 , 0.0944291 , 0.0992592 , 0.0951443 , 0.081556 , 0.0647125 , 0.0571212 , 0.0745347 , 0.133006 , 0.243547 , 0.405124 , 0.602212 , 0.80819 , 0.99035 , 1.11596 , 1.50208 , 1.43539 , 1.23965 , 0.953145 , 0.642433 , 0.378397 , 0.197034 , 0.0946734 , 0.0578502 , 0.0692671 , 0.0953763 , 0.108721 , 0.109899 , 0.103971 , 0.0879849 , 0.0707587 , 0.0654653 , 0.0678158 , 0.0688921 , 0.0705153 , 0.0734518 , 0.0723175 , 0.067539 , 0.0656304 , 0.0676436 , 0.0681677 , 0.0651942 , 0.0632635 , 0.065362 , 0.0680002 , 0.0674922 , 0.0660797 , 0.0664296 , 0.0658299 , 0.0635249 , 0.0640787 , 0.0679559 , 0.0687186 , 0.0657842 , 0.065884 , 0.0695343 , 0.0699638 , 0.0659793 , 0.0636725 , 0.0651979 , 0.0664331 , 0.0645719 , 0.0621675 , 0.0625162 , 0.0654641 , 0.0679906 , 0.0675483 , 0.0646701 , 0.0622444 , 0.0623728 , 0.0640609 , 0.0649939 , 0.0654656 , 0.0675679 , 0.0701934 , 0.0700123 , 0.068062 , 0.0682817 , 0.0695246 , 0.0678366 , 0.0650883 , 0.0654703 , 0.0671922 , 0.0668365 , 0.0662061 , 0.0675368 , 0.0683501 , 0.0664863 , 0.0642176 , 0.0640791 , 0.0652358 , 0.065995 , 0.0669915 , 0.0695838 , 0.0724559 , 0.0728583 , 0.070643 , 0.06776 , 0.065404 , 0.0658755 , 0.0735109 , 0.0885386 , 0.104584 , 0.114259 , 0.111947 , 0.0951329 , 0.0713648 , 0.0616063 , 0.0944252 , 0.197466 , 0.386268 , 0.65111 , 0.952648 , 1.23191 , 1.4291 , 1.84018 , 1.74317 , 1.44794 , 1.03617 , 0.632021 , 0.328263 , 0.14545 , 0.0711381 , 0.0782992 , 0.109987 , 0.12519 , 0.12552 , 0.110352 , 0.0839893 , 0.0709023 , 0.0720937 , 0.0746317 , 0.0820961 , 0.0890582 , 0.0839876 , 0.0753271 , 0.0727177 , 0.072294 , 0.0719126 , 0.0733283 , 0.0749565 , 0.0758877 , 0.0769318 , 0.075939 , 0.0727196 , 0.0722412 , 0.0737746 , 0.0722154 , 0.0711129 , 0.0745689 , 0.0751401 , 0.071603 , 0.0736023 , 0.078416 , 0.0759638 , 0.0721318 , 0.074914 , 0.0777185 , 0.0745278 , 0.071349 , 0.0724268 , 0.0738471 , 0.0723925 , 0.0707564 , 0.0718416 , 0.0743892 , 0.0749994 , 0.0730831 , 0.0714607 , 0.0716286 , 0.0718693 , 0.0718581 , 0.0740493 , 0.0771473 , 0.0768899 , 0.0755401 , 0.0773902 , 0.0782439 , 0.0753109 , 0.0746273 , 0.0764109 , 0.0746909 , 0.0723554 , 0.0738399 , 0.074634 , 0.0728219 , 0.0730277 , 0.0754667 , 0.0763521 , 0.0755405 , 0.0745426 , 0.0730652 , 0.0714747 , 0.0712782 , 0.0730759 , 0.077412 , 0.0833409 , 0.0856876 , 0.0816904 , 0.0755601 , 0.0714654 , 0.0737592 , 0.0880403 , 0.11051 , 0.128776 , 0.130915 , 0.110388 , 0.0793981 , 0.0744501 , 0.144272 , 0.332567 , 0.648356 , 1.04419 , 1.43353 , 1.72392 , 2.22451 , 2.0802 , 1.64756 , 1.08626 , 0.590325 , 0.256447 , 0.0958056 , 0.0800324 , 0.122043 , 0.149267 , 0.152691 , 0.124939 , 0.0882726 , 0.0765312 , 0.0758677 , 0.0838496 , 0.0974173 , 0.0959883 , 0.0900877 , 0.0856552 , 0.0761815 , 0.075136 , 0.0833627 , 0.0842102 , 0.0788918 , 0.0793759 , 0.0839682 , 0.0824741 , 0.07639 , 0.0769389 , 0.0815843 , 0.0814701 , 0.0817016 , 0.083366 , 0.0791943 , 0.0788179 , 0.084976 , 0.0828259 , 0.0791061 , 0.0843445 , 0.0855466 , 0.0801955 , 0.0809385 , 0.0848259 , 0.0830384 , 0.079616 , 0.0797865 , 0.0810259 , 0.0801724 , 0.0788004 , 0.0788101 , 0.0796894 , 0.0804791 , 0.0812099 , 0.080865 , 0.079207 , 0.0797755 , 0.0828641 , 0.0833102 , 0.0828823 , 0.0856839 , 0.0857589 , 0.0829728 , 0.0845119 , 0.0847687 , 0.0813344 , 0.0824795 , 0.0841113 , 0.0818892 , 0.0821086 , 0.082092 , 0.0778376 , 0.0762846 , 0.0797379 , 0.0823108 , 0.082182 , 0.0823233 , 0.0827559 , 0.0803041 , 0.0765305 , 0.0777959 , 0.0840151 , 0.0899586 , 0.0946948 , 0.0945085 , 0.0862906 , 0.0775193 , 0.0770317 , 0.0930198 , 0.125218 , 0.153653 , 0.155246 , 0.122699 , 0.0822819 , 0.0999565 , 0.255925 , 0.600243 , 1.10012 , 1.63394 , 2.05111 , 0.0294352 , 0.032419 , 0.0356298 , 0.0390485 , 0.0426525 , 0.0464146 , 0.0503048 , 0.0542894 , 0.0583328 , 0.0623975 , 0.0664453 , 0.0704374 , 0.0743361 , 0.0781042 , 0.0817064 , 0.0851095 , 0.0882824 , 0.0911968 , 0.0938271 , 0.0961504 , 0.0981474 , 0.0998013 , 0.101099 , 0.102031 , 0.102591 , 0.102777 , 0.102588 , 0.102029 , 0.101106 , 0.0998294 , 0.0982097 , 0.0962607 , 0.0939978 , 0.0914384 , 0.0886013 , 0.0855077 , 0.0821812 , 0.0786484 , 0.0749381 , 0.0710825 , 0.0671168 , 0.063078 , 0.0590051 , 0.054938 , 0.0509166 , 0.0469792 , 0.0431624 , 0.0394998 , 0.036021 , 0.0327514 , 0.0297118 , 0.0269185 , 0.024383 , 0.0221119 , 0.0201078 , 0.018369 , 0.0168899 , 0.0156615 , 0.0146716 , 0.0139049 , 0.0133439 , 0.0129688 , 0.0127582 , 0.0126894 , 0.012739 , 0.0128833 , 0.0130987 , 0.0133623 , 0.013652 , 0.0139475 , 0.0142299 , 0.0144829 , 0.0146922 , 0.0148466 , 0.0149379 , 0.014961 , 0.0149143 , 0.0147997 , 0.0146226 , 0.0143916 , 0.0141187 , 0.0138187 , 0.0135092 , 0.0132096 , 0.012941 , 0.0127255 , 0.0125862 , 0.0125458 , 0.012627 , 0.0128517 , 0.0132408 , 0.0138139 , 0.0145887 , 0.0155814 , 0.0168055 , 0.0182727 , 0.0199913 , 0.0219668 , 0.0242012 , 0.0266926 , 0.0362178 , 0.0413641 , 0.04714 , 0.0535473 , 0.0605792 , 0.0682209 , 0.0764489 , 0.0852336 , 0.0945401 , 0.104329 , 0.114554 , 0.125158 , 0.136066 , 0.147183 , 0.158383 , 0.169513 , 0.180391 , 0.190818 , 0.200587 , 0.209497 , 0.217368 , 0.224049 , 0.229423 , 0.233412 , 0.235966 , 0.237059 , 0.236678 , 0.234817 , 0.231477 , 0.226663 , 0.220404 , 0.212759 , 0.203839 , 0.193811 , 0.182902 , 0.171386 , 0.159559 , 0.147708 , 0.136083 , 0.124873 , 0.114192 , 0.104093 , 0.0945693 , 0.085589 , 0.0771108 , 0.0691054 , 0.0615642 , 0.0545022 , 0.0479524 , 0.0419576 , 0.0365603 , 0.0317949 , 0.0276832 , 0.0242324 , 0.0214356 , 0.0192743 , 0.0177202 , 0.0167383 , 0.0162885 , 0.0163268 , 0.016806 , 0.0176755 , 0.0188808 , 0.0203626 , 0.0220572 , 0.0238962 , 0.025809 , 0.0277236 , 0.0295707 , 0.0312854 , 0.0328104 , 0.0340967 , 0.0351059 , 0.0358097 , 0.0361907 , 0.0362422 , 0.0359689 , 0.0353856 , 0.034516 , 0.0333925 , 0.0320523 , 0.0305364 , 0.0288872 , 0.0271488 , 0.0253669 , 0.0235899 , 0.0218718 , 0.0202727 , 0.0188589 , 0.0177011 , 0.0168699 , 0.0164319 , 0.0164442 , 0.0169524 , 0.0179895 , 0.0195777 , 0.0217315 , 0.0244617 , 0.0277785 , 0.0316931 , 0.0458778 , 0.050203 , 0.053764 , 0.056397 , 0.0580507 , 0.0587931 , 0.0588054 , 0.0583667 , 0.0578336 , 0.0576192 , 0.0581697 , 0.059936 , 0.0633367 , 0.0687103 , 0.0762655 , 0.0860397 , 0.0978767 , 0.111428 , 0.126172 , 0.141448 , 0.156494 , 0.170498 , 0.182651 , 0.192213 , 0.198568 , 0.201277 , 0.200107 , 0.195052 , 0.186334 , 0.174419 , 0.160014 , 0.144045 , 0.127589 , 0.111756 , 0.0975268 , 0.0855905 , 0.0762445 , 0.0693992 , 0.0646863 , 0.0616276 , 0.0597815 , 0.0588103 , 0.0584601 , 0.0584923 , 0.0586255 , 0.0585256 , 0.0578484 , 0.0563147 , 0.0537773 , 0.0502565 , 0.0459298 , 0.0410855 , 0.0360606 , 0.0311806 , 0.0267178 , 0.0228701 , 0.0197588 , 0.0174369 , 0.0159009 , 0.0151033 , 0.0149626 , 0.0153748 , 0.0162264 , 0.0174083 , 0.0188264 , 0.0204042 , 0.0220773 , 0.023784 , 0.0254567 , 0.0270207 , 0.0284015 , 0.0295351 , 0.0303793 , 0.0309178 , 0.0311591 , 0.0311289 , 0.030861 , 0.0303867 , 0.0297283 , 0.028894 , 0.0278806 , 0.0266805 , 0.0252932 , 0.0237388 , 0.0220662 , 0.0203539 , 0.0187022 , 0.0172199 , 0.0160123 , 0.0151749 , 0.0147952 , 0.0149592 , 0.0157559 , 0.0172723 , 0.0195767 , 0.0226932 , 0.0265745 , 0.0310855 , 0.0360043 , 0.041042 , 0.0550953 , 0.0603476 , 0.0642232 , 0.0661696 , 0.0657904 , 0.0628891 , 0.0575135 , 0.0500196 , 0.0411544 , 0.0321041 , 0.024446 , 0.0199892 , 0.0205581 , 0.0278069 , 0.0430979 , 0.0673869 , 0.101026 , 0.143475 , 0.193054 , 0.246932 , 0.301447 , 0.352658 , 0.39691 , 0.431194 , 0.453279 , 0.461668 , 0.45557 , 0.434985 , 0.400934 , 0.3557 , 0.302861 , 0.246898 , 0.192389 , 0.143097 , 0.101403 , 0.0683354 , 0.0440493 , 0.0283014 , 0.0205906 , 0.0199633 , 0.0247979 , 0.0329184 , 0.0420817 , 0.050532 , 0.0572637 , 0.0619044 , 0.0644168 , 0.0648691 , 0.0633688 , 0.0601133 , 0.055445 , 0.0498429 , 0.0438511 , 0.0379859 , 0.0326681 , 0.0281921 , 0.0247231 , 0.0223119 , 0.0209175 , 0.0204387 , 0.0207432 , 0.0216798 , 0.0230706 , 0.0246971 , 0.0263124 , 0.0276937 , 0.0287098 , 0.0293586 , 0.0297424 , 0.0299941 , 0.0302052 , 0.0303962 , 0.0305402 , 0.0306066 , 0.0305964 , 0.0305454 , 0.0304996 , 0.0304862 , 0.0304917 , 0.0304637 , 0.0303301 , 0.0300301 , 0.0295354 , 0.0288504 , 0.0279917 , 0.0269674 , 0.0257766 , 0.0244402 , 0.0230425 , 0.0217557 , 0.020821 , 0.0204895 , 0.0209588 , 0.0223401 , 0.0246698 , 0.0279463 , 0.0321488 , 0.0372137 , 0.0429732 , 0.0490994 , 0.051284 , 0.0525996 , 0.0538751 , 0.0550384 , 0.0558625 , 0.0559595 , 0.0548292 , 0.0520359 , 0.0474974 , 0.0417244 , 0.0358535 , 0.0315219 , 0.0308283 , 0.0364545 , 0.0516548 , 0.079721 , 0.122997 , 0.182051 , 0.255528 , 0.340484 , 0.432561 , 0.525808 , 0.612639 , 0.684451 , 0.732907 , 0.751464 , 0.736812 , 0.690043 , 0.617075 , 0.527652 , 0.432643 , 0.340879 , 0.257522 , 0.184983 , 0.125124 , 0.0801881 , 0.0513529 , 0.0368306 , 0.0322243 , 0.0329916 , 0.0363953 , 0.0413521 , 0.0470356 , 0.0520509 , 0.0550099 , 0.0555956 , 0.0547097 , 0.0535953 , 0.0529415 , 0.052663 , 0.05223 , 0.0511198 , 0.0491054 , 0.0463156 , 0.0431362 , 0.040055 , 0.0375223 , 0.0358483 , 0.0351417 , 0.0352929 , 0.0360163 , 0.0369627 , 0.0378819 , 0.0387409 , 0.0396763 , 0.040783 , 0.0419177 , 0.0427216 , 0.0428636 , 0.0422885 , 0.0412607 , 0.0401864 , 0.0393882 , 0.0389966 , 0.0389816 , 0.0392399 , 0.0396549 , 0.0401212 , 0.0405569 , 0.040915 , 0.041195 , 0.0414259 , 0.0416129 , 0.0416792 , 0.0414697 , 0.0408421 , 0.0397939 , 0.0385073 , 0.0372517 , 0.0362232 , 0.0354827 , 0.0350609 , 0.0351113 , 0.0359067 , 0.0376341 , 0.0401729 , 0.0430826 , 0.0458337 , 0.0480946 , 0.0498484 , 0.0565888 , 0.0539316 , 0.0512392 , 0.0491975 , 0.048172 , 0.0484606 , 0.050391 , 0.0540914 , 0.0591989 , 0.0647677 , 0.0692737 , 0.0708134 , 0.0680945 , 0.0621164 , 0.0571547 , 0.0601044 , 0.0789827 , 0.121527 , 0.193301 , 0.294883 , 0.419959 , 0.556299 , 0.688742 , 0.801563 , 0.879522 , 0.909511 , 0.884305 , 0.806763 , 0.690638 , 0.55592 , 0.421221 , 0.299047 , 0.197249 , 0.122296 , 0.0778778 , 0.0605337 , 0.0597448 , 0.064357 , 0.0686444 , 0.0713484 , 0.0712478 , 0.0669685 , 0.0596253 , 0.0526923 , 0.0489312 , 0.048209 , 0.0487618 , 0.0497051 , 0.0513735 , 0.0539348 , 0.0566197 , 0.0582156 , 0.057994 , 0.0561666 , 0.0535999 , 0.0511762 , 0.0493464 , 0.0481032 , 0.0472567 , 0.0467679 , 0.0468895 , 0.0479439 , 0.0498487 , 0.0518732 , 0.0530559 , 0.0530028 , 0.0522066 , 0.0514812 , 0.0511461 , 0.050846 , 0.050094 , 0.0488555 , 0.0475912 , 0.0468446 , 0.0468353 , 0.0473698 , 0.0480431 , 0.0485318 , 0.0487718 , 0.04894 , 0.0492895 , 0.0499532 , 0.0508252 , 0.051597 , 0.051966 , 0.0518619 , 0.0514172 , 0.0506931 , 0.0495748 , 0.0480848 , 0.0466939 , 0.0460572 , 0.0464058 , 0.0474215 , 0.0487981 , 0.0506755 , 0.0532495 , 0.0560683 , 0.0580357 , 0.0582212 , 0.0625941 , 0.0624119 , 0.061292 , 0.0597455 , 0.0577542 , 0.055327 , 0.0534845 , 0.0543467 , 0.0599596 , 0.0706158 , 0.0836069 , 0.0938481 , 0.0968377 , 0.091123 , 0.0782552 , 0.0631924 , 0.0567617 , 0.0749882 , 0.132911 , 0.239229 , 0.394735 , 0.589744 , 0.800466 , 0.991376 , 1.12507 , 1.17309 , 1.12385 , 0.988306 , 0.798551 , 0.593936 , 0.404053 , 0.246085 , 0.133252 , 0.0736392 , 0.0585302 , 0.0654681 , 0.0781241 , 0.0915296 , 0.100274 , 0.0971739 , 0.0839474 , 0.0701709 , 0.0606785 , 0.0544223 , 0.0516282 , 0.0537731 , 0.0589544 , 0.0626253 , 0.0628652 , 0.0614112 , 0.0602441 , 0.0596931 , 0.0592774 , 0.0588094 , 0.0584231 , 0.0580921 , 0.0575717 , 0.0569144 , 0.0567751 , 0.0577748 , 0.0595073 , 0.0606509 , 0.0604579 , 0.0598025 , 0.0600909 , 0.0612913 , 0.061839 , 0.060668 , 0.0585993 , 0.057253 , 0.0571935 , 0.0576689 , 0.0577908 , 0.0574434 , 0.0570965 , 0.0571124 , 0.0574121 , 0.0577023 , 0.0578426 , 0.0579211 , 0.058082 , 0.0583951 , 0.0588031 , 0.0591947 , 0.0595767 , 0.0599745 , 0.0601595 , 0.0598441 , 0.0592263 , 0.0587874 , 0.0584424 , 0.0576856 , 0.0567458 , 0.056532 , 0.0571358 , 0.0575714 , 0.0574496 , 0.0577515 , 0.0592939 , 0.0613727 , 0.0644004 , 0.0660674 , 0.0679987 , 0.0700421 , 0.0716829 , 0.0721132 , 0.0711714 , 0.0692034 , 0.0670484 , 0.0672391 , 0.0729691 , 0.0849286 , 0.1003 , 0.112225 , 0.111008 , 0.092748 , 0.067792 , 0.0582557 , 0.0917471 , 0.195204 , 0.381985 , 0.641134 , 0.939908 , 1.229 , 1.44432 , 1.52284 , 1.43789 , 1.22119 , 0.939404 , 0.649887 , 0.389982 , 0.195641 , 0.0914735 , 0.061963 , 0.069881 , 0.0925132 , 0.114308 , 0.116577 , 0.100865 , 0.0855213 , 0.0757314 , 0.0675486 , 0.0645834 , 0.0681959 , 0.0713233 , 0.071525 , 0.071972 , 0.072275 , 0.0692982 , 0.0643293 , 0.0616598 , 0.0628186 , 0.0659033 , 0.0682328 , 0.0684799 , 0.0671862 , 0.0659603 , 0.0660309 , 0.0670095 , 0.0673569 , 0.0666568 , 0.0666155 , 0.0684628 , 0.0704383 , 0.0701118 , 0.0683084 , 0.0677144 , 0.0683286 , 0.0675938 , 0.06483 , 0.0624759 , 0.062631 , 0.064629 , 0.066242 , 0.0661766 , 0.0650191 , 0.0640964 , 0.0640562 , 0.064546 , 0.0648609 , 0.0648547 , 0.0650337 , 0.0657191 , 0.0666627 , 0.0674529 , 0.0678654 , 0.0679173 , 0.0678546 , 0.0676295 , 0.0667606 , 0.0655485 , 0.0651502 , 0.065554 , 0.0656829 , 0.0658604 , 0.0670982 , 0.0682395 , 0.0673066 , 0.0649952 , 0.0637415 , 0.0743337 , 0.074437 , 0.0731262 , 0.0706651 , 0.0708552 , 0.0760894 , 0.0839956 , 0.0883984 , 0.0846053 , 0.0751169 , 0.0682025 , 0.0715153 , 0.0870562 , 0.107995 , 0.123612 , 0.125812 , 0.108541 , 0.0792366 , 0.0731423 , 0.139714 , 0.321005 , 0.633946 , 1.04196 , 1.4493 , 1.74266 , 1.84117 , 1.72479 , 1.43573 , 1.04968 , 0.649806 , 0.325206 , 0.13862 , 0.076672 , 0.0804099 , 0.108903 , 0.131547 , 0.126838 , 0.108174 , 0.0912815 , 0.0742714 , 0.0675109 , 0.0760428 , 0.0840476 , 0.0850675 , 0.0839049 , 0.0783896 , 0.0710648 , 0.0695622 , 0.0727043 , 0.0741601 , 0.0739043 , 0.0748731 , 0.0761658 , 0.075443 , 0.0734364 , 0.0727438 , 0.0738496 , 0.0746288 , 0.073951 , 0.0739014 , 0.0761044 , 0.0779584 , 0.0769235 , 0.0757022 , 0.0772151 , 0.0785578 , 0.0765808 , 0.0741007 , 0.0741445 , 0.0743632 , 0.0723327 , 0.0704485 , 0.0715404 , 0.0741524 , 0.0748181 , 0.0730591 , 0.0714721 , 0.0716596 , 0.0725692 , 0.0728585 , 0.0729768 , 0.0737162 , 0.0744933 , 0.0746247 , 0.0746159 , 0.0750467 , 0.0757295 , 0.075905 , 0.0752416 , 0.0747589 , 0.0747458 , 0.0737729 , 0.072457 , 0.0727187 , 0.0731417 , 0.0724655 , 0.0732022 , 0.075768 , 0.0766592 , 0.0752088 , 0.0820158 , 0.0836294 , 0.0838543 , 0.079579 , 0.0748706 , 0.0764104 , 0.0845616 , 0.0924223 , 0.094964 , 0.0934053 , 0.0873191 , 0.0773124 , 0.0736032 , 0.0885258 , 0.122872 , 0.154039 , 0.151871 , 0.115982 , 0.0773807 , 0.0970796 , 0.255015 , 0.591544 , 1.0817 , 1.63571 , 2.08408 , 2.24273 , 2.0539 , 1.62159 , 1.09636 , 0.602379 , 0.254341 , 0.100947 , 0.0795392 , 0.116765 , 0.158103 , 0.155057 , 0.123223 , 0.0932862 , 0.0740847 , 0.0790226 , 0.0896383 , 0.0904586 , 0.0938201 , 0.092269 , 0.0831703 , 0.0784899 , 0.0760309 , 0.0762443 , 0.0824986 , 0.087286 , 0.0846347 , 0.0794969 , 0.0766465 , 0.0766798 , 0.0791165 , 0.0818501 , 0.0822618 , 0.0815805 , 0.0832199 , 0.0854382 , 0.0842213 , 0.0826105 , 0.0848593 , 0.0865914 , 0.0843439 , 0.0834064 , 0.0852043 , 0.0839889 , 0.0804552 , 0.0800933 , 0.0816815 , 0.0811095 , 0.0796639 , 0.0798723 , 0.0803444 , 0.0796015 , 0.0790249 , 0.0793762 , 0.0797241 , 0.0803256 , 0.0820559 , 0.0832602 , 0.0824023 , 0.0815658 , 0.0823273 , 0.0832341 , 0.0830669 , 0.0827042 , 0.0831244 , 0.0829274 , 0.0816479 , 0.081844 , 0.0824539 , 0.0813699 , 0.0813725 , 0.0815652 , 0.0786237 , 0.0766907 , 0.0788336 , 0.0811257 , 0.0146347 , 0.0145889 , 0.0144774 , 0.0143053 , 0.0140809 , 0.0138155 , 0.0135233 , 0.013221 , 0.0129277 , 0.0126641 , 0.0124525 , 0.0123156 , 0.012277 , 0.0123599 , 0.0125871 , 0.0129801 , 0.0135592 , 0.0143425 , 0.0153459 , 0.0165825 , 0.0180623 , 0.019792 , 0.0217748 , 0.0240103 , 0.0264944 , 0.0292199 , 0.0321762 , 0.0353496 , 0.038724 , 0.0422808 , 0.0459988 , 0.0498553 , 0.0538247 , 0.0578798 , 0.0619907 , 0.0661252 , 0.0702487 , 0.0743239 , 0.078312 , 0.0821722 , 0.0858632 , 0.0893439 , 0.0925742 , 0.0955164 , 0.0981356 , 0.100402 , 0.102288 , 0.103775 , 0.104846 , 0.10549 , 0.105705 , 0.105488 , 0.104844 , 0.103784 , 0.102319 , 0.100468 , 0.0982508 , 0.0956936 , 0.0928234 , 0.0896713 , 0.0862707 , 0.0826566 , 0.0788656 , 0.0749351 , 0.0709023 , 0.0668047 , 0.0626786 , 0.0585593 , 0.0544804 , 0.0504742 , 0.0465711 , 0.0427991 , 0.0391843 , 0.0357503 , 0.0325184 , 0.0295068 , 0.0267307 , 0.024202 , 0.0219293 , 0.0199171 , 0.0181662 , 0.0166733 , 0.0154314 , 0.0144296 , 0.0136534 , 0.0130854 , 0.0127058 , 0.0124923 , 0.0124217 , 0.0124695 , 0.0126114 , 0.0128233 , 0.0130821 , 0.0133657 , 0.013654 , 0.0139288 , 0.0141741 , 0.0143765 , 0.0145254 , 0.014613 , 0.0360329 , 0.0357501 , 0.0351428 , 0.034237 , 0.0330679 , 0.0316776 , 0.0301135 , 0.0284261 , 0.0266673 , 0.0248897 , 0.0231462 , 0.0214907 , 0.0199782 , 0.018665 , 0.0176079 , 0.0168624 , 0.0164813 , 0.016512 , 0.0169957 , 0.0179667 , 0.0194527 , 0.0214771 , 0.0240609 , 0.0272262 , 0.0309979 , 0.0354065 , 0.0404876 , 0.0462797 , 0.0528182 , 0.0601274 , 0.0682093 , 0.0770325 , 0.0865263 , 0.0965784 , 0.107045 , 0.117763 , 0.128578 , 0.139358 , 0.150007 , 0.16047 , 0.170718 , 0.180726 , 0.190447 , 0.199793 , 0.208621 , 0.216736 , 0.223906 , 0.229888 , 0.234454 , 0.23742 , 0.238665 , 0.238141 , 0.235875 , 0.23196 , 0.226538 , 0.219784 , 0.21189 , 0.203049 , 0.193444 , 0.183242 , 0.172597 , 0.161651 , 0.150535 , 0.139374 , 0.128289 , 0.117396 , 0.106801 , 0.0965988 , 0.0868703 , 0.0776801 , 0.069077 , 0.0610958 , 0.0537601 , 0.0470857 , 0.0410831 , 0.0357583 , 0.0311134 , 0.0271453 , 0.0238432 , 0.0211877 , 0.0191502 , 0.0176936 , 0.016775 , 0.0163488 , 0.0163694 , 0.0167932 , 0.0175785 , 0.0186838 , 0.0200647 , 0.0216715 , 0.0234467 , 0.0253254 , 0.0272365 , 0.0291064 , 0.0308627 , 0.0324391 , 0.0337782 , 0.034834 , 0.0355731 , 0.0359753 , 0.0313155 , 0.031023 , 0.0304788 , 0.0297268 , 0.0288035 , 0.0277311 , 0.0265156 , 0.0251538 , 0.0236468 , 0.0220153 , 0.0203121 , 0.0186236 , 0.017064 , 0.0157605 , 0.0148406 , 0.0144229 , 0.0146108 , 0.0154884 , 0.0171113 , 0.0194959 , 0.0226087 , 0.0263623 , 0.0306201 , 0.0352101 , 0.0399418 , 0.0446194 , 0.049049 , 0.0530363 , 0.0563845 , 0.0589025 , 0.0604386 , 0.0609389 , 0.0605252 , 0.059548 , 0.0585861 , 0.0583641 , 0.0596099 , 0.0629005 , 0.0685548 , 0.0766086 , 0.0868644 , 0.0989795 , 0.112543 , 0.127107 , 0.142161 , 0.157095 , 0.171171 , 0.18356 , 0.193425 , 0.200029 , 0.202835 , 0.201575 , 0.196276 , 0.187259 , 0.175108 , 0.160626 , 0.14476 , 0.12852 , 0.11287 , 0.0986341 , 0.0864203 , 0.0765826 , 0.0692232 , 0.0642229 , 0.0612846 , 0.0599802 , 0.0598003 , 0.0602046 , 0.0606766 , 0.0607763 , 0.0601803 , 0.0587049 , 0.056304 , 0.0530505 , 0.0491031 , 0.0446721 , 0.0399859 , 0.0352669 , 0.030716 , 0.0265081 , 0.0227911 , 0.0196861 , 0.0172839 , 0.0156372 , 0.0147512 , 0.0145784 , 0.0150235 , 0.0159574 , 0.0172399 , 0.0187422 , 0.0203619 , 0.0220271 , 0.0236888 , 0.025307 , 0.0268382 , 0.0282302 , 0.0294235 , 0.0303611 , 0.0309992 , 0.0313162 , 0.0304801 , 0.0304407 , 0.0304369 , 0.0304705 , 0.0304897 , 0.0303946 , 0.0300753 , 0.0294634 , 0.0285697 , 0.0274784 , 0.0262998 , 0.0251147 , 0.0239551 , 0.0228344 , 0.0218034 , 0.0209869 , 0.0205775 , 0.0207958 , 0.0218448 , 0.0238755 , 0.0269667 , 0.0311109 , 0.0362049 , 0.0420463 , 0.0483337 , 0.0546543 , 0.0604602 , 0.0650506 , 0.0676209 , 0.0674295 , 0.064071 , 0.0577333 , 0.0492612 , 0.0399488 , 0.0311934 , 0.0242693 , 0.020393 , 0.0209979 , 0.0279068 , 0.0430888 , 0.0679852 , 0.102791 , 0.146199 , 0.195766 , 0.248553 , 0.301587 , 0.351928 , 0.396514 , 0.432122 , 0.455642 , 0.464627 , 0.457886 , 0.435832 , 0.400447 , 0.354898 , 0.302967 , 0.24851 , 0.19509 , 0.145797 , 0.103144 , 0.0689236 , 0.0440386 , 0.0284008 , 0.0210334 , 0.0203705 , 0.0246153 , 0.0319996 , 0.0408929 , 0.0498273 , 0.0575491 , 0.0631258 , 0.0660547 , 0.0662911 , 0.0641594 , 0.0601937 , 0.0549814 , 0.0490686 , 0.0429342 , 0.0370053 , 0.0316662 , 0.027238 , 0.0239281 , 0.0217885 , 0.0207125 , 0.0204892 , 0.020887 , 0.0217189 , 0.0228563 , 0.024197 , 0.0256244 , 0.0269968 , 0.0281748 , 0.0290691 , 0.0296693 , 0.0300347 , 0.0302523 , 0.0303914 , 0.0304828 , 0.0305268 , 0.0305206 , 0.0393446 , 0.0398489 , 0.040496 , 0.0410592 , 0.041344 , 0.0413032 , 0.041082 , 0.0409189 , 0.0409389 , 0.0409989 , 0.0407427 , 0.0398619 , 0.038363 , 0.0366121 , 0.0351269 , 0.0343031 , 0.0342763 , 0.0349609 , 0.0361708 , 0.0377324 , 0.0395455 , 0.0415891 , 0.0438838 , 0.0464422 , 0.0492203 , 0.0520583 , 0.0546148 , 0.0563754 , 0.0568595 , 0.055977 , 0.0541943 , 0.0521854 , 0.0501417 , 0.0474315 , 0.043189 , 0.0376236 , 0.032848 , 0.0321324 , 0.038291 , 0.0532535 , 0.0793146 , 0.120059 , 0.179014 , 0.256604 , 0.348121 , 0.444813 , 0.537174 , 0.617536 , 0.680407 , 0.721513 , 0.737293 , 0.725479 , 0.68611 , 0.622093 , 0.539088 , 0.444897 , 0.348485 , 0.258552 , 0.181881 , 0.122138 , 0.0797726 , 0.0529274 , 0.0386142 , 0.0335063 , 0.0343309 , 0.0381739 , 0.0428222 , 0.046973 , 0.0501454 , 0.0523685 , 0.0538645 , 0.0548383 , 0.0553649 , 0.0553545 , 0.0546124 , 0.0529859 , 0.0505195 , 0.047514 , 0.0444284 , 0.0416697 , 0.0394111 , 0.0375908 , 0.036103 , 0.0350017 , 0.034507 , 0.0347956 , 0.0357924 , 0.0371701 , 0.038553 , 0.039726 , 0.0406663 , 0.0414127 , 0.0419368 , 0.0421361 , 0.0419308 , 0.0413597 , 0.0405897 , 0.0398453 , 0.0393244 , 0.0391474 , 0.0475244 , 0.0482365 , 0.0487357 , 0.0487965 , 0.0486254 , 0.0487413 , 0.0495136 , 0.0507392 , 0.0517058 , 0.0517866 , 0.0510303 , 0.0500852 , 0.0494693 , 0.0490225 , 0.0481958 , 0.0468049 , 0.0453662 , 0.0447219 , 0.0454254 , 0.0474175 , 0.0501309 , 0.0528611 , 0.0551364 , 0.0568225 , 0.0578855 , 0.0580293 , 0.0566344 , 0.0533305 , 0.0489171 , 0.0455344 , 0.0453679 , 0.0489082 , 0.0546639 , 0.060823 , 0.0667243 , 0.0716413 , 0.0730482 , 0.0687354 , 0.06088 , 0.0563419 , 0.062487 , 0.0841292 , 0.125804 , 0.194208 , 0.294058 , 0.421124 , 0.560795 , 0.693736 , 0.802776 , 0.875435 , 0.902892 , 0.879938 , 0.807542 , 0.695311 , 0.560414 , 0.422604 , 0.298445 , 0.198312 , 0.126627 , 0.0830081 , 0.0629207 , 0.0589076 , 0.063084 , 0.0693001 , 0.0735636 , 0.0736202 , 0.0690339 , 0.0613592 , 0.0533454 , 0.0474857 , 0.0450654 , 0.0460562 , 0.0494045 , 0.0534603 , 0.0566264 , 0.0580528 , 0.0578765 , 0.0567738 , 0.055231 , 0.0532405 , 0.0507102 , 0.0480504 , 0.0461161 , 0.0454874 , 0.0459758 , 0.0469618 , 0.048068 , 0.0493006 , 0.0506217 , 0.0516996 , 0.0521819 , 0.0520525 , 0.0515855 , 0.0510156 , 0.0503598 , 0.0495309 , 0.0485497 , 0.0476229 , 0.0470453 , 0.047024 , 0.0574794 , 0.0575838 , 0.0574357 , 0.0572513 , 0.0575306 , 0.0583513 , 0.0590339 , 0.0588709 , 0.0581326 , 0.0579408 , 0.0588416 , 0.0598712 , 0.0596363 , 0.0581081 , 0.0566851 , 0.0564188 , 0.0568307 , 0.056741 , 0.0557606 , 0.0546517 , 0.0544458 , 0.0555534 , 0.0577609 , 0.0606724 , 0.0636626 , 0.0654569 , 0.0645848 , 0.0610321 , 0.0569889 , 0.0547892 , 0.0542556 , 0.0537527 , 0.0546241 , 0.0609398 , 0.0723395 , 0.0834602 , 0.0918168 , 0.0979931 , 0.0975136 , 0.083733 , 0.063204 , 0.055143 , 0.0751914 , 0.132445 , 0.237809 , 0.398699 , 0.601994 , 0.813128 , 0.993675 , 1.11435 , 1.15652 , 1.11275 , 0.990288 , 0.811142 , 0.606189 , 0.407991 , 0.244565 , 0.132752 , 0.0739189 , 0.0569009 , 0.0655069 , 0.0836161 , 0.0978968 , 0.101437 , 0.0951461 , 0.0839032 , 0.0719158 , 0.0616915 , 0.0548305 , 0.051998 , 0.0527312 , 0.0558411 , 0.0597415 , 0.0627095 , 0.0636766 , 0.0629772 , 0.061766 , 0.0605529 , 0.0588987 , 0.0567286 , 0.0551072 , 0.0550311 , 0.0560056 , 0.056732 , 0.0569471 , 0.057448 , 0.0584655 , 0.0593145 , 0.0596348 , 0.0598329 , 0.0601564 , 0.0602185 , 0.0596564 , 0.0586975 , 0.057864 , 0.057426 , 0.0572819 , 0.0572276 , 0.057208 , 0.0572923 , 0.0652325 , 0.0638476 , 0.0630699 , 0.0637066 , 0.0651894 , 0.0659725 , 0.0654196 , 0.0648709 , 0.0658225 , 0.0674429 , 0.0676016 , 0.0662159 , 0.0654804 , 0.0662035 , 0.0663849 , 0.064578 , 0.0623767 , 0.0621945 , 0.0640448 , 0.0658245 , 0.0660715 , 0.0654266 , 0.06541 , 0.0665932 , 0.0679964 , 0.0678703 , 0.0659345 , 0.0649075 , 0.0676708 , 0.0724866 , 0.0738904 , 0.0706741 , 0.067311 , 0.0649616 , 0.0640855 , 0.0720334 , 0.0910456 , 0.107115 , 0.111353 , 0.108897 , 0.0965676 , 0.0715453 , 0.0596239 , 0.0962029 , 0.198041 , 0.378563 , 0.645057 , 0.960468 , 1.24532 , 1.43165 , 1.49262 , 1.42538 , 1.23758 , 0.959974 , 0.653754 , 0.386438 , 0.198491 , 0.095964 , 0.0634501 , 0.0736476 , 0.0963158 , 0.112099 , 0.115722 , 0.107749 , 0.091616 , 0.07481 , 0.0645376 , 0.062631 , 0.0661725 , 0.0707913 , 0.0732871 , 0.0726453 , 0.0697025 , 0.0662604 , 0.064326 , 0.0649652 , 0.0668885 , 0.067558 , 0.0665165 , 0.0657432 , 0.0661086 , 0.0659495 , 0.0644688 , 0.0635376 , 0.0643425 , 0.0655935 , 0.0662833 , 0.0670269 , 0.0680014 , 0.0684079 , 0.0680846 , 0.0676782 , 0.0673878 , 0.0666828 , 0.0652324 , 0.0636387 , 0.0630102 , 0.0638013 , 0.065224 , 0.0659137 , 0.0724344 , 0.0704977 , 0.0703437 , 0.072135 , 0.0735224 , 0.0733161 , 0.0733814 , 0.0747314 , 0.0749928 , 0.0731154 , 0.0721935 , 0.0740193 , 0.0752473 , 0.0734269 , 0.0715436 , 0.072197 , 0.0729187 , 0.0712464 , 0.0692063 , 0.0697023 , 0.0722185 , 0.0744799 , 0.0760469 , 0.0777997 , 0.0787895 , 0.076404 , 0.0713165 , 0.0688649 , 0.0717354 , 0.075964 , 0.0791369 , 0.0833009 , 0.0844717 , 0.0795655 , 0.0756709 , 0.074345 , 0.0728253 , 0.0846195 , 0.111387 , 0.1295 , 0.128664 , 0.110423 , 0.0778568 , 0.0711055 , 0.147972 , 0.333961 , 0.634169 , 1.03095 , 1.44216 , 1.74381 , 1.84463 , 1.72455 , 1.42779 , 1.03892 , 0.650511 , 0.33826 , 0.146815 , 0.074483 , 0.0789627 , 0.110827 , 0.134477 , 0.132761 , 0.111519 , 0.0886386 , 0.0757127 , 0.0736691 , 0.076581 , 0.0791346 , 0.0811451 , 0.0826393 , 0.0812533 , 0.0762832 , 0.070702 , 0.0684378 , 0.0710433 , 0.0759589 , 0.0784299 , 0.0772252 , 0.0755838 , 0.0747603 , 0.0727822 , 0.0705139 , 0.0708149 , 0.0725309 , 0.0731109 , 0.0735372 , 0.0746807 , 0.0752192 , 0.0751894 , 0.0755325 , 0.0757058 , 0.0751474 , 0.0744931 , 0.0742727 , 0.073955 , 0.0727738 , 0.0713184 , 0.0712165 , 0.072671 , 0.0736161 , 0.0783199 , 0.078031 , 0.0788913 , 0.0797892 , 0.0802535 , 0.0821746 , 0.0846981 , 0.0834365 , 0.0800103 , 0.0802702 , 0.082484 , 0.081158 , 0.0791432 , 0.080801 , 0.0819372 , 0.0792589 , 0.0775162 , 0.0795057 , 0.0810466 , 0.0794595 , 0.077622 , 0.0781779 , 0.0805495 , 0.0830418 , 0.0831843 , 0.0800511 , 0.0779816 , 0.0808221 , 0.0829441 , 0.0794734 , 0.0789219 , 0.0831879 , 0.0875507 , 0.0959254 , 0.0973962 , 0.0830708 , 0.0756028 , 0.078799 , 0.092011 , 0.125802 , 0.151302 , 0.150479 , 0.125483 , 0.0823433 , 0.0962913 , 0.25576 , 0.591332 , 1.09141 , 1.65162 , 2.07526 , 2.21388 , 2.04615 , 1.63804 , 1.1054 , 0.601327 , 0.255274 , 0.100442 , 0.084628 , 0.126176 , 0.156582 , 0.152364 , 0.126208 , 0.0970176 , 0.0794232 , 0.0773279 , 0.0856226 , 0.0945598 , 0.0946516 , 0.0874363 , 0.0816147 , 0.0805821 , 0.080996 , 0.0799268 , 0.0794239 , 0.0814223 , 0.0829388 , 0.0815609 , 0.0803424 , 0.0804777 , 0.0791112 , 0.078082 , 0.0800702 , 0.0811853 , 0.0801801 , 0.0807318 , 0.0817526 , 0.0817603 , 0.0825285 , 0.0831038 , 0.0826206 , 0.0826735 , 0.0830068 , 0.0824028 , 0.0814576 , 0.0813401 , 0.0817307 , 0.0812272 , 0.0799121 , 0.0791872 , 0.0789626 , 0.0296325 , 0.026843 , 0.024296 , 0.0219995 , 0.0199581 , 0.0181731 , 0.0166429 , 0.0153622 , 0.0143228 , 0.0135134 , 0.0129197 , 0.0125242 , 0.0123071 , 0.0122458 , 0.0123162 , 0.0124927 , 0.0127493 , 0.0130603 , 0.0134008 , 0.0137476 , 0.0140798 , 0.0143791 , 0.0146305 , 0.014822 , 0.0149451 , 0.0149946 , 0.0149688 , 0.0148692 , 0.0147006 , 0.014471 , 0.0141916 , 0.0138768 , 0.0135439 , 0.0132129 , 0.0129065 , 0.0126494 , 0.0124673 , 0.0123872 , 0.0124357 , 0.0126384 , 0.0130198 , 0.0136016 , 0.0144031 , 0.0154404 , 0.0167259 , 0.0182688 , 0.0200746 , 0.0221451 , 0.0244788 , 0.0270704 , 0.0299105 , 0.0329863 , 0.0362812 , 0.0397742 , 0.0434409 , 0.0472531 , 0.0511796 , 0.0551867 , 0.0592388 , 0.0632991 , 0.0673308 , 0.071298 , 0.0751654 , 0.0788996 , 0.0824689 , 0.0858441 , 0.0889971 , 0.0919022 , 0.094535 , 0.0968728 , 0.0988944 , 0.10058 , 0.101913 , 0.102879 , 0.103464 , 0.103662 , 0.103468 , 0.102881 , 0.101906 , 0.100551 , 0.0988303 , 0.09676 , 0.0943606 , 0.091656 , 0.0886729 , 0.0854401 , 0.0819881 , 0.0783496 , 0.0745582 , 0.0706482 , 0.0666556 , 0.0626156 , 0.0585641 , 0.0545363 , 0.0505665 , 0.0466874 , 0.0429297 , 0.0393217 , 0.0358886 , 0.0326527 , 0.0361085 , 0.0313537 , 0.0272518 , 0.0238216 , 0.0210724 , 0.0189981 , 0.0175728 , 0.0167484 , 0.0164575 , 0.0166199 , 0.0171513 , 0.017974 , 0.0190246 , 0.0202574 , 0.0216429 , 0.0231617 , 0.024795 , 0.0265169 , 0.0282881 , 0.0300535 , 0.0317463 , 0.0332922 , 0.034617 , 0.0356535 , 0.0363472 , 0.0366594 , 0.0365702 , 0.03608 , 0.0352083 , 0.0339958 , 0.0325011 , 0.0307969 , 0.0289641 , 0.0270846 , 0.0252314 , 0.0234642 , 0.0218257 , 0.0203449 , 0.0190431 , 0.017944 , 0.0170825 , 0.0165104 , 0.0162988 , 0.0165326 , 0.0173026 , 0.0186943 , 0.0207763 , 0.0235924 , 0.0271573 , 0.031457 , 0.0364532 , 0.0420907 , 0.0483077 , 0.0550453 , 0.0622586 , 0.0699247 , 0.0780469 , 0.0866531 , 0.095786 , 0.105487 , 0.115773 , 0.126619 , 0.13794 , 0.149584 , 0.161345 , 0.172976 , 0.184217 , 0.194816 , 0.204553 , 0.213244 , 0.220749 , 0.22696 , 0.231796 , 0.235195 , 0.23711 , 0.237513 , 0.236399 , 0.233791 , 0.229744 , 0.224346 , 0.217713 , 0.20998 , 0.201296 , 0.191817 , 0.181698 , 0.171097 , 0.160166 , 0.149059 , 0.137926 , 0.12691 , 0.11614 , 0.105726 , 0.0957573 , 0.0862958 , 0.0773823 , 0.0690374 , 0.0612715 , 0.0540883 , 0.0474932 , 0.041495 , 0.0449636 , 0.0402012 , 0.0353854 , 0.0307715 , 0.026553 , 0.0228566 , 0.0197585 , 0.0173084 , 0.0155483 , 0.0145135 , 0.0142153 , 0.0146167 , 0.015616 , 0.0170525 , 0.0187349 , 0.0204828 , 0.0221634 , 0.0237092 , 0.0251114 , 0.0263952 , 0.0275896 , 0.0287027 , 0.0297105 , 0.0305595 , 0.0311793 , 0.0315012 , 0.0314761 , 0.0310873 , 0.0303561 , 0.0293372 , 0.0281047 , 0.0267314 , 0.0252713 , 0.02375 , 0.0221692 , 0.0205287 , 0.0188568 , 0.0172388 , 0.015825 , 0.0148075 , 0.0143721 , 0.0146495 , 0.0156897 , 0.0174727 , 0.0199412 , 0.0230338 , 0.0266967 , 0.0308682 , 0.0354448 , 0.0402492 , 0.0450213 , 0.0494451 , 0.0532112 , 0.0560947 , 0.0580201 , 0.0590836 , 0.0595206 , 0.0596345 , 0.05972 , 0.0600292 , 0.0608004 , 0.0623315 , 0.0650444 , 0.0694805 , 0.0761965 , 0.0855893 , 0.0977195 , 0.112214 , 0.128291 , 0.144894 , 0.160876 , 0.175172 , 0.186905 , 0.195437 , 0.200356 , 0.201477 , 0.198816 , 0.192594 , 0.183213 , 0.171235 , 0.157339 , 0.142286 , 0.12687 , 0.111885 , 0.0980661 , 0.0860311 , 0.0762081 , 0.0687827 , 0.0636854 , 0.0606269 , 0.0591716 , 0.0588203 , 0.0590797 , 0.0595007 , 0.0596959 , 0.0593467 , 0.0582164 , 0.0561696 , 0.0531902 , 0.0493848 , 0.0552133 , 0.0493247 , 0.0431543 , 0.0371538 , 0.0317163 , 0.0271729 , 0.0237626 , 0.0215795 , 0.0205368 , 0.0203917 , 0.0208395 , 0.0216294 , 0.0226272 , 0.0237921 , 0.025095 , 0.0264486 , 0.0277052 , 0.0287163 , 0.0294047 , 0.0297994 , 0.0300094 , 0.0301635 , 0.030349 , 0.030579 , 0.0307994 , 0.0309255 , 0.0308919 , 0.0306929 , 0.0303914 , 0.0300927 , 0.0298815 , 0.0297596 , 0.0296077 , 0.0292203 , 0.0284151 , 0.0271604 , 0.0256148 , 0.0240388 , 0.0226507 , 0.0215499 , 0.0207593 , 0.020342 , 0.0204979 , 0.021551 , 0.0238144 , 0.0274213 , 0.0322434 , 0.0379346 , 0.0440384 , 0.0500738 , 0.0555718 , 0.0600903 , 0.0632401 , 0.0647167 , 0.0643068 , 0.0618535 , 0.0572394 , 0.0504832 , 0.0419855 , 0.0328027 , 0.0247211 , 0.0199793 , 0.0207389 , 0.0286176 , 0.0445609 , 0.0690326 , 0.102213 , 0.143885 , 0.192998 , 0.24723 , 0.302961 , 0.355766 , 0.401226 , 0.435676 , 0.456634 , 0.462887 , 0.454354 , 0.431902 , 0.397223 , 0.352744 , 0.301561 , 0.247275 , 0.193675 , 0.144273 , 0.101838 , 0.0680761 , 0.0436015 , 0.0281209 , 0.020705 , 0.0200009 , 0.0243655 , 0.0319853 , 0.0410476 , 0.0499488 , 0.0574642 , 0.0628208 , 0.0656702 , 0.0660078 , 0.0640817 , 0.0603118 , 0.0526922 , 0.050586 , 0.0477251 , 0.0445381 , 0.0415788 , 0.0391942 , 0.0373619 , 0.0358599 , 0.03463 , 0.0339609 , 0.0342414 , 0.0354952 , 0.0372016 , 0.038646 , 0.0394692 , 0.0398624 , 0.0402514 , 0.0408346 , 0.041437 , 0.0417445 , 0.0416197 , 0.0411922 , 0.0407045 , 0.0403052 , 0.039975 , 0.0396319 , 0.0393008 , 0.0391802 , 0.0395342 , 0.0404578 , 0.0416794 , 0.0426009 , 0.0426724 , 0.041855 , 0.0406878 , 0.0397991 , 0.0393384 , 0.0388652 , 0.0378259 , 0.0362265 , 0.0347661 , 0.0341957 , 0.0346672 , 0.0358021 , 0.0372517 , 0.0390824 , 0.0416359 , 0.0450195 , 0.0487359 , 0.0518596 , 0.0536399 , 0.054029 , 0.0537219 , 0.053635 , 0.0541293 , 0.0545726 , 0.0537361 , 0.0508552 , 0.0463789 , 0.0415742 , 0.0374344 , 0.0343728 , 0.0334097 , 0.0376053 , 0.0517667 , 0.0802257 , 0.124773 , 0.184691 , 0.258263 , 0.343459 , 0.436803 , 0.531844 , 0.619245 , 0.688902 , 0.732681 , 0.746143 , 0.728778 , 0.683301 , 0.614786 , 0.529982 , 0.436709 , 0.343037 , 0.25623 , 0.181735 , 0.122641 , 0.0797601 , 0.0520805 , 0.0372405 , 0.0320018 , 0.0328866 , 0.0368988 , 0.0419623 , 0.0468519 , 0.0508549 , 0.0535693 , 0.0549422 , 0.0552908 , 0.0550967 , 0.0546725 , 0.0539714 , 0.0576782 , 0.0579851 , 0.0570253 , 0.0554256 , 0.0533858 , 0.0506663 , 0.0474739 , 0.044997 , 0.0444776 , 0.0457359 , 0.047224 , 0.0478378 , 0.0481455 , 0.0492953 , 0.0511346 , 0.0522214 , 0.0516623 , 0.0502166 , 0.0493346 , 0.0495668 , 0.0502422 , 0.0504117 , 0.0497201 , 0.0484498 , 0.0470965 , 0.046111 , 0.0459073 , 0.0467753 , 0.0485428 , 0.050338 , 0.0510543 , 0.050453 , 0.0496665 , 0.0501242 , 0.0519159 , 0.0533891 , 0.0527993 , 0.0504864 , 0.0484431 , 0.0477164 , 0.047442 , 0.046487 , 0.0453451 , 0.0456804 , 0.0480361 , 0.0511706 , 0.0537278 , 0.0555202 , 0.0569794 , 0.0579708 , 0.0577029 , 0.0557984 , 0.0529282 , 0.0500891 , 0.0477013 , 0.046122 , 0.0468802 , 0.051973 , 0.060898 , 0.069481 , 0.0733881 , 0.0723065 , 0.0689162 , 0.0647173 , 0.0604923 , 0.0616984 , 0.0796051 , 0.124439 , 0.198958 , 0.299717 , 0.421451 , 0.556725 , 0.691986 , 0.807489 , 0.883563 , 0.907978 , 0.878762 , 0.802279 , 0.690094 , 0.557101 , 0.420181 , 0.295539 , 0.194974 , 0.123645 , 0.0806917 , 0.061215 , 0.0578601 , 0.0624682 , 0.0683523 , 0.0717629 , 0.0713827 , 0.0672214 , 0.0604415 , 0.0533476 , 0.0483047 , 0.0463642 , 0.0471169 , 0.0495765 , 0.0527872 , 0.0557955 , 0.0621571 , 0.0614281 , 0.0609364 , 0.0596944 , 0.0571209 , 0.0545736 , 0.0541485 , 0.0556868 , 0.0566747 , 0.056073 , 0.056014 , 0.0579624 , 0.0598423 , 0.0594142 , 0.058044 , 0.0582532 , 0.0595076 , 0.0594356 , 0.0578853 , 0.057059 , 0.058134 , 0.0597515 , 0.0599054 , 0.0581597 , 0.0558223 , 0.0546337 , 0.0555063 , 0.0578951 , 0.0598691 , 0.0595187 , 0.0572609 , 0.0559617 , 0.0577243 , 0.0609607 , 0.0618179 , 0.0595975 , 0.0580265 , 0.0594004 , 0.0611195 , 0.059867 , 0.0570996 , 0.0561982 , 0.0567164 , 0.0560846 , 0.0545838 , 0.0550935 , 0.058116 , 0.0607574 , 0.0609361 , 0.0597792 , 0.0598125 , 0.0619827 , 0.0643243 , 0.0633628 , 0.0579098 , 0.05178 , 0.0505471 , 0.0550001 , 0.0616817 , 0.0703762 , 0.0837947 , 0.0979194 , 0.102443 , 0.0942457 , 0.0802059 , 0.0663186 , 0.0583115 , 0.0729463 , 0.132742 , 0.246876 , 0.407021 , 0.59834 , 0.801981 , 0.98835 , 1.12018 , 1.16784 , 1.12138 , 0.991365 , 0.803877 , 0.594174 , 0.397697 , 0.240031 , 0.132428 , 0.0742731 , 0.0565282 , 0.0640211 , 0.0803327 , 0.0937945 , 0.0989135 , 0.0945661 , 0.0834293 , 0.0708143 , 0.0609474 , 0.0549081 , 0.0523889 , 0.0532548 , 0.0566412 , 0.0604407 , 0.0627018 , 0.0629668 , 0.0641638 , 0.0664449 , 0.0681692 , 0.0673082 , 0.0659001 , 0.0660109 , 0.0656393 , 0.0631365 , 0.0618134 , 0.0640221 , 0.0659853 , 0.0651152 , 0.0651169 , 0.0675827 , 0.068227 , 0.0659058 , 0.0649573 , 0.0665045 , 0.0669041 , 0.0651904 , 0.0643611 , 0.0655352 , 0.0662508 , 0.0646999 , 0.0624421 , 0.0622028 , 0.0645013 , 0.0668881 , 0.0663501 , 0.0633208 , 0.0620188 , 0.0650019 , 0.0687951 , 0.0681458 , 0.0652225 , 0.0663518 , 0.0704322 , 0.0701869 , 0.065964 , 0.0649874 , 0.0670981 , 0.0662449 , 0.0632444 , 0.0635006 , 0.0657384 , 0.0660923 , 0.0661087 , 0.0682152 , 0.0690843 , 0.0655334 , 0.0614492 , 0.0627301 , 0.0684613 , 0.0722961 , 0.0723076 , 0.0722276 , 0.0715679 , 0.0663774 , 0.0616906 , 0.0662343 , 0.0769679 , 0.0884659 , 0.104044 , 0.118123 , 0.114438 , 0.0935907 , 0.0720064 , 0.0637809 , 0.0930816 , 0.19682 , 0.390224 , 0.652406 , 0.946393 , 1.22664 , 1.4343 , 1.51365 , 1.44071 , 1.23446 , 0.946892 , 0.643624 , 0.382215 , 0.196364 , 0.093347 , 0.0600063 , 0.0699161 , 0.0938348 , 0.111163 , 0.113763 , 0.103461 , 0.0878584 , 0.0741826 , 0.0658794 , 0.0640709 , 0.0673884 , 0.0714249 , 0.0728265 , 0.0720006 , 0.0700766 , 0.0671658 , 0.0644323 , 0.0759052 , 0.0778184 , 0.0772664 , 0.0761345 , 0.0753085 , 0.0724602 , 0.0690002 , 0.0697804 , 0.0725039 , 0.0719339 , 0.0717805 , 0.0747522 , 0.0750194 , 0.0728018 , 0.0738708 , 0.075111 , 0.0727281 , 0.0718826 , 0.0746751 , 0.0757931 , 0.0738525 , 0.0725362 , 0.0724228 , 0.0713745 , 0.0700862 , 0.0709696 , 0.0733851 , 0.0738637 , 0.0714088 , 0.070146 , 0.0732241 , 0.0765 , 0.0743467 , 0.071429 , 0.0747422 , 0.0787223 , 0.0754421 , 0.0727092 , 0.0767796 , 0.078063 , 0.0732031 , 0.0721289 , 0.0740359 , 0.0716704 , 0.0696636 , 0.0727256 , 0.0755157 , 0.0758223 , 0.0767777 , 0.0774677 , 0.0754434 , 0.0722612 , 0.069421 , 0.0688343 , 0.0744921 , 0.082502 , 0.0831744 , 0.0806144 , 0.0815279 , 0.0768773 , 0.069549 , 0.0762951 , 0.0918127 , 0.109103 , 0.130728 , 0.135113 , 0.109395 , 0.0799745 , 0.0766915 , 0.14099 , 0.330489 , 0.651965 , 1.04497 , 1.4305 , 1.72533 , 1.84516 , 1.74319 , 1.44404 , 1.0373 , 0.636113 , 0.326308 , 0.142058 , 0.0730942 , 0.0787914 , 0.109032 , 0.129292 , 0.127494 , 0.108951 , 0.0876243 , 0.0735073 , 0.0702912 , 0.0759749 , 0.0821121 , 0.0839073 , 0.0832845 , 0.0801615 , 0.0743212 , 0.0699547 , 0.0698317 , 0.0725468 , 0.0827684 , 0.0801387 , 0.0794184 , 0.0809436 , 0.0798342 , 0.0778223 , 0.0797136 , 0.0804604 , 0.0783293 , 0.0802074 , 0.0820073 , 0.0798875 , 0.0812846 , 0.082811 , 0.0798597 , 0.0803 , 0.0824451 , 0.0804823 , 0.0808728 , 0.0849521 , 0.0848992 , 0.0808759 , 0.0784391 , 0.0780765 , 0.0782655 , 0.079104 , 0.0796246 , 0.0785641 , 0.0783194 , 0.0815587 , 0.0844868 , 0.0816888 , 0.0789109 , 0.0829219 , 0.0853705 , 0.080519 , 0.0811828 , 0.086789 , 0.0830531 , 0.0795976 , 0.0845594 , 0.0837288 , 0.0790194 , 0.080624 , 0.0805795 , 0.0780712 , 0.0803541 , 0.0809607 , 0.0772782 , 0.0785381 , 0.0854782 , 0.0864748 , 0.0790828 , 0.0758265 , 0.080411 , 0.0810765 , 0.081075 , 0.0900388 , 0.0936603 , 0.0903898 , 0.0889243 , 0.0786915 , 0.0764275 , 0.0972962 , 0.12425 , 0.153869 , 0.159426 , 0.120266 , 0.0819278 , 0.101444 , 0.253598 , 0.603233 , 1.10225 , 1.62675 , 2.04841 , 2.23002 , 2.07862 , 1.64087 , 1.08742 , 0.592302 , 0.254234 , 0.0975818 , 0.0797397 , 0.119446 , 0.15307 , 0.15275 , 0.123776 , 0.0922759 , 0.0758163 , 0.0769061 , 0.0864901 , 0.0932647 , 0.0947808 , 0.090174 , 0.082403 , 0.0789509 , 0.0791304 , 0.0791153 , 0.0803785 , 0.0828169 , 0.792639 , 0.792994 , 0.793838 , 0.795202 , 0.797125 , 0.799656 , 0.802853 , 0.806785 , 0.811527 , 0.817163 , 0.823784 , 0.831483 , 0.84036 , 0.850523 , 0.86208 , 0.875152 , 0.889869 , 0.906375 , 0.924832 , 0.945423 , 0.968355 , 0.993863 , 1.0222 , 1.05366 , 1.08853 , 1.1271 , 1.16965 , 1.21641 , 1.26754 , 1.32306 , 1.38284 , 1.44655 , 1.51364 , 1.58332 , 1.6546 , 1.72635 , 1.79737 , 1.86648 , 1.9326 , 1.99484 , 2.05247 , 2.10498 , 2.15202 , 2.19342 , 2.22909 , 2.25905 , 2.28333 , 2.30202 , 2.3152 , 2.32291 , 2.32523 , 2.32215 , 2.31368 , 2.29977 , 2.28038 , 2.25543 , 2.22484 , 2.18856 , 2.14657 , 2.09891 , 2.04574 , 1.98733 , 1.92416 , 1.85691 , 1.78646 , 1.71391 , 1.64047 , 1.56742 , 1.49598 , 1.42724 , 1.36205 , 1.30105 , 1.24459 , 1.19284 , 1.14575 , 1.10316 , 1.06483 , 1.03047 , 0.999745 , 0.972347 , 0.94797 , 0.926326 , 0.907153 , 0.890211 , 0.875282 , 0.862167 , 0.850688 , 0.840678 , 0.831987 , 0.824477 , 0.818022 , 0.812509 , 0.807838 , 0.803919 , 0.800676 , 0.798043 , 0.79597 , 0.794415 , 0.793351 , 0.792761 , -0.246852 , -0.245358 , -0.241948 , -0.236468 , -0.22875 , -0.21865 , -0.20607 , -0.190992 , -0.173488 , -0.153714 , -0.131888 , -0.108235 , -0.0829336 , -0.0560553 , -0.0275215 , 0.00290911 , 0.0356296 , 0.0711432 , 0.110009 , 0.152778 , 0.199938 , 0.251865 , 0.308807 , 0.37089 , 0.438153 , 0.510614 , 0.588356 , 0.671613 , 0.760813 , 0.856563 , 0.959468 , 1.06978 , 1.18687 , 1.3087 , 1.43168 , 1.55108 , 1.66215 , 1.76122 , 1.8464 , 1.91751 , 1.97558 , 2.02232 , 2.05962 , 2.08926 , 2.11273 , 2.13116 , 2.14535 , 2.15579 , 2.16276 , 2.16639 , 2.1667 , 2.16372 , 2.15743 , 2.14778 , 2.13464 , 2.11774 , 2.09662 , 2.0705 , 2.03829 , 1.99856 , 1.94962 , 1.8897 , 1.81723 , 1.73128 , 1.63208 , 1.52138 , 1.40241 , 1.27938 , 1.15649 , 1.03716 , 0.923624 , 0.817018 , 0.717744 , 0.625797 , 0.540991 , 0.463058 , 0.391689 , 0.326508 , 0.267093 , 0.212965 , 0.163605 , 0.118483 , 0.077084 , 0.0389351 , 0.00363754 , -0.0291108 , -0.0595051 , -0.0876327 , -0.113495 , -0.137037 , -0.158187 , -0.176892 , -0.193143 , -0.206991 , -0.218536 , -0.227919 , -0.235293 , -0.240795 , -0.244532 , -0.246554 , -1.85777 , -1.85293 , -1.8403 , -1.81926 , -1.78913 , -1.74923 , -1.69905 , -1.6382 , -1.56628 , -1.48269 , -1.38654 , -1.2769 , -1.15376 , -1.01932 , -0.878866 , -0.739929 , -0.609748 , -0.492799 , -0.390056 , -0.299923 , -0.21967 , -0.146535 , -0.0782535 , -0.0131054 , 0.0503311 , 0.11357 , 0.17858 , 0.24801 , 0.325158 , 0.413555 , 0.516018 , 0.633157 , 0.76195 , 0.895666 , 1.02612 , 1.14693 , 1.25523 , 1.3509 , 1.43468 , 1.50678 , 1.56669 , 1.61374 , 1.64785 , 1.66982 , 1.68138 , 1.68485 , 1.6828 , 1.6777 , 1.67165 , 1.66619 , 1.66225 , 1.66016 , 1.65971 , 1.66017 , 1.6603 , 1.65836 , 1.65216 , 1.63926 , 1.6172 , 1.5839 , 1.53796 , 1.47872 , 1.40601 , 1.31963 , 1.21906 , 1.10402 , 0.975904 , 0.839214 , 0.701542 , 0.57116 , 0.453901 , 0.351705 , 0.263304 , 0.185781 , 0.115852 , 0.0505541 , -0.0125706 , -0.0755558 , -0.140218 , -0.208368 , -0.281952 , -0.363043 , -0.45366 , -0.555336 , -0.668519 , -0.791974 , -0.922499 , -1.0553 , -1.185 , -1.30677 , -1.41711 , -1.51408 , -1.59703 , -1.66628 , -1.72276 , -1.76769 , -1.80235 , -1.82793 , -1.84535 , -1.85521 , 2.67233 , 2.67759 , 2.68815 , 2.70435 , 2.72651 , 2.75516 , 2.79144 , 2.83799 , 2.90007 , 2.98722 , 3.11534 , -2.97499 , -2.69132 , -2.31899 , -1.92656 , -1.59613 , -1.34912 , -1.16947 , -1.03763 , -0.938413 , -0.860238 , -0.794088 , -0.732965 , -0.671548 , -0.60586 , -0.532931 , -0.450606 , -0.357621 , -0.253902 , -0.140637 , -0.0196093 , 0.108081 , 0.242579 , 0.383676 , 0.527525 , 0.664992 , 0.78478 , 0.87913 , 0.946812 , 0.992035 , 1.0216 , 1.04215 , 1.0582 , 1.07143 , 1.0813 , 1.08649 , 1.08623 , 1.08093 , 1.07225 , 1.06263 , 1.05452 , 1.04962 , 1.04838 , 1.04993 , 1.05239 , 1.05337 , 1.05058 , 1.04239 , 1.028 , 1.00708 , 0.97874 , 0.940647 , 0.888682 , 0.817806 , 0.724092 , 0.60723 , 0.471866 , 0.32609 , 0.178019 , 0.03353 , -0.103545 , -0.230409 , -0.345062 , -0.4467 , -0.535936 , -0.614514 , -0.684847 , -0.749705 , -0.812176 , -0.875799 , -0.944814 , -1.02452 , -1.12198 , -1.24751 , -1.41736 , -1.65589 , -1.98563 , -2.38184 , -2.74931 , -3.02158 , 3.07752 , 2.95322 , 2.86685 , 2.80505 , 2.76003 , 2.72715 , 2.70352 , 2.68722 , 2.67702 , 2.67215 , 1.22126 , 1.23053 , 1.248 , 1.27324 , 1.30624 , 1.34842 , 1.40376 , 1.47959 , 1.58622 , 1.73485 , 1.9348 , 2.19356 , 2.51774 , 2.89476 , -3.01911 , -2.71984 , -2.49322 , -2.30809 , -2.13734 , -1.97018 , -1.80923 , -1.66002 , -1.5224 , -1.39034 , -1.25702 , -1.11995 , -0.982653 , -0.851407 , -0.729096 , -0.6115 , -0.48927 , -0.354701 , -0.209578 , -0.0665661 , 0.0606815 , 0.168907 , 0.265401 , 0.358553 , 0.448922 , 0.529318 , 0.592493 , 0.63693 , 0.666742 , 0.688558 , 0.70792 , 0.726841 , 0.743914 , 0.756445 , 0.762501 , 0.76195 , 0.756437 , 0.748379 , 0.739304 , 0.728737 , 0.71458 , 0.694857 , 0.669468 , 0.640251 , 0.608737 , 0.573345 , 0.52881 , 0.468887 , 0.390828 , 0.297808 , 0.195705 , 0.0868613 , -0.0316689 , -0.16309 , -0.304654 , -0.44771 , -0.58459 , -0.713991 , -0.839665 , -0.965906 , -1.09455 , -1.22496 , -1.35605 , -1.48828 , -1.62397 , -1.76594 , -1.91544 , -2.0715 , -2.2331 , -2.40348 , -2.59352 , -2.8217 , -3.10743 , 2.83052 , 2.46293 , 2.13234 , 1.87228 , 1.6814 , 1.5442 , 1.44492 , 1.37171 , 1.31696 , 1.27633 , 1.24751 , 1.22922 , 1.22067 , -0.327102 , -0.314268 , -0.289092 , -0.251911 , -0.200623 , -0.128319 , -0.0217867 , 0.136775 , 0.363177 , 0.664008 , 1.03286 , 1.4293 , 1.7743 , 2.03158 , 2.23286 , 2.42503 , 2.63364 , 2.85497 , 3.07818 , -2.97429 , -2.72543 , -2.469 , -2.23247 , -2.02954 , -1.85422 , -1.69709 , -1.5532 , -1.41718 , -1.27856 , -1.1268 , -0.963539 , -0.805964 , -0.668222 , -0.542827 , -0.409865 , -0.264837 , -0.129155 , -0.0239961 , 0.0529231 , 0.11989 , 0.190883 , 0.263541 , 0.327157 , 0.376567 , 0.41445 , 0.446205 , 0.475142 , 0.500977 , 0.521591 , 0.535038 , 0.539972 , 0.534974 , 0.517994 , 0.487175 , 0.443873 , 0.394879 , 0.348617 , 0.306925 , 0.261932 , 0.202878 , 0.127573 , 0.0457298 , -0.0342083 , -0.118577 , -0.223055 , -0.354118 , -0.499122 , -0.642226 , -0.783758 , -0.933725 , -1.09369 , -1.25276 , -1.40096 , -1.53941 , -1.67657 , -1.82081 , -1.97795 , -2.15368 , -2.35489 , -2.58517 , -2.83712 , -3.09296 , 2.94681 , 2.72287 , 2.51858 , 2.32893 , 2.14153 , 1.93561 , 1.68461 , 1.36539 , 0.984933 , 0.606767 , 0.302214 , 0.0868017 , -0.0600225 , -0.161196 , -0.231974 , -0.280699 , -0.311523 , -0.326556 , -1.93132 , -1.91649 , -1.88914 , -1.84784 , -1.78467 , -1.68255 , -1.51595 , -1.25482 , -0.863375 , -0.33492 , 0.168406 , 0.502233 , 0.730011 , 0.939344 , 1.15997 , 1.38037 , 1.60939 , 1.8864 , 2.21042 , 2.51526 , 2.77553 , 3.02418 , -3.00527 , -2.76868 , -2.56874 , -2.39539 , -2.22278 , -2.02842 , -1.812 , -1.60464 , -1.43483 , -1.28926 , -1.13156 , -0.953923 , -0.791624 , -0.668045 , -0.562415 , -0.446214 , -0.325947 , -0.225938 , -0.148823 , -0.0794703 , -0.00836521 , 0.0602775 , 0.120391 , 0.170017 , 0.207266 , 0.231162 , 0.24585 , 0.25746 , 0.267054 , 0.268534 , 0.252259 , 0.213056 , 0.158115 , 0.102602 , 0.0522761 , -0.00362706 , -0.0773512 , -0.163031 , -0.244063 , -0.321944 , -0.417202 , -0.53815 , -0.665764 , -0.78708 , -0.918582 , -1.07754 , -1.25156 , -1.41893 , -1.58411 , -1.76624 , -1.96782 , -2.17032 , -2.3593 , -2.53844 , -2.72127 , -2.92046 , -3.14151 , 2.90157 , 2.64499 , 2.36569 , 2.06155 , 1.75694 , 1.48426 , 1.24934 , 1.03994 , 0.843567 , 0.643606 , 0.407124 , 0.0788062 , -0.391347 , -0.918339 , -1.31842 , -1.56676 , -1.71936 , -1.81663 , -1.87885 , -1.9156 , -1.93194 , 2.7742 , 2.79111 , 2.82245 , 2.87209 , 2.95465 , 3.09517 , -2.95573 , -2.56666 , -1.93706 , -1.32411 , -0.959508 , -0.69695 , -0.441803 , -0.194844 , 0.0617904 , 0.390879 , 0.770468 , 1.10542 , 1.40687 , 1.70427 , 1.97048 , 2.20812 , 2.46124 , 2.73742 , 2.99469 , -3.06137 , -2.8318 , -2.58614 , -2.35846 , -2.16848 , -1.98234 , -1.77273 , -1.57241 , -1.41435 , -1.26716 , -1.09655 , -0.933217 , -0.807951 , -0.701228 , -0.592017 , -0.489998 , -0.40218 , -0.321834 , -0.249979 , -0.19044 , -0.135555 , -0.0815202 , -0.0405851 , -0.0238379 , -0.0216852 , -0.0187165 , -0.0163414 , -0.0307379 , -0.0729645 , -0.132406 , -0.191333 , -0.253105 , -0.332521 , -0.422577 , -0.502957 , -0.582489 , -0.688275 , -0.814814 , -0.93479 , -1.05948 , -1.21768 , -1.39464 , -1.56129 , -1.73348 , -1.9327 , -2.14029 , -2.33654 , -2.53662 , -2.75792 , -2.99124 , 3.06013 , 2.8221 , 2.56691 , 2.30528 , 2.05507 , 1.81104 , 1.55152 , 1.26503 , 0.94975 , 0.603043 , 0.250052 , -0.0623999 , -0.326196 , -0.561774 , -0.789938 , -1.04868 , -1.42038 , -1.99653 , -2.61609 , -3.01778 , 3.0442 , 2.91756 , 2.84079 , 2.79529 , 2.77429 , 1.19545 , 1.2127 , 1.2497 , 1.31585 , 1.43258 , 1.63118 , 1.97652 , 2.61671 , -2.94604 , -2.52392 , -2.22108 , -1.93906 , -1.66837 , -1.32911 , -0.884257 , -0.467503 , -0.0927985 , 0.255857 , 0.534269 , 0.796135 , 1.10743 , 1.43344 , 1.73554 , 2.03298 , 2.32414 , 2.59569 , 2.85254 , 3.09193 , -2.964 , -2.71837 , -2.46407 , -2.25809 , -2.08527 , -1.88066 , -1.66769 , -1.50253 , -1.35726 , -1.20271 , -1.06459 , -0.945405 , -0.83309 , -0.737906 , -0.652359 , -0.56233 , -0.488242 , -0.444303 , -0.402473 , -0.349 , -0.308889 , -0.300031 , -0.306797 , -0.315934 , -0.33744 , -0.380018 , -0.433384 , -0.493156 , -0.573003 , -0.667194 , -0.749952 , -0.832726 , -0.946483 , -1.07306 , -1.1885 , -1.32513 , -1.49422 , -1.65687 , -1.82413 , -2.02996 , -2.2442 , -2.44365 , -2.66185 , -2.90635 , 3.13726 , 2.90499 , 2.66204 , 2.4053 , 2.13483 , 1.84594 , 1.54482 , 1.24074 , 0.936543 , 0.643513 , 0.363951 , 0.0719317 , -0.256971 , -0.631151 , -1.04976 , -1.45604 , -1.78866 , -2.06333 , -2.31975 , -2.60908 , -3.04107 , 2.55874 , 1.92216 , 1.5673 , 1.38232 , 1.28063 , 1.22429 , 1.19803 , -0.394388 , -0.375423 , -0.331942 , -0.248437 , -0.0979604 , 0.170498 , 0.733455 , 1.60307 , 2.13537 , 2.48015 , 2.77866 , 3.07005 , -2.80489 , -2.30406 , -1.83573 , -1.38287 , -1.03499 , -0.734715 , -0.39446 , -0.059937 , 0.29723 , 0.699019 , 1.0327 , 1.30992 , 1.61649 , 1.94086 , 2.21752 , 2.46517 , 2.75009 , 3.04907 , -3.00367 , -2.79582 , -2.54743 , -2.32396 , -2.1437 , -1.94435 , -1.75202 , -1.5922 , -1.44013 , -1.30724 , -1.18414 , -1.0594 , -0.9677 , -0.891297 , -0.796413 , -0.724208 , -0.697221 , -0.664131 , -0.6098 , -0.576562 , -0.584482 , -0.613865 , -0.648515 , -0.68438 , -0.725967 , -0.793588 , -0.891039 , -0.984617 , -1.07375 , -1.19067 , -1.31474 , -1.43036 , -1.57978 , -1.74978 , -1.90827 , -2.09767 , -2.31158 , -2.51101 , -2.73377 , -2.98123 , 3.06946 , 2.82355 , 2.54465 , 2.27221 , 2.00988 , 1.72468 , 1.41463 , 1.10858 , 0.810418 , 0.483819 , 0.11844 , -0.243779 , -0.586099 , -0.909484 , -1.22176 , -1.57029 , -1.98944 , -2.46538 , -2.94118 , 2.95339 , 2.65231 , 2.37931 , 2.05191 , 1.50326 , 0.674971 , 0.10959 , -0.154727 , -0.287951 , -0.358763 , -0.390839 , 1.12238 , 1.08434 , 1.05001 , 1.01909 , 0.991264 , 0.966253 , 0.943789 , 0.92363 , 0.905557 , 0.889376 , 0.874916 , 0.862023 , 0.850564 , 0.840418 , 0.831477 , 0.823644 , 0.816832 , 0.81096 , 0.805956 , 0.801754 , 0.798294 , 0.795524 , 0.7934 , 0.791885 , 0.790948 , 0.790566 , 0.790724 , 0.791413 , 0.792634 , 0.794394 , 0.796708 , 0.799603 , 0.803114 , 0.807289 , 0.812185 , 0.817877 , 0.82445 , 0.832008 , 0.84067 , 0.850568 , 0.861858 , 0.874707 , 0.889305 , 0.90586 , 0.924602 , 0.94578 , 0.969666 , 0.996554 , 1.02676 , 1.06059 , 1.0984 , 1.14048 , 1.18712 , 1.23852 , 1.29477 , 1.35579 , 1.42128 , 1.4907 , 1.56319 , 1.63765 , 1.71277 , 1.78712 , 1.85932 , 1.9281 , 1.99241 , 2.0515 , 2.10486 , 2.15224 , 2.19357 , 2.22891 , 2.25841 , 2.28227 , 2.30067 , 2.31379 , 2.32176 , 2.32465 , 2.3225 , 2.31527 , 2.30287 , 2.28515 , 2.26195 , 2.23307 , 2.19833 , 2.15758 , 2.11081 , 2.0581 , 1.99977 , 1.93637 , 1.8687 , 1.79784 , 1.72502 , 1.65161 , 1.57895 , 1.50825 , 1.44052 , 1.37652 , 1.31675 , 1.26146 , 1.21071 , 1.16441 , 0.505322 , 0.43493 , 0.37009 , 0.310221 , 0.254852 , 0.203608 , 0.156183 , 0.112317 , 0.0717844 , 0.0343743 , -0.00011548 , -0.0318807 , -0.0611079 , -0.0879654 , -0.112595 , -0.135103 , -0.155556 , -0.17398 , -0.190367 , -0.204687 , -0.216899 , -0.226967 , -0.234869 , -0.240611 , -0.244229 , -0.245791 , -0.245388 , -0.243126 , -0.239108 , -0.233412 , -0.226072 , -0.217062 , -0.206282 , -0.193562 , -0.178675 , -0.161366 , -0.141385 , -0.118537 , -0.0927062 , -0.0638794 , -0.0321364 , 0.00238799 , 0.0395618 , 0.079321 , 0.121733 , 0.167033 , 0.215646 , 0.268176 , 0.325391 , 0.388185 , 0.457527 , 0.534396 , 0.619668 , 0.713949 , 0.817345 , 0.92918 , 1.04776 , 1.17033 , 1.29329 , 1.41281 , 1.52548 , 1.62882 , 1.72149 , 1.80313 , 1.87409 , 1.93513 , 1.98717 , 2.03117 , 2.06799 , 2.09843 , 2.12317 , 2.14281 , 2.15783 , 2.16865 , 2.17555 , 2.17873 , 2.17824 , 2.17404 , 2.16593 , 2.1536 , 2.13666 , 2.11457 , 2.08674 , 2.05244 , 2.01084 , 1.96099 , 1.90182 , 1.83227 , 1.75144 , 1.65892 , 1.55521 , 1.44203 , 1.32244 , 1.20041 , 1.08005 , 0.964769 , 0.856779 , 0.757052 , 0.665618 , 0.581959 , 0.113151 , 0.0512128 , -0.0109668 , -0.0755113 , -0.144355 , -0.219343 , -0.302277 , -0.394848 , -0.498393 , -0.613456 , -0.73923 , -0.873086 , -1.01055 , -1.14601 , -1.2739 , -1.38987 , -1.4914 , -1.5777 , -1.64924 , -1.70725 , -1.75322 , -1.78867 , -1.81492 , -1.83307 , -1.84398 , -1.84826 , -1.8463 , -1.8382 , -1.82367 , -1.80195 , -1.77169 , -1.73094 , -1.67731 , -1.60844 , -1.52275 , -1.42033 , -1.30364 , -1.17727 , -1.04674 , -0.916991 , -0.791486 , -0.67237 , -0.56104 , -0.458513 , -0.365376 , -0.281543 , -0.206106 , -0.137439 , -0.0734307 , -0.0117341 , 0.050077 , 0.11446 , 0.183928 , 0.261082 , 0.348541 , 0.448632 , 0.562746 , 0.690339 , 0.827981 , 0.969217 , 1.10584 , 1.2302 , 1.3371 , 1.42443 , 1.49268 , 1.54403 , 1.5815 , 1.6083 , 1.62738 , 1.6411 , 1.65124 , 1.65897 , 1.66507 , 1.67001 , 1.6741 , 1.6775 , 1.6802 , 1.68206 , 1.68268 , 1.68149 , 1.6777 , 1.67028 , 1.65792 , 1.63896 , 1.61139 , 1.57285 , 1.52086 , 1.45317 , 1.3683 , 1.26611 , 1.14838 , 1.0191 , 0.88423 , 0.750706 , 0.624849 , 0.510926 , 0.410562 , 0.323062 , 0.246243 , 0.177253 , -0.53382 , -0.603663 , -0.667774 , -0.729343 , -0.792037 , -0.860242 , -0.939501 , -1.03733 , -1.16482 , -1.33945 , -1.58719 , -1.93088 , -2.34005 , -2.71239 , -2.98435 , 3.11561 , 2.99123 , 2.90355 , 2.83919 , 2.79052 , 2.75323 , 2.72487 , 2.70388 , 2.68913 , 2.6797 , 2.67489 , 2.67432 , 2.67809 , 2.68688 , 2.70199 , 2.7253 , 2.7591 , 2.80621 , 2.87027 , 2.95712 , 3.07758 , -3.03112 , -2.77042 , -2.40298 , -1.98992 , -1.64721 , -1.40819 , -1.24319 , -1.12174 , -1.02543 , -0.944371 , -0.873065 , -0.807827 , -0.745419 , -0.682525 , -0.615684 , -0.541365 , -0.45613 , -0.356912 , -0.241574 , -0.109765 , 0.0360454 , 0.189954 , 0.343271 , 0.486807 , 0.613729 , 0.721195 , 0.809871 , 0.882084 , 0.940012 , 0.984859 , 1.01711 , 1.03739 , 1.04734 , 1.04986 , 1.04863 , 1.04702 , 1.04728 , 1.0502 , 1.05538 , 1.06184 , 1.06842 , 1.07416 , 1.07849 , 1.08116 , 1.08202 , 1.08073 , 1.07644 , 1.06763 , 1.05217 , 1.02759 , 0.9912 , 0.940106 , 0.871294 , 0.782139 , 0.671712 , 0.542401 , 0.400377 , 0.253823 , 0.110067 , -0.0259942 , -0.151599 , -0.265266 , -0.366426 , -0.455478 , -1.12133 , -1.25416 , -1.38816 , -1.52404 , -1.66382 , -1.80981 , -1.96393 , -2.12816 , -2.30611 , -2.50553 , -2.74036 , -3.02853 , 2.90717 , 2.53368 , 2.1959 , 1.92895 , 1.73048 , 1.58542 , 1.47972 , 1.40257 , 1.34582 , 1.30354 , 1.27183 , 1.24853 , 1.23269 , 1.22406 , 1.22287 , 1.2298 , 1.24611 , 1.2736 , 1.31438 , 1.37065 , 1.4452 , 1.54349 , 1.6771 , 1.86644 , 2.13477 , 2.47896 , 2.84223 , -3.11726 , -2.84254 , -2.60596 , -2.4014 , -2.22405 , -2.06545 , -1.91591 , -1.76918 , -1.62462 , -1.48523 , -1.35314 , -1.22706 , -1.10315 , -0.977504 , -0.847946 , -0.7145 , -0.578714 , -0.442385 , -0.3065 , -0.171283 , -0.0375755 , 0.0916156 , 0.21103 , 0.31496 , 0.399987 , 0.467006 , 0.520872 , 0.567342 , 0.609573 , 0.647121 , 0.677727 , 0.700043 , 0.71527 , 0.726627 , 0.73716 , 0.74773 , 0.756787 , 0.761831 , 0.761243 , 0.755187 , 0.745051 , 0.732029 , 0.71592 , 0.695052 , 0.667245 , 0.630811 , 0.584647 , 0.527635 , 0.458429 , 0.376086 , 0.280755 , 0.173295 , 0.0544172 , -0.074878 , -0.211481 , -0.349681 , -0.48395 , -0.612431 , -0.737362 , -0.862509 , -0.990408 , -1.69338 , -1.85103 , -2.03047 , -2.23497 , -2.4649 , -2.71541 , -2.97405 , 3.05993 , 2.83377 , 2.63192 , 2.44277 , 2.24782 , 2.02695 , 1.75912 , 1.42564 , 1.03577 , 0.65537 , 0.352224 , 0.135113 , -0.0177395 , -0.126243 , -0.202754 , -0.255155 , -0.289502 , -0.310669 , -0.32172 , -0.323043 , -0.312065 , -0.284066 , -0.234121 , -0.158989 , -0.0557932 , 0.0853544 , 0.291358 , 0.598242 , 0.988017 , 1.36143 , 1.66958 , 1.93096 , 2.15643 , 2.3461 , 2.51646 , 2.70165 , 2.92878 , -3.09196 , -2.82705 , -2.58162 , -2.35778 , -2.15474 , -1.97483 , -1.81721 , -1.67502 , -1.53923 , -1.40254 , -1.26029 , -1.10995 , -0.951173 , -0.787131 , -0.62522 , -0.475023 , -0.343419 , -0.230034 , -0.128185 , -0.0319389 , 0.058213 , 0.13701 , 0.200911 , 0.253007 , 0.301219 , 0.351312 , 0.402631 , 0.450121 , 0.488768 , 0.51614 , 0.532104 , 0.537351 , 0.532956 , 0.520825 , 0.503316 , 0.481435 , 0.453581 , 0.417028 , 0.371147 , 0.318394 , 0.261509 , 0.200403 , 0.132443 , 0.0549918 , -0.0337432 , -0.135963 , -0.253678 , -0.385484 , -0.526313 , -0.672418 , -0.823801 , -0.979576 , -1.1339 , -1.27994 , -1.41699 , -1.55158 , -2.38718 , -2.57003 , -2.77141 , -2.99685 , 3.03853 , 2.77204 , 2.48955 , 2.19599 , 1.90324 , 1.62618 , 1.37354 , 1.14657 , 0.940094 , 0.736988 , 0.499329 , 0.164444 , -0.322825 , -0.868074 , -1.27654 , -1.52716 , -1.67949 , -1.77706 , -1.84317 , -1.8889 , -1.9189 , -1.93438 , -1.93412 , -1.91511 , -1.87408 , -1.80911 , -1.71651 , -1.57799 , -1.34014 , -0.922608 , -0.37914 , 0.0750939 , 0.40422 , 0.65081 , 0.843827 , 1.02627 , 1.24237 , 1.50128 , 1.77349 , 2.04715 , 2.34054 , 2.64258 , 2.91454 , -3.13575 , -2.92439 , -2.72169 , -2.5277 , -2.34438 , -2.16664 , -1.98539 , -1.79532 , -1.59955 , -1.40806 , -1.23143 , -1.07341 , -0.928074 , -0.786751 , -0.649148 , -0.524273 , -0.419505 , -0.330061 , -0.243549 , -0.155558 , -0.0736174 , -0.00590443 , 0.0496945 , 0.103115 , 0.159264 , 0.212121 , 0.250886 , 0.268994 , 0.268558 , 0.257933 , 0.244859 , 0.230421 , 0.208347 , 0.171256 , 0.118798 , 0.058042 , -0.00523891 , -0.0726348 , -0.148607 , -0.234059 , -0.328038 , -0.431557 , -0.545298 , -0.668035 , -0.801348 , -0.949648 , -1.11155 , -1.27923 , -1.45085 , -1.6333 , -1.82824 , -2.02425 , -2.20974 , -3.05577 , 2.99025 , 2.73947 , 2.47532 , 2.21063 , 1.95535 , 1.70294 , 1.43013 , 1.11207 , 0.74983 , 0.386119 , 0.0686247 , -0.198605 , -0.442382 , -0.688962 , -0.969281 , -1.35401 , -1.93609 , -2.55537 , -2.95878 , 3.0928 , 2.95547 , 2.87277 , 2.82241 , 2.79138 , 2.77481 , 2.77431 , 2.79489 , 2.84145 , 2.91888 , 3.04257 , -3.02071 , -2.60518 , -1.99637 , -1.45016 , -1.05811 , -0.782002 , -0.562393 , -0.330167 , -0.0557355 , 0.244518 , 0.582425 , 0.957263 , 1.28875 , 1.55004 , 1.79615 , 2.05732 , 2.31724 , 2.56792 , 2.81928 , 3.06889 , -2.98138 , -2.7661 , -2.55551 , -2.34302 , -2.13444 , -1.93675 , -1.74671 , -1.55718 , -1.37249 , -1.20712 , -1.0665 , -0.937199 , -0.807145 , -0.684613 , -0.584059 , -0.501813 , -0.419871 , -0.333209 , -0.253518 , -0.188909 , -0.133406 , -0.0804276 , -0.0361495 , -0.0116083 , -0.00872023 , -0.0179946 , -0.0302645 , -0.0481097 , -0.0811496 , -0.131596 , -0.190657 , -0.251777 , -0.319908 , -0.400956 , -0.491695 , -0.588175 , -0.693384 , -0.810447 , -0.940041 , -1.08488 , -1.24403 , -1.41079 , -1.58617 , -1.7768 , -1.97684 , -2.17521 , -2.37651 , -2.59279 , -2.82267 , 2.59385 , 2.32072 , 2.03655 , 1.7318 , 1.41616 , 1.11335 , 0.825835 , 0.5365 , 0.234837 , -0.0879501 , -0.458182 , -0.886781 , -1.31514 , -1.6761 , -1.96112 , -2.2134 , -2.51023 , -2.96395 , 2.61318 , 1.96822 , 1.61856 , 1.43376 , 1.32322 , 1.25314 , 1.21087 , 1.19159 , 1.19605 , 1.22766 , 1.28801 , 1.38348 , 1.55459 , 1.91458 , 2.55577 , -3.05879 , -2.59537 , -2.31286 , -2.08564 , -1.79606 , -1.44312 , -1.05307 , -0.621705 , -0.252507 , 0.0509014 , 0.367569 , 0.673551 , 0.940991 , 1.22328 , 1.54218 , 1.84992 , 2.1314 , 2.40356 , 2.66664 , 2.91267 , -3.13882 , -2.91396 , -2.68854 , -2.45684 , -2.22499 , -2.01231 , -1.82891 , -1.6581 , -1.48453 , -1.32301 , -1.19088 , -1.0725 , -0.947851 , -0.831066 , -0.741491 , -0.665323 , -0.580784 , -0.495767 , -0.429482 , -0.382545 , -0.344015 , -0.313473 , -0.297812 , -0.299444 , -0.317114 , -0.350756 , -0.396896 , -0.446287 , -0.496954 , -0.56065 , -0.643643 , -0.736776 , -0.835137 , -0.944933 , -1.06703 , -1.1999 , -1.34645 , -1.50379 , -1.67236 , -1.86196 , -2.06633 , -2.27258 , -2.49059 , -2.72498 , -2.95682 , 3.09932 , 2.85705 , 1.93617 , 1.61986 , 1.30933 , 1.00947 , 0.690003 , 0.327404 , -0.0550352 , -0.412471 , -0.726539 , -1.02835 , -1.38239 , -1.8248 , -2.32793 , -2.80459 , 3.08825 , 2.76565 , 2.47328 , 2.1434 , 1.59434 , 0.744334 , 0.166356 , -0.104274 , -0.243078 , -0.325244 , -0.377073 , -0.401002 , -0.392475 , -0.352044 , -0.282605 , -0.160915 , 0.105469 , 0.686298 , 1.49506 , 2.06002 , 2.37235 , 2.63939 , 2.97142 , -2.94184 , -2.48951 , -1.97916 , -1.57019 , -1.21403 , -0.900358 , -0.603895 , -0.237951 , 0.147143 , 0.474277 , 0.789163 , 1.10919 , 1.41473 , 1.71813 , 2.02005 , 2.29545 , 2.54288 , 2.79164 , 3.05646 , -2.96538 , -2.73124 , -2.51675 , -2.29956 , -2.09 , -1.91267 , -1.75179 , -1.58248 , -1.42849 , -1.31068 , -1.19724 , -1.0735 , -0.970267 , -0.89256 , -0.811181 , -0.729498 , -0.674571 , -0.644568 , -0.616293 , -0.587955 , -0.580308 , -0.606357 , -0.653849 , -0.700322 , -0.741779 , -0.797164 , -0.876554 , -0.968313 , -1.06703 , -1.18012 , -1.30577 , -1.44265 , -1.59415 , -1.75653 , -1.93492 , -2.1304 , -2.33019 , -2.54467 , -2.78022 , -3.01766 , 3.01817 , 2.74958 , 2.48519 , 2.22516 , 2.35036 , 2.34806 , 2.34036 , 2.3272 , 2.30845 , 2.28398 , 2.25362 , 2.21725 , 2.17475 , 2.12611 , 2.07147 , 2.01116 , 1.94575 , 1.87607 , 1.80325 , 1.72857 , 1.65347 , 1.57934 , 1.50743 , 1.43878 , 1.37414 , 1.31397 , 1.25847 , 1.20764 , 1.16134 , 1.11931 , 1.08125 , 1.04682 , 1.01569 , 0.987544 , 0.962077 , 0.939025 , 0.918149 , 0.899241 , 0.882121 , 0.866636 , 0.852655 , 0.840067 , 0.828776 , 0.818696 , 0.809755 , 0.801886 , 0.795027 , 0.78912 , 0.784115 , 0.779962 , 0.776615 , 0.774038 , 0.772194 , 0.771057 , 0.770605 , 0.770822 , 0.771701 , 0.773242 , 0.775451 , 0.778339 , 0.781928 , 0.786246 , 0.791329 , 0.797223 , 0.803984 , 0.811679 , 0.820389 , 0.830208 , 0.841248 , 0.853635 , 0.867519 , 0.883066 , 0.900464 , 0.91993 , 0.941704 , 0.966053 , 0.99327 , 1.02368 , 1.05762 , 1.09544 , 1.13751 , 1.18414 , 1.23559 , 1.29203 , 1.35343 , 1.41954 , 1.48986 , 1.56354 , 1.63947 , 1.71627 , 1.79249 , 1.86665 , 1.93743 , 2.00375 , 2.06482 , 2.1201 , 2.16932 , 2.2124 , 2.24938 , 2.28036 , 2.30549 , 2.32494 , 2.33884 , 2.3473 , 2.21646 , 2.2159 , 2.21148 , 2.20308 , 2.19042 , 2.17311 , 2.15061 , 2.12221 , 2.08705 , 2.04411 , 1.99231 , 1.93054 , 1.85784 , 1.77358 , 1.67769 , 1.57099 , 1.45533 , 1.33365 , 1.20962 , 1.08703 , 0.969165 , 0.858327 , 0.755717 , 0.661605 , 0.575618 , 0.497052 , 0.425095 , 0.358984 , 0.298061 , 0.241811 , 0.189816 , 0.141728 , 0.0972281 , 0.0559993 , 0.0177166 , -0.0179433 , -0.0512766 , -0.082524 , -0.111841 , -0.139286 , -0.164813 , -0.188298 , -0.209574 , -0.228465 , -0.244833 , -0.258597 , -0.26975 , -0.278348 , -0.284494 , -0.288319 , -0.289955 , -0.289514 , -0.287076 , -0.282674 , -0.276293 , -0.26787 , -0.257306 , -0.244475 , -0.229243 , -0.21149 , -0.191126 , -0.168102 , -0.14242 , -0.114125 , -0.083293 , -0.0500073 , -0.0143319 , 0.0237211 , 0.0642202 , 0.107343 , 0.1534 , 0.202853 , 0.256331 , 0.314621 , 0.378664 , 0.449504 , 0.528199 , 0.61568 , 0.712527 , 0.818714 , 0.933347 , 1.05451 , 1.17935 , 1.30439 , 1.42609 , 1.54131 , 1.64768 , 1.74372 , 1.82878 , 1.90287 , 1.9665 , 2.0205 , 2.06584 , 2.10354 , 2.13455 , 2.1597 , 2.17968 , 2.19503 , 2.20612 , 2.21322 , 1.7374 , 1.74038 , 1.74302 , 1.74475 , 1.74459 , 1.74121 , 1.73304 , 1.71836 , 1.69546 , 1.66265 , 1.61829 , 1.56067 , 1.48809 , 1.39913 , 1.2934 , 1.17252 , 1.04083 , 0.904903 , 0.771892 , 0.647412 , 0.534455 , 0.433526 , 0.343449 , 0.262178 , 0.187369 , 0.116706 , 0.0480549 , -0.0204826 , -0.0906214 , -0.16402 , -0.242454 , -0.327971 , -0.422863 , -0.529337 , -0.648686 , -0.780093 , -0.919685 , -1.0609 , -1.19653 , -1.32117 , -1.43241 , -1.53023 , -1.61569 , -1.68987 , -1.75345 , -1.80671 , -1.84979 , -1.88285 , -1.90623 , -1.92037 , -1.92572 , -1.92263 , -1.91126 , -1.89151 , -1.86303 , -1.8252 , -1.7772 , -1.71798 , -1.64645 , -1.56161 , -1.46296 , -1.35101 , -1.22778 , -1.09694 , -0.963357 , -0.832116 , -0.707381 , -0.591791 , -0.486415 , -0.391075 , -0.304741 , -0.22587 , -0.152616 , -0.0829801 , -0.0149169 , 0.053598 , 0.124532 , 0.199824 , 0.281455 , 0.37153 , 0.472218 , 0.585346 , 0.711456 , 0.848444 , 0.990669 , 1.12979 , 1.25743 , 1.36798 , 1.45944 , 1.53258 , 1.58953 , 1.6328 , 1.66482 , 1.68783 , 1.70387 , 1.71476 , 1.72211 , 1.72721 , 1.73105 , 1.73432 , 1.14352 , 1.15068 , 1.1578 , 1.16324 , 1.16554 , 1.16383 , 1.15801 , 1.1484 , 1.13506 , 1.11688 , 1.09104 , 1.05324 , 0.998764 , 0.923886 , 0.826907 , 0.708698 , 0.572901 , 0.425761 , 0.275207 , 0.128983 , -0.00732148 , -0.131157 , -0.242533 , -0.342923 , -0.434287 , -0.518537 , -0.597374 , -0.672399 , -0.745411 , -0.818844 , -0.896285 , -0.983093 , -1.08716 , -1.21996 , -1.39792 , -1.64288 , -1.97274 , -2.36602 , -2.74065 , -3.02926 , 3.05416 , 2.91907 , 2.82632 , 2.76052 , 2.71187 , 2.67446 , 2.64506 , 2.62214 , 2.60517 , 2.59399 , 2.5885 , 2.58856 , 2.59404 , 2.60503 , 2.62208 , 2.64647 , 2.68047 , 2.7276 , 2.79312 , 2.88509 , 3.01634 , -3.07586 , -2.79858 , -2.42874 , -2.03171 , -1.70256 , -1.46604 , -1.29783 , -1.1714 , -1.0692 , -0.980912 , -0.900558 , -0.824528 , -0.750404 , -0.676231 , -0.600064 , -0.519614 , -0.431991 , -0.333665 , -0.220956 , -0.0913593 , 0.054381 , 0.210724 , 0.368243 , 0.517186 , 0.65082 , 0.766105 , 0.862473 , 0.940594 , 1.00186 , 1.0482 , 1.0818 , 1.10484 , 1.11934 , 1.12724 , 1.13061 , 1.13157 , 1.13215 , 1.1339 , 1.13767 , 0.862437 , 0.867743 , 0.867776 , 0.8613 , 0.848997 , 0.832879 , 0.814446 , 0.792743 , 0.763976 , 0.723507 , 0.668756 , 0.600426 , 0.521166 , 0.433029 , 0.335647 , 0.226651 , 0.104309 , -0.0298821 , -0.170996 , -0.31284 , -0.450704 , -0.583085 , -0.71173 , -0.840202 , -0.971804 , -1.10807 , -1.24867 , -1.39274 , -1.54046 , -1.69342 , -1.853 , -2.01819 , -2.18636 , -2.35914 , -2.5493 , -2.78238 , -3.08664 , 2.82289 , 2.43913 , 2.10904 , 1.84827 , 1.64805 , 1.49807 , 1.38829 , 1.30838 , 1.24923 , 1.20423 , 1.16958 , 1.14357 , 1.12581 , 1.11648 , 1.11594 , 1.12479 , 1.14386 , 1.17433 , 1.21773 , 1.27626 , 1.35353 , 1.45595 , 1.59449 , 1.78563 , 2.04766 , 2.38402 , 2.75832 , 3.10831 , -2.88406 , -2.6496 , -2.45456 , -2.2821 , -2.11946 , -1.95924 , -1.7994 , -1.64199 , -1.49061 , -1.34774 , -1.21312 , -1.08361 , -0.954619 , -0.822317 , -0.685551 , -0.545947 , -0.405843 , -0.266121 , -0.12652 , 0.0117838 , 0.144474 , 0.265821 , 0.3721 , 0.462912 , 0.539848 , 0.604838 , 0.659573 , 0.705577 , 0.74408 , 0.775739 , 0.800757 , 0.819574 , 0.833493 , 0.844485 , 0.854125 , 0.665112 , 0.659678 , 0.645751 , 0.626318 , 0.603711 , 0.577186 , 0.542173 , 0.493874 , 0.432423 , 0.363667 , 0.294119 , 0.224268 , 0.146592 , 0.051661 , -0.0636519 , -0.194657 , -0.333816 , -0.476412 , -0.622285 , -0.772835 , -0.927399 , -1.08317 , -1.23776 , -1.39107 , -1.54508 , -1.70281 , -1.86855 , -2.0498 , -2.25739 , -2.49792 , -2.76003 , -3.01833 , 3.0223 , 2.78799 , 2.5579 , 2.342 , 2.14468 , 1.94282 , 1.68902 , 1.34624 , 0.94508 , 0.566973 , 0.257454 , 0.0244333 , -0.13827 , -0.246891 , -0.320675 , -0.374076 , -0.414083 , -0.4418 , -0.455655 , -0.454064 , -0.436475 , -0.402826 , -0.352003 , -0.279757 , -0.176493 , -0.0255183 , 0.196544 , 0.509721 , 0.896874 , 1.28194 , 1.59923 , 1.84681 , 2.05325 , 2.24588 , 2.44316 , 2.65618 , 2.89073 , -3.13717 , -2.87175 , -2.61424 , -2.38002 , -2.17416 , -1.99254 , -1.82672 , -1.66853 , -1.51325 , -1.36012 , -1.20914 , -1.05752 , -0.900662 , -0.737881 , -0.575811 , -0.423334 , -0.284283 , -0.157819 , -0.0432146 , 0.0590368 , 0.149528 , 0.230282 , 0.302663 , 0.366832 , 0.423686 , 0.475839 , 0.525571 , 0.572244 , 0.612394 , 0.642221 , 0.659756 , 0.41738 , 0.404955 , 0.388737 , 0.373375 , 0.353542 , 0.318068 , 0.261895 , 0.192652 , 0.122797 , 0.0548222 , -0.0226999 , -0.119933 , -0.232126 , -0.346661 , -0.45999 , -0.583077 , -0.727189 , -0.889775 , -1.05879 , -1.22606 , -1.39467 , -1.57454 , -1.77179 , -1.98273 , -2.19649 , -2.40356 , -2.60428 , -2.80965 , -3.03192 , 3.01086 , 2.75552 , 2.47098 , 2.14066 , 1.80035 , 1.51457 , 1.28553 , 1.07101 , 0.850613 , 0.62993 , 0.387672 , 0.0473739 , -0.455033 , -0.983129 , -1.37677 , -1.64533 , -1.82171 , -1.93072 , -1.99602 , -2.03623 , -2.0618 , -2.07586 , -2.07726 , -2.06269 , -2.02703 , -1.96272 , -1.85864 , -1.69633 , -1.44039 , -1.03779 , -0.511279 , -0.0420495 , 0.292646 , 0.543307 , 0.754435 , 0.950858 , 1.15453 , 1.38946 , 1.67098 , 1.99182 , 2.32146 , 2.62566 , 2.88837 , 3.11507 , -2.96097 , -2.75688 , -2.54674 , -2.33282 , -2.12477 , -1.92762 , -1.73609 , -1.54399 , -1.35526 , -1.17837 , -1.01312 , -0.853969 , -0.701907 , -0.56317 , -0.438472 , -0.323261 , -0.215963 , -0.118092 , -0.0288117 , 0.053889 , 0.128777 , 0.193659 , 0.250725 , 0.304622 , 0.35512 , 0.395041 , 0.416306 , 0.160239 , 0.150387 , 0.136379 , 0.116877 , 0.0812257 , 0.0269377 , -0.0341184 , -0.0928323 , -0.159978 , -0.251328 , -0.36054 , -0.467629 , -0.570969 , -0.692184 , -0.842039 , -1.00584 , -1.16793 , -1.33286 , -1.51417 , -1.71187 , -1.91512 , -2.11895 , -2.33036 , -2.55848 , -2.80307 , -3.0581 , 2.96132 , 2.69111 , 2.42989 , 2.18696 , 1.93689 , 1.64543 , 1.32326 , 1.00524 , 0.668218 , 0.294052 , -0.0434987 , -0.317323 , -0.570829 , -0.822276 , -1.08364 , -1.45341 , -2.07605 , -2.71857 , -3.11251 , 2.93689 , 2.79369 , 2.70784 , 2.6559 , 2.62366 , 2.60658 , 2.60683 , 2.62875 , 2.67655 , 2.75662 , 2.88596 , 3.10865 , -2.76826 , -2.13618 , -1.54973 , -1.17277 , -0.915202 , -0.690459 , -0.448315 , -0.16759 , 0.153429 , 0.500728 , 0.849073 , 1.18145 , 1.49278 , 1.77768 , 2.03398 , 2.27395 , 2.52091 , 2.78847 , 3.06352 , -2.96258 , -2.73023 , -2.50846 , -2.28706 , -2.073 , -1.87183 , -1.67532 , -1.47978 , -1.29549 , -1.12707 , -0.968365 , -0.819222 , -0.684877 , -0.564 , -0.453028 , -0.352442 , -0.261073 , -0.175421 , -0.0967224 , -0.0286864 , 0.0304795 , 0.0846172 , 0.129671 , 0.155976 , -0.107831 , -0.108628 , -0.12873 , -0.169457 , -0.222187 , -0.269944 , -0.312087 , -0.374853 , -0.471826 , -0.57945 , -0.678681 , -0.79113 , -0.935117 , -1.08905 , -1.23449 , -1.39265 , -1.58385 , -1.79138 , -1.99214 , -2.19117 , -2.40915 , -2.65175 , -2.90932 , 3.11284 , 2.85405 , 2.59183 , 2.31088 , 2.00742 , 1.70034 , 1.38856 , 1.05317 , 0.727861 , 0.450884 , 0.164872 , -0.194761 , -0.594875 , -1.02541 , -1.45885 , -1.79315 , -2.07373 , -2.37113 , -2.68393 , -3.10438 , 2.45823 , 1.80903 , 1.4547 , 1.248 , 1.12736 , 1.06129 , 1.02552 , 1.00882 , 1.01084 , 1.03584 , 1.09213 , 1.19773 , 1.39081 , 1.75462 , 2.4 , 3.08355 , -2.76929 , -2.47031 , -2.19844 , -1.91376 , -1.58606 , -1.19046 , -0.75837 , -0.358833 , -0.0194039 , 0.280558 , 0.575347 , 0.881847 , 1.19595 , 1.50964 , 1.82031 , 2.12139 , 2.40144 , 2.66366 , 2.92585 , -3.0914 , -2.83989 , -2.60689 , -2.37674 , -2.15056 , -1.94033 , -1.73997 , -1.54674 , -1.37135 , -1.21137 , -1.05905 , -0.918999 , -0.792243 , -0.67436 , -0.569427 , -0.479368 , -0.396384 , -0.318604 , -0.252731 , -0.200204 , -0.157125 , -0.124721 , -0.384082 , -0.37488 , -0.403733 , -0.45967 , -0.507524 , -0.539878 , -0.594033 , -0.685349 , -0.782015 , -0.876336 , -1.00326 , -1.15561 , -1.29687 , -1.44573 , -1.6344 , -1.83561 , -2.02453 , -2.22918 , -2.46347 , -2.69984 , -2.92914 , 3.10259 , 2.81172 , 2.50387 , 2.20658 , 1.91628 , 1.61464 , 1.30774 , 0.994035 , 0.629342 , 0.226839 , -0.128737 , -0.467261 , -0.828742 , -1.16666 , -1.52838 , -1.96639 , -2.437 , -2.95461 , 2.90556 , 2.61338 , 2.31357 , 1.95273 , 1.40777 , 0.538291 , -0.0249285 , -0.29774 , -0.456209 , -0.542789 , -0.58334 , -0.59996 , -0.59877 , -0.569615 , -0.495644 , -0.354443 , -0.0858337 , 0.479683 , 1.30832 , 1.86936 , 2.21292 , 2.48735 , 2.78908 , -3.09099 , -2.59848 , -2.12062 , -1.71609 , -1.35425 , -1.00376 , -0.659106 , -0.312608 , 0.0475374 , 0.41444 , 0.771455 , 1.10665 , 1.41266 , 1.70048 , 1.99841 , 2.31084 , 2.60617 , 2.877 , -3.1398 , -2.88535 , -2.64974 , -2.41595 , -2.19171 , -1.98789 , -1.78979 , -1.6025 , -1.43545 , -1.27788 , -1.13346 , -1.00714 , -0.887722 , -0.778127 , -0.687828 , -0.608782 , -0.535951 , -0.479483 , -0.442234 , -0.412053 , 1.11494 , 1.15647 , 1.20232 , 1.25278 , 1.30805 , 1.36822 , 1.43314 , 1.50242 , 1.57536 , 1.65089 , 1.72768 , 1.80419 , 1.87881 , 1.95004 , 2.01662 , 2.07762 , 2.13243 , 2.18077 , 2.22258 , 2.25801 , 2.28727 , 2.31064 , 2.32841 , 2.3408 , 2.34801 , 2.35015 , 2.34727 , 2.33932 , 2.32621 , 2.30776 , 2.28372 , 2.25384 , 2.21783 , 2.17545 , 2.12654 , 2.0711 , 2.00937 , 1.94189 , 1.86957 , 1.79362 , 1.71557 , 1.63705 , 1.5597 , 1.48496 , 1.41397 , 1.34755 , 1.28614 , 1.22991 , 1.1788 , 1.13261 , 1.09103 , 1.05371 , 1.02027 , 0.990364 , 0.963633 , 0.939755 , 0.918427 , 0.899374 , 0.882347 , 0.86712 , 0.853493 , 0.841292 , 0.830363 , 0.820577 , 0.811823 , 0.804016 , 0.797084 , 0.790974 , 0.785644 , 0.781067 , 0.777224 , 0.774102 , 0.771695 , 0.770002 , 0.769026 , 0.768773 , 0.769253 , 0.77048 , 0.77247 , 0.775242 , 0.77882 , 0.783228 , 0.788496 , 0.794655 , 0.801741 , 0.809794 , 0.818861 , 0.828998 , 0.840273 , 0.852763 , 0.866569 , 0.881805 , 0.898611 , 0.91715 , 0.937613 , 0.960217 , 0.985209 , 1.01286 , 1.04349 , 1.0774 , 0.492008 , 0.569441 , 0.655754 , 0.75204 , 0.858638 , 0.974624 , 1.09748 , 1.22323 , 1.34728 , 1.46548 , 1.57499 , 1.67452 , 1.76396 , 1.84384 , 1.91488 , 1.97765 , 2.03255 , 2.07985 , 2.11976 , 2.15262 , 2.17882 , 2.1989 , 2.21344 , 2.22298 , 2.22796 , 2.22869 , 2.22525 , 2.21756 , 2.20531 , 2.18808 , 2.16534 , 2.13651 , 2.10111 , 2.0587 , 2.00904 , 1.95194 , 1.88727 , 1.8148 , 1.73408 , 1.64447 , 1.54529 , 1.43629 , 1.31821 , 1.19327 , 1.0653 , 0.939073 , 0.819142 , 0.708817 , 0.609695 , 0.521827 , 0.444222 , 0.375367 , 0.313609 , 0.257398 , 0.205419 , 0.156648 , 0.110387 , 0.0662584 , 0.0241778 , -0.0157108 , -0.0531186 , -0.0877131 , -0.119213 , -0.147459 , -0.172444 , -0.194305 , -0.213288 , -0.229689 , -0.243803 , -0.255872 , -0.266061 , -0.274443 , -0.28101 , -0.285685 , -0.288357 , -0.288901 , -0.2872 , -0.283172 , -0.276773 , -0.268 , -0.256887 , -0.243492 , -0.227879 , -0.210095 , -0.190156 , -0.168031 , -0.143641 , -0.116873 , -0.0875938 , -0.0556841 , -0.0210564 , 0.0163289 , 0.0564668 , 0.0993323 , 0.144918 , 0.193296 , 0.244674 , 0.299455 , 0.358284 , 0.422077 , 0.116842 , 0.186596 , 0.260847 , 0.341563 , 0.430458 , 0.529052 , 0.638697 , 0.760253 , 0.893084 , 1.0335 , 1.17394 , 1.30453 , 1.41679 , 1.50662 , 1.57462 , 1.62431 , 1.6602 , 1.68648 , 1.70635 , 1.72184 , 1.73404 , 1.74335 , 1.74986 , 1.75356 , 1.7545 , 1.75274 , 1.74837 , 1.74148 , 1.73223 , 1.72085 , 1.70762 , 1.69272 , 1.67584 , 1.65584 , 1.6303 , 1.59545 , 1.5464 , 1.47789 , 1.38571 , 1.26884 , 1.13159 , 0.983688 , 0.836832 , 0.699861 , 0.576534 , 0.466646 , 0.368299 , 0.27945 , 0.198413 , 0.123698 , 0.0536801 , -0.013626 , -0.0805882 , -0.149799 , -0.223788 , -0.304663 , -0.393779 , -0.491563 , -0.597646 , -0.711259 , -0.831619 , -0.957869 , -1.08843 , -1.2201 , -1.34792 , -1.46617 , -1.57022 , -1.65777 , -1.72891 , -1.78529 , -1.82914 , -1.86257 , -1.88719 , -1.90405 , -1.91367 , -1.91615 , -1.91135 , -1.89894 , -1.87844 , -1.84926 , -1.81062 , -1.76155 , -1.70081 , -1.62701 , -1.53888 , -1.43573 , -1.31819 , -1.18883 , -1.05219 , -0.913925 , -0.779385 , -0.652423 , -0.535085 , -0.427952 , -0.330695 , -0.242456 , -0.162018 , -0.0878422 , -0.0180784 , 0.0493884 , -0.519151 , -0.439054 , -0.351769 , -0.254018 , -0.14226 , -0.0137914 , 0.131069 , 0.286692 , 0.442516 , 0.587492 , 0.715049 , 0.824027 , 0.91595 , 0.992103 , 1.05258 , 1.09721 , 1.12694 , 1.14438 , 1.1532 , 1.15717 , 1.15907 , 1.16029 , 1.16082 , 1.1599 , 1.15674 , 1.15114 , 1.14372 , 1.1359 , 1.12951 , 1.12602 , 1.12558 , 1.12625 , 1.12408 , 1.11411 , 1.09186 , 1.05448 , 1.0014 , 0.934042 , 0.854438 , 0.762978 , 0.656966 , 0.531855 , 0.385492 , 0.222924 , 0.0570153 , -0.0976593 , -0.232259 , -0.345629 , -0.441454 , -0.524937 , -0.600966 , -0.673549 , -0.745813 , -0.82007 , -0.897919 , -0.98061 , -1.07007 , -1.17086 , -1.29313 , -1.45671 , -1.69409 , -2.03636 , -2.45001 , -2.81966 , -3.08539 , 3.01638 , 2.88894 , 2.79648 , 2.72869 , 2.67948 , 2.64455 , 2.62049 , 2.60463 , 2.59508 , 2.59072 , 2.59107 , 2.5961 , 2.60611 , 2.62161 , 2.64334 , 2.67247 , 2.71088 , 2.76166 , 2.82974 , 2.92303 , 3.05441 , -3.03861 , -2.76169 , -2.38723 , -1.97742 , -1.63413 , -1.38807 , -1.21491 , -1.08655 , -0.984133 , -0.89642 , -0.816881 , -0.741677 , -0.668318 , -0.59481 , -1.11001 , -0.979671 , -0.850959 , -0.719408 , -0.583383 , -0.445124 , -0.308278 , -0.17376 , -0.0389319 , 0.097799 , 0.231383 , 0.351365 , 0.450486 , 0.530339 , 0.598267 , 0.660368 , 0.717085 , 0.764471 , 0.799273 , 0.822381 , 0.837912 , 0.849909 , 0.859853 , 0.86651 , 0.86784 , 0.863105 , 0.8537 , 0.842339 , 0.831196 , 0.820047 , 0.805802 , 0.783952 , 0.751007 , 0.706428 , 0.653286 , 0.596371 , 0.53751 , 0.471769 , 0.389204 , 0.281531 , 0.149601 , 0.00565283 , -0.13544 , -0.268816 , -0.40096 , -0.539792 , -0.685406 , -0.830007 , -0.965888 , -1.09233 , -1.21578 , -1.34548 , -1.48798 , -1.64273 , -1.8022 , -1.9587 , -2.11234 , -2.27254 , -2.45273 , -2.66254 , -2.90496 , 3.0986 , 2.76969 , 2.39952 , 2.04976 , 1.77967 , 1.5901 , 1.45518 , 1.35387 , 1.2754 , 1.21541 , 1.17182 , 1.14257 , 1.12532 , 1.11799 , 1.11908 , 1.12782 , 1.14406 , 1.16828 , 1.20176 , 1.2469 , 1.3074 , 1.38849 , 1.49721 , 1.64359 , 1.84229 , 2.11105 , 2.45453 , 2.83493 , -3.09594 , -2.80298 , -2.56212 , -2.35743 , -2.17669 , -2.01086 , -1.85258 , -1.6968 , -1.54219 , -1.39096 , -1.24644 , -1.70008 , -1.54291 , -1.38949 , -1.23809 , -1.09024 , -0.944087 , -0.791502 , -0.626782 , -0.459568 , -0.309631 , -0.183826 , -0.0705368 , 0.0419131 , 0.148936 , 0.236899 , 0.303529 , 0.361285 , 0.422869 , 0.487923 , 0.544843 , 0.585095 , 0.610817 , 0.629268 , 0.644883 , 0.656772 , 0.661377 , 0.656037 , 0.640253 , 0.614613 , 0.579321 , 0.533867 , 0.478636 , 0.417452 , 0.35712 , 0.300358 , 0.239636 , 0.162109 , 0.061596 , -0.0528093 , -0.164862 , -0.273893 , -0.399439 , -0.558468 , -0.741527 , -0.91894 , -1.07443 , -1.21664 , -1.36066 , -1.51171 , -1.66643 , -1.8241 , -1.99145 , -2.17684 , -2.38297 , -2.60936 , -2.86012 , -3.13575 , 2.87231 , 2.63508 , 2.44119 , 2.26262 , 2.06751 , 1.84173 , 1.584 , 1.27785 , 0.900183 , 0.501784 , 0.186133 , -0.0268274 , -0.172291 , -0.277618 , -0.354187 , -0.406198 , -0.437016 , -0.450548 , -0.450267 , -0.438201 , -0.414494 , -0.377327 , -0.322844 , -0.244884 , -0.13425 , 0.0229131 , 0.246946 , 0.558912 , 0.948227 , 1.34245 , 1.67365 , 1.93775 , 2.15899 , 2.35929 , 2.55634 , 2.76692 , 3.00366 , -3.01771 , -2.74846 , -2.49246 , -2.25997 , -2.05235 , -1.86741 , -2.39573 , -2.18224 , -1.97779 , -1.7878 , -1.60322 , -1.41092 , -1.21588 , -1.03951 , -0.886266 , -0.736974 , -0.582504 , -0.44271 , -0.332301 , -0.234447 , -0.128467 , -0.0225931 , 0.0615113 , 0.125824 , 0.189904 , 0.259756 , 0.319571 , 0.355988 , 0.374373 , 0.388195 , 0.404038 , 0.416721 , 0.415388 , 0.394139 , 0.355946 , 0.307134 , 0.251504 , 0.190516 , 0.126 , 0.0575362 , -0.0214764 , -0.117674 , -0.224476 , -0.325772 , -0.424871 , -0.546408 , -0.701007 , -0.863511 , -1.00974 , -1.15898 , -1.34423 , -1.55963 , -1.76518 , -1.94493 , -2.12035 , -2.31671 , -2.53637 , -2.75881 , -2.96543 , 3.12076 , 2.90117 , 2.62318 , 2.29634 , 1.97832 , 1.68759 , 1.40556 , 1.14695 , 0.937109 , 0.75467 , 0.550469 , 0.289779 , -0.0456067 , -0.499113 , -1.04216 , -1.46238 , -1.70774 , -1.85575 , -1.95503 , -2.0221 , -2.06215 , -2.07953 , -2.07907 , -2.0643 , -2.03594 , -1.99119 , -1.92295 , -1.81861 , -1.65672 , -1.39876 , -0.987951 , -0.442984 , 0.0435643 , 0.384795 , 0.63687 , 0.851333 , 1.05753 , 1.2781 , 1.53045 , 1.81728 , 2.12715 , 2.4453 , 2.75195 , 3.0251 , -3.02357 , -2.81293 , -2.60704 , -3.0526 , -2.79528 , -2.5659 , -2.34763 , -2.12452 , -1.90951 , -1.71696 , -1.5286 , -1.32877 , -1.14471 , -0.994327 , -0.84873 , -0.694197 , -0.563109 , -0.463943 , -0.362924 , -0.250635 , -0.158332 , -0.0944562 , -0.0344322 , 0.0306728 , 0.0822135 , 0.111093 , 0.130876 , 0.153035 , 0.168522 , 0.15971 , 0.12516 , 0.0788986 , 0.0301183 , -0.0265051 , -0.0972269 , -0.175934 , -0.258628 , -0.351822 , -0.455262 , -0.560472 , -0.67718 , -0.821137 , -0.975237 , -1.11665 , -1.27328 , -1.47512 , -1.68921 , -1.87689 , -2.06707 , -2.29237 , -2.52658 , -2.73913 , -2.9541 , 3.07218 , 2.78675 , 2.52386 , 2.28562 , 2.03443 , 1.7621 , 1.4922 , 1.20619 , 0.856154 , 0.480003 , 0.147922 , -0.161696 , -0.452751 , -0.690921 , -0.907399 , -1.18207 , -1.57865 , -2.13592 , -2.75729 , 3.10584 , 2.88429 , 2.75809 , 2.67755 , 2.62854 , 2.60667 , 2.60681 , 2.62375 , 2.65605 , 2.70885 , 2.79464 , 2.93449 , -3.11544 , -2.70721 , -2.075 , -1.48241 , -1.09329 , -0.81443 , -0.571253 , -0.321576 , -0.0374626 , 0.289304 , 0.647464 , 1.01146 , 1.34749 , 1.64501 , 1.92107 , 2.18771 , 2.44368 , 2.69518 , 2.95797 , 2.58804 , 2.85526 , 3.12018 , -2.90014 , -2.65791 , -2.43633 , -2.20615 , -1.97324 , -1.77333 , -1.58795 , -1.39283 , -1.22365 , -1.08611 , -0.937675 , -0.790849 , -0.680876 , -0.578802 , -0.463811 , -0.373466 , -0.319518 , -0.270906 , -0.216639 , -0.171261 , -0.136688 , -0.108274 , -0.0993127 , -0.122457 , -0.163433 , -0.202802 , -0.248871 , -0.32019 , -0.402934 , -0.477846 , -0.561688 , -0.673154 , -0.793761 , -0.918629 , -1.06152 , -1.20912 , -1.36164 , -1.54689 , -1.74417 , -1.92333 , -2.13144 , -2.39041 , -2.63427 , -2.84702 , -3.08231 , 2.9335 , 2.66495 , 2.39774 , 2.12036 , 1.8275 , 1.50639 , 1.17722 , 0.886275 , 0.605743 , 0.28427 , -0.0396193 , -0.354007 , -0.75077 , -1.19397 , -1.57341 , -1.92157 , -2.22021 , -2.46346 , -2.75627 , 3.06492 , 2.39663 , 1.74823 , 1.37902 , 1.19887 , 1.0991 , 1.03893 , 1.00891 , 1.00518 , 1.02374 , 1.06445 , 1.13432 , 1.24916 , 1.44302 , 1.80192 , 2.45429 , -3.12323 , -2.67094 , -2.3635 , -2.09522 , -1.80127 , -1.44519 , -1.02813 , -0.58736 , -0.18956 , 0.144643 , 0.453157 , 0.757883 , 1.05904 , 1.37 , 1.69599 , 2.01414 , 2.30985 , 1.91038 , 2.21263 , 2.52486 , 2.81132 , 3.07189 , -2.94303 , -2.68517 , -2.46145 , -2.23412 , -2.01054 , -1.8258 , -1.63795 , -1.44731 , -1.29936 , -1.15411 , -0.999663 , -0.884439 , -0.782453 , -0.670718 , -0.594779 , -0.555431 , -0.50966 , -0.450443 , -0.401606 , -0.378331 , -0.386234 , -0.414164 , -0.439595 , -0.470737 , -0.53859 , -0.624373 , -0.689331 , -0.763878 , -0.887312 , -1.01416 , -1.1298 , -1.27596 , -1.43815 , -1.60414 , -1.79321 , -1.97989 , -2.17909 , -2.42041 , -2.64796 , -2.87049 , 3.13052 , 2.84533 , 2.60455 , 2.33503 , 2.00698 , 1.6927 , 1.41338 , 1.10867 , 0.750076 , 0.405336 , 0.0753555 , -0.307159 , -0.675607 , -0.993754 , -1.34751 , -1.71509 , -2.11092 , -2.62302 , -3.09158 , 2.8065 , 2.47371 , 2.20568 , 1.87806 , 1.30132 , 0.490873 , -0.0905068 , -0.360873 , -0.490045 , -0.562563 , -0.600431 , -0.606871 , -0.585013 , -0.535755 , -0.450594 , -0.304294 , -0.0296239 , 0.549024 , 1.40027 , 1.96137 , 2.30641 , 2.59962 , 2.92314 , -2.95428 , -2.46132 , -1.95605 , -1.52695 , -1.16093 , -0.819662 , -0.483985 , -0.124178 , 0.256058 , 0.620753 , 0.970681 , 1.30857 , 1.61869 }; 

/*
    for (int i = 0; i < numberofexcitationangles*numberofobservationangles; i++) {
        cout << "D[" << i << " ]: " << D[i] << endl;
    }
    */

    float del_Phi = 0;
    float fit = 0;
	for(int i = 0; i < numberofobservationangles*numberofexcitationangles*numberoffrequencies; i++)
	{
		fit -= pow(D[i]-measurement[i],2)/(pow(measurement[i],2)*numberofexcitationangles*numberoffrequencies);
        del_Phi = P[i]-measurement[i+numberofobservationangles*numberofexcitationangles*numberoffrequencies];
        if(del_Phi>PI)
        {
            del_Phi -= 2*PI;
        }
        else if(del_Phi<-1*PI)
        {
            del_Phi += 2*PI;
        }
        fit -= del_Phi*del_Phi/(PI*PI*numberofexcitationangles*numberoffrequencies);

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
        Cezeic[dgetCell(x,y,nx+1)] = (-2*eps0*eps_infinity[dgetCell(x,y,nx+1)]+2*eps0 - sigma_e_z[dgetCell(x,y,nx+1)]*dx - dev_Beta_p[dgetCell(x,y,nx+1)])/(2*eps0*eps_infinity[dgetCell(x,y,nx+1)] + sigma_e_z[dgetCell(x,y,nx+1)]*dt + dev_Beta_p[dgetCell(x,y,nx+1)]);
        Cezeip[dgetCell(x,y,nx+1)] = (2*eps0*eps_infinity[dgetCell(x,y,nx+1)] - 2*eps0 - sigma_e_z[dgetCell(x,y,nx+1)]*dt + dev_Beta_p[dgetCell(x,y,nx+1)])/(2*eps0*eps_infinity[dgetCell(x,y,nx+1)] + sigma_e_z[dgetCell(x,y,nx+1)]*dt + dev_Beta_p[dgetCell(x,y,nx+1)]); 
        Cezjp[dgetCell(x,y,nx+1)] = ((1+Kp)*dt)/(2*eps0*eps_infinity[dgetCell(x,y,nx+1)] + sigma_e_z[dgetCell(x,y,nx+1)]*dt + dev_Beta_p[dgetCell(x,y,nx+1)]);

    }
}
