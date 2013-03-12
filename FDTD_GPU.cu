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
            Ez = (dev_Ez[getCell(x,y+1,nx+1)]+dev_Ez[getCell(x,y,nx+1)])/2;
            float Hx = dev_Hx[getCell(x,y,nx)];
            cjzyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Hx*deltatime*cuexp((float)(-1)*j*(float)2*pi*freq*(float)(*timestep)*deltatime);//cjzyp and cmxyp have nx - 2*NF2FFBoundary -2 elements

            cmxyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Ez*deltatime*cuexp((float)-1.0*j*(float)2.0*(float)PI*freq*((float)(*timestep)+0.5)*(float)dt);
        }
        else if(isOnxp(x))//X faces override y faces at their intersections
        {
            Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x+1,y,nx+1)])/2;
            float Hy = dev_Hy[getCell(x,y,nx)];

            cjzxp[index-(nx-2*NF2FFdistfromboundary-2)] += Hy*deltatime*cuexp(-1*j*2*pi*freq*(float)(*timestep)*(float)dt);//cjzxp and cmyxp have ny-2*NF2FFBound elements

            cmyxp[index-(nx-2*NF2FFdistfromboundary-2)] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*pi*freq*((float)(*timestep)+0.5)*(float)dt);// this is the discrete fourier transform, by the way.
        }
        else if(isOnyn(x,y))
        {  
            Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x,y+1,nx+1)])/2;
            float Hx=dev_Hx[getCell(x,y,nx)];

            cjzyn[index] += Hx*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt);  //cjzyn and cmxyn need to have nx-2*NF2FFbound-2 elements
            cmxyn[index] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*((float)(*timestep)+0.5)*(float)dt);
        }
        else if(isOnxn(x))
        {
            Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x+1,y,nx+1)])/2;
            cjzxn[index-(2*nx+ny-6*NF2FFdistfromboundary-4)] += -1*dev_Hy[getCell(x,y,nx)]*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt); // cjzxn and cmyxn must have ny-2*NFdistfromboundary elements
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
        buffer_Hy = dev_Hy[getCell(x,y,nx)]+Chez*(dev_Ez[getCell(x+1,y,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
        buffer_Hx = dev_Hx[getCell(x,y,nx)]-Chez*(dev_Ez[getCell(x,y+1,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
        if(x<ncells)
        {
            buffer_Hy= dev_Hy[getCell(x,y,nx)]+Chez*(dev_Ez[getCell(x+1,y,nx+1)]-dev_Ez[getCell(x,y,nx+1)])/kex[ncells-1-x];
            dev_Psi_hyx[getCell(x,y,20)]=dev_bmx[ncells-1-x]*dev_Psi_hyx[getCell(x,y,20)]+dev_amx[ncells-1-x]*(dev_Ez[getCell(x+1,y,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
            buffer_Hy+=Chez*dx*dev_Psi_hyx[getCell(x,y,20)] ;
        } 
        if(x>=(nx-ncells))
        {
            buffer_Hy=dev_Hy[getCell(x,y,nx)]+Chez*(dev_Ez[getCell(x+1,y,nx+1)]-dev_Ez[getCell(x,y,nx+1)])/kex[x-nx+ncells];
            dev_Psi_hyx[getCell(x-nx+20,y,2*ncells)]=dev_bmx[x-nx+ncells]*dev_Psi_hyx[getCell(x-nx+20,y,20)]+dev_amx[x-nx+ncells]*(dev_Ez[getCell(x+1,y,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
            buffer_Hy+=Chez*dx*dev_Psi_hyx[getCell(x-nx+20,y,20)];
        }
        if(y<ncells)
        {
            buffer_Hx=dev_Hx[getCell(x,y,nx)]-Chez*(dev_Ez[getCell(x,y+1,nx+1)]-dev_Ez[getCell(x,y,nx+1)])/kex[ncells-1-y];
            dev_Psi_hxy[getCell(x,y,nx)]=dev_bmy[ncells-1-y]*dev_Psi_hxy[getCell(x,y,nx)]+dev_amy[ncells-1-y]*(dev_Ez[getCell(x,y+1,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
            buffer_Hx-=Chez*dy*dev_Psi_hxy[getCell(x,y,nx)];  
        }
        if(y>=(ny-ncells))
        {
            buffer_Hx=dev_Hx[getCell(x,y,nx)]-Chez*(dev_Ez[getCell(x,y+1,nx+1)]-dev_Ez[getCell(x,y,nx+1)])/kex[y-ny+ncells];
            dev_Psi_hxy[getCell(x,y-ny+20,nx)]=dev_bmy[y-ny+ncells]*dev_Psi_hxy[getCell(x,y-ny+20,nx)]+dev_amy[y-ny+ncells]*(dev_Ez[getCell(x,y+1,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
            buffer_Hx-=Chez*dy*dev_Psi_hxy[getCell(x,y-nx+20,nx)];
        }
        //__syncthreads();
        if(isnan(buffer_Hx)) 
        {
            dev_Hx[getCell(x,y,nx)] = 0.0;
        }
        else 
        {
            dev_Hx[getCell(x,y,nx)] = buffer_Hx;
        }

        if(isnan(buffer_Hy)) {
            dev_Hy[getCell(x,y,nx)] = 0.0;
        }
        else
        {
            dev_Hy[getCell(x,y,nx)] = buffer_Hy;
        }

        //dev_Hx[getCell(x,y,nx)] = buffer_Hx;
        //dev_Hy[getCell(x,y,nx)] = buffer_Hy;
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
        buffer_Hy = dev_Hy[getCell(x,y,nx)]+Chez*(dev_Ez[getCell(x+1,y,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
        buffer_Hx = dev_Hx[getCell(x,y,nx)]-Chez*(dev_Ez[getCell(x,y+1,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
        if(x<ncells)
        {
            buffer_Hy= dev_Hy[getCell(x,y,nx)]+Chez*(dev_Ez[getCell(x+1,y,nx+1)]-dev_Ez[getCell(x,y,nx+1)])/kex[ncells-1-x];
            dev_Psi_hyx[getCell(x,y,20)]=dev_bmx[ncells-1-x]*dev_Psi_hyx[getCell(x,y,20)]+dev_amx[ncells-1-x]*(dev_Ez[getCell(x+1,y,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
            buffer_Hy+=Chez*dx*dev_Psi_hyx[getCell(x,y,20)] ;
        } 
        if(x>=(nx-ncells))
        {
            buffer_Hy=dev_Hy[getCell(x,y,nx)]+Chez*(dev_Ez[getCell(x+1,y,nx+1)]-dev_Ez[getCell(x,y,nx+1)])/kex[x-nx+ncells];
            dev_Psi_hyx[getCell(x-nx+20,y,2*ncells)]=dev_bmx[x-nx+ncells]*dev_Psi_hyx[getCell(x-nx+20,y,20)]+dev_amx[x-nx+ncells]*(dev_Ez[getCell(x+1,y,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
            buffer_Hy+=Chez*dx*dev_Psi_hyx[getCell(x-nx+20,y,20)];
        }
        if(y<ncells)
        {
            buffer_Hx=dev_Hx[getCell(x,y,nx)]-Chez*(dev_Ez[getCell(x,y+1,nx+1)]-dev_Ez[getCell(x,y,nx+1)])/kex[ncells-1-y];
            dev_Psi_hxy[getCell(x,y,nx)]=dev_bmy[ncells-1-y]*dev_Psi_hxy[getCell(x,y,nx)]+dev_amy[ncells-1-y]*(dev_Ez[getCell(x,y+1,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
            buffer_Hx-=Chez*dy*dev_Psi_hxy[getCell(x,y,nx)];  
        }
        if(y>=(ny-ncells))
        {
            buffer_Hx=dev_Hx[getCell(x,y,nx)]-Chez*(dev_Ez[getCell(x,y+1,nx+1)]-dev_Ez[getCell(x,y,nx+1)])/kex[y-ny+ncells];
            dev_Psi_hxy[getCell(x,y-ny+20,nx)]=dev_bmy[y-ny+ncells]*dev_Psi_hxy[getCell(x,y-ny+20,nx)]+dev_amy[y-ny+ncells]*(dev_Ez[getCell(x,y+1,nx+1)]-dev_Ez[getCell(x,y,nx+1)]);
            buffer_Hx-=Chez*dy*dev_Psi_hxy[getCell(x,y-nx+20,nx)];
        }
        //__syncthreads();
        if(isnan(buffer_Hx)) 
        {
            dev_Hx[getCell(x,y,nx)] = 0.0;
        }
        else 
        {
            dev_Hx[getCell(x,y,nx)] = buffer_Hx;
        }

        if(isnan(buffer_Hy)) {
            dev_Hy[getCell(x,y,nx)] = 0.0;
        }
        else
        {
            dev_Hy[getCell(x,y,nx)] = buffer_Hy;
        }

        //dev_Hx[getCell(x,y,nx)] = buffer_Hx;
        //dev_Hy[getCell(x,y,nx)] = buffer_Hy;
    }
}

__global__ void E_field_update(int *i,float*dev_Ez,float*dev_Hy,float*dev_Hx,float*dev_Psi_ezx,float*dev_aex,float*dev_aey,float*dev_bex,float*dev_bey,float*dev_Psi_ezy,float*kex,float*Cezhy,float*Cezhx,float*Ceze,float*Cezeip,float*Cezeic,float*Phi)
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
            if(isscattering)
            {

                buffer_Ez = Ceze[getCell(x,y,nx+1)]*dev_Ez[getCell(x,y,nx+1)]+Cezhy[getCell(x,y,nx+1)]*(dev_Hy[getCell(x,y,nx)]-dev_Hy[getCell(x-1,y,nx)])
                    -Cezhx[getCell(x,y,nx+1)]*(dev_Hx[getCell(x,y,nx)]-dev_Hx[getCell(x,y-1,nx)])
                    +Cezeic[getCell(x,y,nx+1)]*fwf((float)(*i)+0.5,x-nx/2,y-ny/2,*Phi,-breast_radius)
                    +Cezeip[getCell(x,y,nx+1)]*fwf((float)(*i)-0.5,x-nx/2,y-ny/2,*Phi,-breast_radius);

            }
            else
            {
                buffer_Ez = Ceze[getCell(x,y,nx+1)]*dev_Ez[getCell(x,y,nx+1)]+Cezhy[getCell(x,y,nx+1)]*(dev_Hy[getCell(x,y,nx)]-dev_Hy[getCell(x-1,y,nx)])
                    -Cezhx[getCell(x,y,nx+1)]*(dev_Hx[getCell(x,y,nx)]-dev_Hx[getCell(x,y-1,nx)]);
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
                buffer_Ez = Ceze[getCell(x,y,nx+1)]*dev_Ez[getCell(x,y,nx+1)]+Cezhy[getCell(x,y,nx+1)]*(dev_Hy[getCell(x,y,nx)]-dev_Hy[getCell(x-1,y,nx)])/kex[ncells-x]
                    -Cezhx[getCell(x,y,nx+1)]*(dev_Hx[getCell(x,y,nx)]-dev_Hx[getCell(x,y-1,nx)])/kex[ncells-x];
                dev_Psi_ezx[getCell(x-1,y-1,20)] = dev_bex[ncells-x]*dev_Psi_ezx[getCell(x-1,y-1,20)]+dev_aex[ncells-x]*(dev_Hy[getCell(x,y,nx)]-dev_Hy[getCell(x-1,y,nx)]);
                buffer_Ez += Cezhy[getCell(x,y,nx+1)]*dx*dev_Psi_ezx[getCell(x-1,y-1,2*ncells)];
            }
            if(x>=(nx-ncells)&&x!=nx)
            {
                buffer_Ez = Ceze[getCell(x,y,nx+1)]*dev_Ez[getCell(x,y,nx+1)]+Cezhy[getCell(x,y,nx+1)]*(dev_Hy[getCell(x,y,nx)]-dev_Hy[getCell(x-1,y,nx)])/kex[x-nx+ncells]
                    -Cezhx[getCell(x,y,nx+1)]*(dev_Hx[getCell(x,y,nx)]-dev_Hx[getCell(x,y-1,nx)])/kex[x-nx+ncells];
                dev_Psi_ezx[getCell(x-nx+20,y-1,20)]=dev_bex[x-nx+ncells]*dev_Psi_ezx[getCell(x-nx+20,y-1,20)]+dev_aex[x-nx+ncells]*(dev_Hy[getCell(x,y,nx)]-dev_Hy[getCell(x-1,y,nx)]);
                buffer_Ez+=Cezhy[getCell(x,y,nx+1)]*dx*dev_Psi_ezx[getCell(x-nx+20,y-1,2*ncells)];
            }
            if(y<=ncells&&y!=0)
            {
                buffer_Ez = Ceze[getCell(x,y,nx+1)]*dev_Ez[getCell(x,y,nx+1)]+Cezhy[getCell(x,y,nx+1)]*(dev_Hy[getCell(x,y,nx)]-dev_Hy[getCell(x-1,y,nx)])/kex[ncells-y]
                    -Cezhx[getCell(x,y,nx+1)]*(dev_Hx[getCell(x,y,nx)]-dev_Hx[getCell(x,y-1,nx)])/kex[ncells-y];
                dev_Psi_ezy[getCell(x-1,y-1,nx)]=dev_bey[(ncells-y)]*dev_Psi_ezy[getCell(x-1,y-1,nx)]+dev_aey[(ncells-y)]*(dev_Hx[getCell(x,y,nx)]-dev_Hx[getCell(x,y-1,nx)]);
                buffer_Ez-=Cezhx[getCell(x,y,nx+1)]*dy*dev_Psi_ezy[getCell(x-1,y-1,nx)];
            }
            if(y>=(ny-ncells)&&y!=ny)
            {
                buffer_Ez = Ceze[getCell(x,y,nx+1)]*dev_Ez[getCell(x,y,nx+1)]+Cezhy[getCell(x,y,nx+1)]*(dev_Hy[getCell(x,y,nx)]-dev_Hy[getCell(x-1,y,nx)])/kex[y-ny+ncells]
                    -Cezhx[getCell(x,y,nx+1)]*(dev_Hx[getCell(x,y,nx)]-dev_Hx[getCell(x,y-1,nx)])/kex[y-ny+ncells];
                dev_Psi_ezy[getCell(x-1,y-ny+20,nx)]=dev_bey[y-ny+ncells]*dev_Psi_ezy[getCell(x-1,y-ny+20,nx)]+dev_aey[y-ny+ncells]*(dev_Hx[getCell(x,y,nx)]-dev_Hx[getCell(x,y-1,nx)]);
                buffer_Ez-=Cezhx[getCell(x,y,nx+1)]*dy*dev_Psi_ezy[getCell(x-1,y-ny+20,nx)];
            }
        }
        //		unsigned char green = 128+127*buffer_Ez/0.4;
        /*ptr[offset].x = 0;
          ptr[offset].y = green;
          ptr[offset].z = 0;
          ptr[offset].w = 255;*///OpenGL stuff

        //__syncthreads();
        if(isnan(buffer_Ez)) {
            dev_Ez[getCell(x,y,nx+1)] = 0.0;
        }
        else {
            dev_Ez[getCell(x,y,nx+1)] = buffer_Ez;
        }
        //dev_Ez[getCell(x,y,nx+1)] = buffer_Ez;
    }

}

__global__ void Field_reset(float* Ez, float* Hy, float* Hx, float* Psi_ezy,float* Psi_ezx,float* Psi_hyx,float* Psi_hxy,cuComplex*cjzyn,cuComplex*cjzxp,cuComplex*cjzyp,cuComplex*cjzxn,cuComplex*cmxyn,cuComplex*cmyxp,cuComplex*cmxyp,cuComplex*cmyxn)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    int index = x + y*blockDim.x*gridDim.x;
    if(x<=ncells&&x!=0)
    {
        Psi_ezx[getCell(x-1,y-1,20)] =0;
    }
    if(x>=(nx-ncells)&&x!=nx)
    {
        Psi_ezx[getCell(x-nx+20,y-1,20)]=0;
    }
    if(y<=ncells&&y!=0)
    {
        Psi_ezy[getCell(x-1,y-1,nx)]=0;
    }
    if(y>=(ny-ncells)&&y!=ny)
    {
        Psi_ezy[getCell(x-1,y-ny+20,nx)]=0;
    }
    if(x<ncells)
    {

        Psi_hyx[getCell(x,y,20)]=0;
    } 
    if(x>=(nx-ncells))
    {
        Psi_hyx[getCell(x-nx+20,y,2*ncells)]=0.0;
    }
    if(y<ncells)
    {
        Psi_hxy[getCell(x,y,nx)]=0.0;
    }
    if(y>=(ny-ncells))
    {
        Psi_hxy[getCell(x,y-ny+20,nx)]=0.0;
    }
    if(x<=nx&&y<=ny)
    {
        Ez[getCell(x,y,nx+1)] = 0.0;
    }
    if(x<nx&&y<ny)
    {
        Hy[getCell(x,y,nx)] = 0.0;
        Hx[getCell(x,y,nx)] = 0.0;
    }

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


}

__global__ void E_inc_update(int *i,float*dev_Hy_inc,float*dev_Hx_inc,float*dev_Psi_ezx_inc,float*dev_aex,float*dev_aey,float*dev_bex,float*dev_bey,float*dev_Psi_ezy_inc,float*kex,float*dev_Ezip,float*dev_Ezic,float*Phi)
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
            buffer_Ez = Ceze*dev_Ezic[getCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)])
                -Cezhy*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)]);

            if(x==((int)source_x)&&y==(int)(source_y))
            {
                //buffer_Ez=buffer_Ez + Cezj*dev_Jz[*i];
                buffer_Ez=buffer_Ez + 100*Cezj*fwf((float)(*i),0,0,0,0);
            }
            if(x<=ncells&&x!=0)
            {
                buffer_Ez = Ceze*dev_Ezic[getCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)])/kex[ncells-x]
                    -Cezhy*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)])/kex[ncells-x];
                dev_Psi_ezx_inc[getCell(x-1,y-1,20)] = dev_bex[ncells-x]*dev_Psi_ezx_inc[getCell(x-1,y-1,20)]+dev_aex[ncells-x]*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)]);
                buffer_Ez += Cezhy*dx*dev_Psi_ezx_inc[getCell(x-1,y-1,2*ncells)];
            }
            if(x>=(nx-ncells)&&x!=nx)
            {
                buffer_Ez = Ceze*dev_Ezic[getCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)])/kex[x-nx+ncells]
                    -Cezhy*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)])/kex[x-nx+ncells];
                dev_Psi_ezx_inc[getCell(x-nx+20,y-1,20)]=dev_bex[x-nx+ncells]*dev_Psi_ezx_inc[getCell(x-nx+20,y-1,20)]+dev_aex[x-nx+ncells]*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)]);
                buffer_Ez+=Cezhy*dx*dev_Psi_ezx_inc[getCell(x-nx+20,y-1,2*ncells)];
            }
            if(y<=ncells&&y!=0)
            {
                buffer_Ez = Ceze*dev_Ezic[getCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)])/kex[ncells-y]
                    -Cezhy*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)])/kex[ncells-y];
                dev_Psi_ezy_inc[getCell(x-1,y-1,nx)]=dev_bey[(ncells-y)]*dev_Psi_ezy_inc[getCell(x-1,y-1,nx)]+dev_aey[(ncells-y)]*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)]);
                buffer_Ez-=Cezhy*dy*dev_Psi_ezy_inc[getCell(x-1,y-1,nx)];
            }
            if(y>=(ny-ncells)&&y!=ny)
            {
                buffer_Ez = Ceze*dev_Ezic[getCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)])/kex[y-ny+ncells]
                    -Cezhy*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)])/kex[y-ny+ncells];
                dev_Psi_ezy_inc[getCell(x-1,y-ny+20,nx)]=dev_bey[y-ny+ncells]*dev_Psi_ezy_inc[getCell(x-1,y-ny+20,nx)]+dev_aey[y-ny+ncells]*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)]);
                buffer_Ez-=Cezhy*dy*dev_Psi_ezy_inc[getCell(x-1,y-ny+20,nx)];
            }
        }
        dev_Ezip[getCell(x,y,nx+1)] = dev_Ezic[getCell(x,y,nx+1)];
        dev_Ezic[getCell(x,y,nx+1)] = buffer_Ez;
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
            Ez = (dev_Ez[getCell(x,y+1,nx+1)]+dev_Ez[getCell(x,y,nx+1)])/2;
            Ez += (dev_Ezic[getCell(x,y+1,nx+1)] + dev_Ezic[getCell(x,y,nx+1)] + dev_Ezip[getCell(x,y+1,nx+1)] + dev_Ezip[getCell(x,y,nx+1)])/4;
            float Hx = dev_Hx[getCell(x,y,nx)] + dev_Hx_inc[getCell(x,y,nx)];
            cjzyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Hx*deltatime*cuexp((float)(-1)*j*(float)2*pi*freq*(float)(*timestep)*deltatime);//cjzyp and cmxyp have nx - 2*NF2FFBoundary -2 elements
            cmxyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Ez*deltatime*cuexp((float)-1.0*j*(float)2.0*(float)PI*freq*((float)(*timestep)-0.5)*(float)dt);
        }
        else if(isOnxp(x))//X faces override y faces at their intersections
        {
            Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x+1,y,nx+1)])/2;
            Ez += (dev_Ezic[getCell(x+1,y,nx+1)] + dev_Ezic[getCell(x,y,nx+1)] + dev_Ezip[getCell(x+1,y,nx+1)] + dev_Ezip[getCell(x,y,nx+1)])/4;
            float Hy = dev_Hy[getCell(x,y,nx)] + dev_Hy_inc[getCell(x,y,nx)];

            cjzxp[index-(nx-2*NF2FFdistfromboundary-2)] += Hy*deltatime*cuexp(-1*j*2*pi*freq*(float)(*timestep)*(float)dt);//cjzxp and cmyxp have ny-2*NF2FFBound elements

            cmyxp[index-(nx-2*NF2FFdistfromboundary-2)] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*pi*freq*((float)(*timestep)-0.5)*(float)dt);// this is the discrete fourier transform, by the way.
        }
        else if(isOnyn(x,y))
        {  
            Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x,y+1,nx+1)])/2;
            Ez += (dev_Ezic[getCell(x,y+1,nx+1)] + dev_Ezic[getCell(x,y,nx+1)] + dev_Ezip[getCell(x,y+1,nx+1)] + dev_Ezip[getCell(x,y,nx+1)])/4;
            float Hx=dev_Hx[getCell(x,y,nx)]+dev_Hx_inc[getCell(x,y,nx)];

            cjzyn[index] += Hx*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt);	//cjzyn and cmxyn need to have nx-2*NF2FFbound-2 elements
            cmxyn[index] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*((float)(*timestep)-0.5)*(float)dt);
        }
        else if(isOnxn(x))
        {
            Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x+1,y,nx+1)])/2;
            Ez += (dev_Ezic[getCell(x+1,y,nx+1)] + dev_Ezic[getCell(x,y,nx+1)] + dev_Ezip[getCell(x+1,y,nx+1)] + dev_Ezip[getCell(x,y,nx+1)])/4;
            float Hy = dev_Hy[getCell(x,y,nx)] + dev_Hy_inc[getCell(x,y,nx)];
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

//static void draw_func(void){
//	glDrawPixels(nx,ny,GL_RGBA,GL_UNSIGNED_BYTE,0);
//	glutSwapBuffers;
//}

using namespace std;

__global__ void scattered_parameter_init(float*eps_r_z,float*sigma_e_z,float*Cezeic,float*Cezeip);

double FDTD_GPU(const vector<double> &arguments) {
    cout << "calculating FDTD GPU" << endl;

    cudaSetDevice(0);

    vector<float> image;
    //This is setting the material parameters of the optimization cells.
    for (int lerp = 0; lerp < 81; lerp++) {
        image.push_back((float)arguments.at(lerp));
        //image.push_back(10);
    }

    for (int lerp = 81; lerp < 81 * 2; lerp++) {
        image.push_back((float)arguments.at(lerp));
        // image.push_back(0);
    }
    cudaError_t error;

    float freq = center_freq;

    int grid_x = int(ceil((float)nx / 22));
    int grid_y = int(ceil((float)ny / 22));

    dim3 grid(grid_x, grid_y);
    dim3 block(22, 22);

    float *Ez = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    float *eps_r_z = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    float *sigma_e_z = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    float *Ceze = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    float *Cezhy = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    float *Cezhx = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    //Cezj later if using loop current source
    //float *Cezj = (float*)malloc(sizeof(float)*(1+nx)*(1+ny)); // if using loop current source

    int size = nx + 1;
    float radius;//tumor_radius,tumor_radius_2,tumor_radius_3;

    for (int j = 0; j < ny + 1; j++) {
        for (int i = 0; i < nx + 1; i++) {
            Ez[getCell(i,j,size)] = (float)0;
            sigma_e_z[getCell(i,j,size)] = 0;
            eps_r_z[getCell(i,j,size)] = 1;

            radius = sqrt(pow( ((float)i-nx/2)*dx,2) + pow( ((float)j-ny/2)*dy,2));

            //tumor_radius = sqrt(pow( ((float)i - target_x)*dx,2) + pow( ((float)j-target_y)*dy,2));
            if (radius <= breast_radius) {
                eps_r_z[getCell(i,j,size)] = (float)image.at(getOptimizationCell(i,j)); //This is the line that should be uncommented if using as forward solver
                sigma_e_z[getCell(i,j,size)] = (float)image.at(getOptimizationCell(i,j)+9*9);

                //eps_r_z[getCell(i,j,size)] = 10;
                //sigma_e_z[getCell(i,j,size)] = 0.15;
                //if(tumor_radius <= tumor_size)//delete this if using as forward solver
                //{
                //	eps_r_z[getCell(i,j,size)] = 60;
                //	sigma_e_z[getCell(i,j,size)] = 0.7;
                //}
            }
            Ceze[getCell(i,j,nx+1)] = (2*eps_r_z[getCell(i,j,nx+1)]*eps0-dt*sigma_e_z[getCell(i,j,nx+1)])/(2*eps_r_z[getCell(i,j,nx+1)]*eps0+dt*sigma_e_z[getCell(i,j,nx+1)]);
            Cezhy[getCell(i,j,size)] = (2*dt/dx)/(2*eps_r_z[getCell(i,j,size)]*eps0+dt*sigma_e_z[getCell(i,j,size)]);
            Cezhx[getCell(i,j,size)] = (2*dt/dy)/(2*eps_r_z[getCell(i,j,size)]*eps0+dt*sigma_e_z[getCell(i,j,size)]);
        }
    }

    float *sigma_e_pml = (float*)malloc(sizeof(float)*ncells);
    float *sigma_m_pml = (float*)malloc(sizeof(float)*ncells);

    //initialize
    float sigma_max = (npml+1)/(150*PI*dx);
    float rho;
    for (int i = 0; i < ncells; i++) {
        rho = ((float)i+0.25)/ncells;
        sigma_e_pml[i] = sigma_max*sigma_factor*pow(rho,npml);
        sigma_m_pml[i] = (mu0/eps0)*sigma_max*sigma_factor*pow(rho,npml);
        //cout<<"sigma_e_pml = "<<sigma_e_pml[i]<<endl;
        //cout<<"sigma_m_pml "<<sigma_m_pml[i]<<endl;
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
        kmx[i]=pow(rho,npml)*(kmax-1)+1;
        alpha_e[i]=alpha_min+(alpha_max-alpha_min)*rho;

        rho = ((float)i+0.75)/ncells;
        alpha_m[i]=(mu0/eps0)*(alpha_min+(alpha_max-alpha_min)*rho);

        aex[i]=((bex[i]-1)*sigma_e_pml[i])/(dx*(sigma_e_pml[i]*kex[i]+alpha_e[i]*kex[i]*kex[i]));
        bex[i]=exp(-1*(dt/eps0)*(sigma_e_pml[i]/kex[i]+alpha_e[i]));
        amx[i]=(bmx[i]-1)*sigma_m_pml[i]/(dx*(sigma_m_pml[i]*kmx[i]+alpha_m[i]*kmx[i]*kmx[i]));

        float argument = -1*(dt/mu0)*((sigma_m_pml[i]/kmx[i])+alpha_m[i]);
        bmx[i]=exp(argument);
        //cout<<"kex["<<i<<"]= "<<kex[i]<<endl;
        //cout<<"kmx["<<i<<"]= "<<kmx[i]<<endl;
        //cout<<"aex["<<i<<"]= "<<aex[i]<<endl;
        //cout<<"amx["<<i<<"]= "<<amx[i]<<endl;
        //cout<<"bex["<<i<<"]= "<<bex[i]<<endl;
        //cout<<"bmx["<<i<<"]= "<<bmx[i]<<endl;
        //cout<<"alpha_e = "<<alpha_e[i]<<endl;
        //cout<<"alpha_m = "<<alpha_m[i]<<endl;
    }

    float *Psi_ezy = (float*)malloc(sizeof(float)*ny*20);
    float *Psi_ezx = (float*)malloc(sizeof(float)*nx*20);
    float *Psi_hyx = (float*)malloc(sizeof(float)*ny*20);
    float *Psi_hxy = (float*)malloc(sizeof(float)*nx*20);

    for (int i = 0; i < nx * 20; i++) {
        Psi_ezy[i] = 0.0;
        Psi_hxy[i] = 0.0;
    }

    for (int i = 0; i< ny * 20; i++) {
        Psi_ezx[i] = 0.0;
        Psi_hyx[i] = 0.0;
    }

    float *D = (float*)malloc(sizeof(float)*numberofexcitationangles*numberofobservationangles);//D = (float*)malloc(numberofobservationangles*sizeof(float));

    float *Hy = (float*)malloc(sizeof(float)*nx*ny);
    float *Hx = (float*)malloc(sizeof(float)*nx*ny);

    //This are output values from the device
    cuComplex *hcjzyp = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzy);
    cuComplex *hcjzyn = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzy);
    cuComplex *hcjzxp = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzx);
    cuComplex *hcjzxn = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzx);
    cuComplex *hcmxyn = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzy);
    cuComplex *hcmxyp = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzy);
    cuComplex *hcmyxp = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzx);
    cuComplex *hcmyxn = (cuComplex*)malloc(sizeof(cuComplex)*size_cjzx);

    cuComplex *L  = (cuComplex*)malloc(sizeof(cuComplex)*size_NF2FF_total);
    cuComplex *N  = (cuComplex*)malloc(sizeof(cuComplex)*size_NF2FF_total);

    cuComplex *cjzxp, *cjzyp, *cjzxn, *cjzyn, *cmxyp, *cmyxp, *cmxyn, *cmyxn;

    float *dev_Cezeic, *dev_Cezeip;
    float *dev_sigma_e_z, *dev_eps_r_z;
    float *dev_freq, *dev_Phi;
    float *dev_Ceze, *dev_Cezhy, *dev_Cezhx, *dev_bex, *dev_aex, *dev_bmx, *dev_amx, *dev_kex, *dev_kmx;//dev_Cezj if using loop current source
    float *dev_Ez, *dev_Hy, *dev_Hx;

    float *dev_Psi_ezy, *dev_Psi_ezx, *dev_Psi_hyx, *dev_Psi_hxy;

    cudaMalloc(&dev_eps_r_z,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_sigma_e_z,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_Cezeic,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_Cezeip,sizeof(float)*(nx+1)*(ny+1));
    cudaMemcpy(dev_eps_r_z,eps_r_z,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sigma_e_z,sigma_e_z,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyHostToDevice);
    scattered_parameter_init<<<grid,block>>>(dev_eps_r_z,dev_sigma_e_z,dev_Cezeic,dev_Cezeip);
    //float *Cezeic = (float*)malloc((sizeof(float))*(nx+1)*(ny+1));
    // float *Cezeip = (float*)malloc((sizeof(float))*(nx+1)*(ny+1));
    //cudaMemcpy(Cezeic,dev_Cezeic,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyDeviceToHost);
    //cudaMemcpy(Cezeip,dev_Cezeip,sizeof(float)*(nx+1)*(ny+1),cudaMemcpyDeviceToHost);


    cudaMalloc(&dev_Phi,sizeof(float));
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

    cudaMalloc(&cjzxp,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cjzyp,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cjzxn,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cjzyn,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmxyp,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmxyn,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmyxp,sizeof(cuComplex)*size_NF2FF_total);
    cudaMalloc(&cmyxn,sizeof(cuComplex)*size_NF2FF_total);

    cudaMemcpy(dev_freq,&freq,sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc(&dev_bex,sizeof(float)*10);
    cudaMalloc(&dev_bmx,sizeof(float)*10);
    cudaMalloc(&dev_amx,sizeof(float)*10);
    cudaMalloc(&dev_aex,sizeof(float)*10);
    cudaMalloc(&dev_Ceze,sizeof(float)*(nx+1)*(ny+1));
    cudaMalloc(&dev_Cezhy,sizeof(float)*(nx+1)*(ny+1));


    //cudaMalloc(&dev_Cezj,sizeof(float)*(nx+1)*(ny+1)); if using current source

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(error));
    }
    Field_reset<<<grid,block>>>(dev_Ez, dev_Hy, dev_Hx, dev_Psi_ezy, dev_Psi_ezx, dev_Psi_hyx, dev_Psi_hxy,cjzyn,cjzxp,cjzyp,cjzxn,cmxyn,cmyxp,cmxyp,cmyxn);
    //Field_reset is also good for making all these values zero.


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

    int *dev_i;
    cudaMalloc(&dev_i,sizeof(int));
    float test_Ez;

    dim3 gridNF2FF((int)ceil(size_NF2FF_total/512.0));
    dim3 blockNF2FF(512);

    float test_Ez_2;
    float Phi;

    for(int Phi_index = 0; Phi_index < numberofexcitationangles; Phi_index++) {

        Phi = Phi_index*2*PI/numberofexcitationangles;
        cudaMemcpy(dev_Phi,&Phi,sizeof(float),cudaMemcpyHostToDevice);

        for (int i = 0; i < number_of_time_steps; i++) {
            cudaMemcpy(dev_i,&i,sizeof(int),cudaMemcpyHostToDevice);
            H_field_update<<<grid,block>>>(dev_Hy,dev_Hx,dev_Ez,dev_bmx,dev_Psi_hyx,dev_amx,dev_bmx,dev_amx,dev_Psi_hxy,dev_kmx);
            E_field_update<<<grid,block>>>(dev_i,dev_Ez,dev_Hy,dev_Hx,dev_Psi_ezx,dev_aex,dev_aex,dev_bex,dev_bex,dev_Psi_ezy,dev_kex,dev_Cezhy,dev_Cezhy,dev_Ceze,dev_Cezeip,dev_Cezeic,dev_Phi);
            calculate_JandM<<<gridNF2FF,blockNF2FF>>>(dev_freq, dev_i,dev_Ez,dev_Hy,dev_Hx,cjzxp,cjzyp,cjzxn,cjzyn,cmxyp,cmyxp,cmxyn,cmyxn);

        }

        cudaMemcpy(hcjzyn,cjzyn,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
        cudaMemcpy(hcjzxp,cjzxp,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost);
        cudaMemcpy(hcjzyp,cjzyp,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
        cudaMemcpy(hcjzxn,cjzxn,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost);
        cudaMemcpy(hcmxyn,cmxyn,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
        cudaMemcpy(hcmyxp,cmyxp,sizeof(cuComplex)*size_cjzx,cudaMemcpyDeviceToHost);
        cudaMemcpy(hcmxyp,cmxyp,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);
        cudaMemcpy(hcmyxn,cmyxn,sizeof(cuComplex)*size_cjzy,cudaMemcpyDeviceToHost);

        cuComplex nullComplex(0,0);
        for (int i = 0; i < size_NF2FF_total; i++) {
            L[i] = nullComplex;
            N[i] = nullComplex;
        }

        N2FPostProcess(D + Phi_index*numberofobservationangles, freq,N,L,hcjzxp,hcjzyp,hcjzxn,hcjzyn,hcmxyp,hcmyxp,hcmxyn,hcmyxn);
        //notice the D + Phi_index*numberofobservationangles. D is in total 4*numberofobservaion angles, so that's how we fill them in sequentially.

        //for(int i = 0;i<numberofobservationangles;i++)  // This is for recording simulated measured data
        //{
        //measurement_data<<*(D+Phi_index*numberofobservationangles+i)<<" , ";
        //cout<<*(D+Phi_index*numberofobservationangles+i)<<endl;
        //}

        //measurement_data<<endl;
        Field_reset<<<grid,block>>>(dev_Ez, dev_Hy, dev_Hx, dev_Psi_ezy, dev_Psi_ezx, dev_Psi_hyx, dev_Psi_hxy,cjzyn,cjzxp,cjzyp,cjzxn,cmxyn,cmyxp,cmxyp,cmyxn);

    }



    float measurement[numberofobservationangles*numberofexcitationangles] = {0.544912 , 0.518606 , 0.439233 , 0.330533 , 0.219116 , 0.135115 , 0.0923969 , 0.0774134 , 0.0740459 , 0.0739238 , 0.0660047 , 0.0465372 , 0.0248307 , 0.00913681 , 0.00186162 , 0.0038402 , 0.0130785 , 0.0238094 , 0.0312918 , 0.035705 , 0.0388307 , 0.039513 , 0.0368443 , 0.0338221 , 0.0324815 , 0.0305907 , 0.0270149 , 0.0239178 , 0.0224438 , 0.021849 , 0.0217346 , 0.0222152 , 0.023146 , 0.0245181 , 0.0267161 , 0.0286964 , 0.0276803 , 0.0235098 , 0.0197177 , 0.0183168 , 0.0196998 , 0.0261493 , 0.0375584 , 0.0479223 , 0.0511598 , 0.0461443 , 0.035713 , 0.0249863 , 0.0203708 , 0.0260456 , 0.0395441 , 0.054163 , 0.0660136 , 0.0763823 , 0.0935922 , 0.132053 , 0.201299 , 0.299247 , 0.410792 , 0.504467 , 
        0.0490085 , 0.0278468 , 0.0123693 , 0.00899709 , 0.0196632 , 0.0401112 , 0.0623734 , 0.0809561 , 0.096057 , 0.113814 , 0.145125 , 0.200388 , 0.283438 , 0.386362 , 0.486139 , 0.549594 , 0.547993 , 0.475775 , 0.358033 , 0.230962 , 0.118935 , 0.039843 , 0.00700227 , 0.0112335 , 0.0300356 , 0.0494414 , 0.0605159 , 0.0585777 , 0.0503323 , 0.045704 , 0.0474064 , 0.0523123 , 0.0558987 , 0.0545722 , 0.0475098 , 0.0366045 , 0.0248037 , 0.0155752 , 0.0115322 , 0.0127167 , 0.0176523 , 0.0243556 , 0.0310764 , 0.037444 , 0.0432292 , 0.0469609 , 0.0471761 , 0.0435653 , 0.0369347 , 0.0293987 , 0.0235478 , 0.0206039 , 0.020754 , 0.0247748 , 0.0336772 , 0.047007 , 0.0618746 , 0.0734482 , 0.0763332 , 0.0674785 , 
        0.0463129 , 0.0448933 , 0.0398454 , 0.0319834 , 0.0239428 , 0.0174267 , 0.0129155 , 0.0116624 , 0.0154122 , 0.0247183 , 0.0376821 , 0.0494142 , 0.0552493 , 0.0544909 , 0.0501016 , 0.0466044 , 0.047395 , 0.0522298 , 0.0576919 , 0.0588555 , 0.0504011 , 0.0311956 , 0.0107719 , 0.00755493 , 0.0394798 , 0.116099 , 0.232324 , 0.36478 , 0.478314 , 0.541685 , 0.541186 , 0.484009 , 0.391878 , 0.291105 , 0.204554 , 0.145352 , 0.113254 , 0.0973423 , 0.0835717 , 0.0637299 , 0.0397899 , 0.0189781 , 0.00814281 , 0.0118845 , 0.0291142 , 0.0513172 , 0.0680543 , 0.0744519 , 0.0718442 , 0.0622228 , 0.0473734 , 0.0329352 , 0.0245156 , 0.0212818 , 0.0204027 , 0.0228792 , 0.0298908 , 0.0380399 , 0.0432513 , 0.0455291 , 
        0.0469428 , 0.049667 , 0.0453111 , 0.0370016 , 0.0278006 , 0.0201062 , 0.0173687 , 0.020228 , 0.0242543 , 0.0264199 , 0.0275476 , 0.027771 , 0.0262174 , 0.0237332 , 0.0219206 , 0.0212424 , 0.0214967 , 0.0226845 , 0.0248514 , 0.0275874 , 0.0300439 , 0.0318892 , 0.0340621 , 0.0369823 , 0.0388068 , 0.0379494 , 0.0350817 , 0.030462 , 0.0230471 , 0.0133404 , 0.00457234 , 0.00152755 , 0.00874873 , 0.0260448 , 0.0463293 , 0.0633742 , 0.0751071 , 0.0775575 , 0.0756597 , 0.0916989 , 0.141021 , 0.22185 , 0.328433 , 0.44207 , 0.524772 , 0.544711 , 0.498668 , 0.407614 , 0.29953 , 0.199594 , 0.128704 , 0.0929922 , 0.0772499 , 0.0654169 , 0.0536587 , 0.0399619 , 0.0255793 , 0.0193488 , 0.0253531 , 0.0373143 , 
    };//I've just hardcoded the measurement values.  Maybe later we'll read them from a text file.


    float fit;
    fit = fitness(D, numberofobservationangles * numberofexcitationangles, measurement);

    error = cudaGetLastError();

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

    free(hcjzxp);
    free(hcjzyp);
    free(hcjzxn);
    free(hcjzyn);
    free(hcmxyp);
    free(hcmyxp);
    free(hcmxyn);
    free(hcmyxn);

    free(L);
    free(N);

    cudaFree(dev_Cezeic);
    cudaFree(dev_Cezeip);
    cudaFree(dev_sigma_e_z);
    cudaFree(dev_eps_r_z);
    cudaFree(dev_freq);

    cudaFree(cjzxp);
    cudaFree(cjzyp);
    cudaFree(cjzxn);
    cudaFree(cjzyn);
    cudaFree(cmxyp);
    cudaFree(cmyxp);
    cudaFree(cmxyn);
    cudaFree(cmyxn);

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

    cout << "fitness is: " << fit << endl;
    return (double)fit;
}

__global__ void scattered_parameter_init(float*eps_r_z,float*sigma_e_z,float*Cezeic,float*Cezeip)
{
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    int y=threadIdx.y+blockDim.y*blockIdx.y;
    if(x<(nx+1)&&y<(ny+1))
    {
        Cezeic[getCell(x,y,nx+1)] = (2*(eps0-eps0*eps_r_z[getCell(x,y,nx+1)])-sigma_e_z[getCell(x,y,nx+1)]*dt)/(2*eps0*eps_r_z[getCell(x,y,nx+1)]+sigma_e_z[getCell(x,y,nx+1)]*dt);
        Cezeip[getCell(x,y,nx+1)] = -1*(2*(eps0-eps0*eps_r_z[getCell(x,y,nx+1)])+sigma_e_z[getCell(x,y,nx+1)]*dt)/(2*eps0*eps_r_z[getCell(x,y,nx+1)]+sigma_e_z[getCell(x,y,nx+1)]*dt);

    }
}
