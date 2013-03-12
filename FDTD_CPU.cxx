
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
//#include "stdafx.h"
#include <iomanip>
#include <time.h>
//#include <cuda_gl_interop.h>

//#include <cuComplex.h>
#include <vector>
#include <cmath>
#include "FDTD_common.hxx"


//#include "EasyBMP.h"
//#include "EasyBMP_DataStructures.h"
//#include "EasyBMP_VariousBMPutilities.h"
//
//__constant__ float*dev_Ceze,*dev_Cezhy,*dev_Cezhx,*dev_Cezj,*dev_Jz,*dev_Chyh,*dev_Chxh,*dev_Chyez,*dev_Chxez,*dev_bex,*dev_bey,*dev_aex,*dev_aey,*dev_bmy,*dev_bmx,*dev_amy,*dev_amx,*dev_C_Psi_ezy,
//*dev_C_Psi_ezx,*dev_C_Psi_hxy,*dev_C_Psi_hyx;
struct cpu_cuComplex {
    float   r;
    float   i;
    cpu_cuComplex( float a, float b ) : r(a), i(b)  {}
    cpu_cuComplex(float a): r(a), i(0) {}
    float magnitude2( void ) { return r * r + i * i; }
    cpu_cuComplex operator*(const cpu_cuComplex& a) {
        return cpu_cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cpu_cuComplex operator*(const float& a){
        return cpu_cuComplex(r*a,i*a);
    }  

    cpu_cuComplex operator+(const cpu_cuComplex& a) {
        return cpu_cuComplex(r+a.r, i+a.i);
    }
    cpu_cuComplex operator+(const float& a){
        return cpu_cuComplex(r+a,i);
    }
    void operator+=(const float& f){
        r += f;
    }
    void operator+=(const cpu_cuComplex& C);
    cpu_cuComplex();
};

cpu_cuComplex operator*(const float &f, const cpu_cuComplex &C)
{

    return cpu_cuComplex(C.r*f,C.i*f);
}
void cpu_cuComplex::operator+=(const cpu_cuComplex& C)
{
    r +=C.r;
    i += C.i;
}

float cuabs(cpu_cuComplex x)
{
    return sqrt(x.i*x.i + x.r*x.r);
}
cpu_cuComplex cuexp(cpu_cuComplex arg)
{
    cpu_cuComplex res(0,0);
    float s, c;
    float e = expf(arg.r);
    c = cos(arg.i);
    s = sin(arg.i);
    res.r = c * e;
    res.i = s * e;
    return res;

}
int cpu_isOnNF2FFBound(int x, int y)
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

int cpu_isOnxn(int x)
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

int cpu_isOnxp(int x)
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
int cpu_isOnyp(int x,int y)
{
    if(y==(ny-NF2FFdistfromboundary-1)&&!cpu_isOnxn(x)&&!cpu_isOnxp(x))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}



int cpu_isOnyn(int x, int y)
{
    if((y==(NF2FFdistfromboundary))&&!cpu_isOnxn(x)&&!(cpu_isOnxp(x)))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}


void calculate_JandM(float* f,int* timestep,float*dev_Ez,float*dev_Hy,float*dev_Hx,cpu_cuComplex *cjzxp,cpu_cuComplex *cjzyp,cpu_cuComplex*cjzxn,cpu_cuComplex*cjzyn,cpu_cuComplex*cmxyp,cpu_cuComplex*cmyxp,cpu_cuComplex*cmxyn,cpu_cuComplex*cmyxn)
{
    float freq = *f;
    for(int index = 0;index<=size_NF2FF_total;index++)
    {
        if(index<=size_NF2FF_total)
        {
            const cpu_cuComplex j(0.0,1.0);
            int x = CPUgetxfromthreadIdNF2FF(index);
            int y = CPUgetyfromthreadIdNF2FF(index);

            float Ez;
            cpu_cuComplex pi(PI , 0);
            cpu_cuComplex two(2.0,0.0);
            cpu_cuComplex negativeone(-1.0,0);
            cpu_cuComplex deltatime(dt,0);

            if(cpu_isOnyp(x,y))
            {
                Ez = (dev_Ez[getCell(x,y+1,nx+1)]+dev_Ez[getCell(x,y,nx+1)])/2;
                float Hx = dev_Hx[getCell(x,y,nx)];
                cjzyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Hx*deltatime*cuexp((float)(-1)*j*(float)2*pi*freq*(float)(*timestep)*deltatime);//cjzyp and cmxyp have nx - 2*NF2FFBoundary -2 elements

                cmxyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Ez*deltatime*cuexp((float)-1.0*j*(float)2.0*(float)PI*freq*((float)(*timestep)+0.5)*(float)dt);
            }
            else if(cpu_isOnxp(x))//X faces override y faces at their intersections
            {
                Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x+1,y,nx+1)])/2;
                float Hy = dev_Hy[getCell(x,y,nx)];

                cjzxp[index-(nx-2*NF2FFdistfromboundary-2)] += Hy*deltatime*cuexp(-1*j*2*pi*freq*(float)(*timestep)*(float)dt);//cjzxp and cmyxp have ny-2*NF2FFBound elements

                cmyxp[index-(nx-2*NF2FFdistfromboundary-2)] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*pi*freq*((float)(*timestep)+0.5)*(float)dt);// this is the discrete fourier transform, by the way.
            }
            else if(cpu_isOnyn(x,y))
            {  
                Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x,y+1,nx+1)])/2;
                float Hx=dev_Hx[getCell(x,y,nx)];

                cjzyn[index] += Hx*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt);	//cjzyn and cmxyn need to have nx-2*NF2FFbound-2 elements
                cmxyn[index] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*((float)(*timestep)+0.5)*(float)dt);
            }
            else if(cpu_isOnxn(x))
            {
                Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x+1,y,nx+1)])/2;
                cjzxn[index-(2*nx+ny-6*NF2FFdistfromboundary-4)] += -1*dev_Hy[getCell(x,y,nx)]*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt); // cjzxn and cmyxn must have ny-2*NFdistfromboundary elements
                cmyxn[index-(2*nx+ny-6*NF2FFdistfromboundary-4)] += -1*Ez*(float)dt*cuexp(-1.0*j*2.0*(float)PI*freq*((float)(*timestep)+0.5)*(float)dt);
            }
        }
    }

}

void float_Init(float* array, int size)
{
    for(int i =0;i<size;i++)
    {

        array[i] = 0.0;
    }


}


float cpu_fwf(float timestep,float x, float y,float Phi_inc,float l)
{

    float ar;
    float ky, kx;//k hat
    ky = sin(Phi_inc);
    kx = cos(Phi_inc);

    ar = (float)timestep*dt-(float)t0-(1/(float)c0)*(ky*y*dx+kx*x*dy-l);
    //ar = timestep*dt-t0;

    //return exp(-1*(ar*ar)/(tau*tau));// gaussian pulse  argument is k dot r, 
    return exp(-1*ar*ar/(tau*tau));
    //return sin(2*PI*1e9*timestep*dt);
}

void cpu_H_field_update(float*dev_Hy,float*dev_Hx,float*dev_Ez,float*dev_bmx,float*dev_Psi_hyx,float*dev_amx,float*dev_bmy,float*dev_amy,float*dev_Psi_hxy,float*kex)
{
    float buffer_Hy;
    float buffer_Hx;
    float Chez = (dt/dx)/(mu0);
    for(int x = 0; x<nx;x++)
    {
        for(int y = 0; y<ny;y++)
        {




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



                dev_Hx[getCell(x,y,nx)] = buffer_Hx;




                dev_Hy[getCell(x,y,nx)] = buffer_Hy;


                //dev_Hx[getCell(x,y,nx)] = buffer_Hx;
                //dev_Hy[getCell(x,y,nx)] = buffer_Hy;
            }
        }
    }
}


void E_field_update(int *i,float*dev_Ez,float*dev_Hy,float*dev_Hx,float*dev_Psi_ezx,float*dev_aex,float*dev_aey,float*dev_bex,float*dev_bey,float*dev_Psi_ezy,float*kex,float*Cezhy,float*Cezhx,float*Ceze,float*Cezeip,float*Cezeic,float Phi)
{
    for( int x = 0; x<(nx+1);x++)
    {
        for(int y = 0; y<(ny+1);y++)
        {
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



                        buffer_Ez = Ceze[getCell(x,y,nx+1)]*dev_Ez[getCell(x,y,nx+1)]+Cezhy[getCell(x,y,nx+1)]*(dev_Hy[getCell(x,y,nx)]-dev_Hy[getCell(x-1,y,nx)])
                            -Cezhx[getCell(x,y,nx+1)]*(dev_Hx[getCell(x,y,nx)]-dev_Hx[getCell(x,y-1,nx)])
                            +Cezeic[getCell(x,y,nx+1)]*cpu_fwf((float)(*i)+0.5,x,y,Phi,-breast_radius)
                            +Cezeip[getCell(x,y,nx+1)]*cpu_fwf((float)(*i)-0.5,x,y,Phi,-breast_radius);

                    }
                    else
                    {
                        buffer_Ez = Ceze[getCell(x,y,nx+1)]*dev_Ez[getCell(x,y,nx+1)]+Cezhy[getCell(x,y,nx+1)]*(dev_Hy[getCell(x,y,nx)]-dev_Hy[getCell(x-1,y,nx)])
                            -Cezhx[getCell(x,y,nx+1)]*(dev_Hx[getCell(x,y,nx)]-dev_Hx[getCell(x,y-1,nx)]);
                        if(x==(int)(source_x)&&y==(int)(source_y))
                        {
                            buffer_Ez=buffer_Ez + 100*Cezj*cpu_fwf((float)(*i),0,0,0,0);
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


                dev_Ez[getCell(x,y,nx+1)] = buffer_Ez;

                //dev_Ez[getCell(x,y,nx+1)] = buffer_Ez;
            }
        }
    }

}
// void E_inc_update(int *i,float*dev_Hy_inc,float*dev_Hx_inc,float*dev_Psi_ezx_inc,float*dev_aex,float*dev_aey,float*dev_bex,float*dev_bey,float*dev_Psi_ezy_inc,float*kex,float*dev_Ezip,float*dev_Ezic)
//{
//	int x=threadIdx.x+blockDim.x*blockIdx.x;
//	int y=threadIdx.y+blockDim.y*blockIdx.y;
////	int offset = x+y*blockDim.x*gridDim.x;
//	float buffer_Ez;
//	//float Ceh = (dt/dx)/(eps0);
//	float Cezj = -dt/eps0;
//	float Ceze = 1;
//	float Cezhy = (dt/(dx*eps0));
//
//	if(x<=nx&&y<=ny)
//	{
//	
//		//if(x==0||x==nx||y==0||y==ny)
//		if(x==nx||y==ny||x==0||y==0)
//		{
//			buffer_Ez=0.0;
//		}
//	else
//		{
//		buffer_Ez = Ceze*dev_Ezic[getCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)])
//		-Cezhy*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)]);
//		
//		if(x==((int)source_x)&&y==(int)(source_y))
//		{
//			//buffer_Ez=buffer_Ez + Cezj*dev_Jz[*i];
//			buffer_Ez=buffer_Ez + 100*Cezj*fwf((float)(*i),0,0,0,0);
//		}
//		if(x<=ncells&&x!=0)
//		{
//			buffer_Ez = Ceze*dev_Ezic[getCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)])/kex[ncells-x]
//		-Cezhy*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)])/kex[ncells-x];
//			dev_Psi_ezx_inc[getCell(x-1,y-1,20)] = dev_bex[ncells-x]*dev_Psi_ezx_inc[getCell(x-1,y-1,20)]+dev_aex[ncells-x]*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)]);
//			buffer_Ez += Cezhy*dx*dev_Psi_ezx_inc[getCell(x-1,y-1,2*ncells)];
//		}
//		if(x>=(nx-ncells)&&x!=nx)
//		{
//				buffer_Ez = Ceze*dev_Ezic[getCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)])/kex[x-nx+ncells]
//		-Cezhy*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)])/kex[x-nx+ncells];
//			dev_Psi_ezx_inc[getCell(x-nx+20,y-1,20)]=dev_bex[x-nx+ncells]*dev_Psi_ezx_inc[getCell(x-nx+20,y-1,20)]+dev_aex[x-nx+ncells]*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)]);
//			buffer_Ez+=Cezhy*dx*dev_Psi_ezx_inc[getCell(x-nx+20,y-1,2*ncells)];
//		}
//		if(y<=ncells&&y!=0)
//		{
//				buffer_Ez = Ceze*dev_Ezic[getCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)])/kex[ncells-y]
//		-Cezhy*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)])/kex[ncells-y];
//			dev_Psi_ezy_inc[getCell(x-1,y-1,nx)]=dev_bey[(ncells-y)]*dev_Psi_ezy_inc[getCell(x-1,y-1,nx)]+dev_aey[(ncells-y)]*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)]);
//			buffer_Ez-=Cezhy*dy*dev_Psi_ezy_inc[getCell(x-1,y-1,nx)];
//		}
//		if(y>=(ny-ncells)&&y!=ny)
//		{
//			buffer_Ez = Ceze*dev_Ezic[getCell(x,y,nx+1)]+Cezhy*(dev_Hy_inc[getCell(x,y,nx)]-dev_Hy_inc[getCell(x-1,y,nx)])/kex[y-ny+ncells]
//		-Cezhy*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)])/kex[y-ny+ncells];
//			dev_Psi_ezy_inc[getCell(x-1,y-ny+20,nx)]=dev_bey[y-ny+ncells]*dev_Psi_ezy_inc[getCell(x-1,y-ny+20,nx)]+dev_aey[y-ny+ncells]*(dev_Hx_inc[getCell(x,y,nx)]-dev_Hx_inc[getCell(x,y-1,nx)]);
//			buffer_Ez-=Cezhy*dy*dev_Psi_ezy_inc[getCell(x-1,y-ny+20,nx)];
//		}
//		}
//		dev_Ezip[getCell(x,y,nx+1)] = dev_Ezic[getCell(x,y,nx+1)];
//		dev_Ezic[getCell(x,y,nx+1)] = buffer_Ez;
//	}
//
//}
float calc_radiated_power(cpu_cuComplex *cjzxp,cpu_cuComplex *cjzyp,cpu_cuComplex *cjzxn,cpu_cuComplex *cjzyn,cpu_cuComplex *cmxyp,cpu_cuComplex *cmyxp,cpu_cuComplex *cmxyn,cpu_cuComplex *cmyxn)
{
    int indexofleg1 = nx-2*NF2FFdistfromboundary-2;
    int indexofleg2 = nx+ny-4*NF2FFdistfromboundary-2;
    int indexofleg3 = 2*nx+ny-6*NF2FFdistfromboundary-4;
    int maxindex = 2*nx-8*NF2FFdistfromboundary+2*ny-4;
    int index;
    cpu_cuComplex cjz(0,0);
    cpu_cuComplex power = 0;

    for(index = 0; index<indexofleg1;index++)
    {   cjz = cpu_cuComplex(cjzyn[index].r,-1.0*cjzyn[index].i);//conjugation
        //z x x = y dot -y = -1
        power+=-1.0*cjz*cmxyn[index]*dx;// the negative one comes from the dot product between JxM and the n hat vector
    }
    for(index = indexofleg1; index<indexofleg2;index++)
    {
        cjz = cpu_cuComplex(cjzxp[index-indexofleg1].r,-1.0*cjzxp[index-indexofleg1].i);//making the conjugate
        // z cross y = -x dot x = -1
        power+= -1.0*cjz*cmyxp[index-indexofleg1]*dy;//positive x unit normal vector
    }
    for(index = indexofleg2;index<indexofleg3;index++)
    {
        // z cross x = y dot y = 1
        cjz = cpu_cuComplex(cjzyp[index-indexofleg2].r,-1.0*cjzyp[index-indexofleg2].i);
        power+= cjz*cmxyp[index-indexofleg2]*dx;//postive y unit normal vector
    }
    for(index = indexofleg3;index<maxindex;index++)
    {
        // z cross y = -x dot -x = 1 
        cjz = cpu_cuComplex(cjzxn[index-indexofleg3].r,-1.0*cjzxn[index-indexofleg3].i);
        power += cjz*cmyxn[index-indexofleg3]*dy;// negative x hat n vector
    }
    float realpower = power.r;
    realpower *= 0.5;
    return realpower;
}

// void calculate_JandM_total(float* f,int* timestep,float*dev_Ez,float*dev_Hy,float*dev_Hx,cpu_cuComplex *cjzxp,cpu_cuComplex *cjzyp,cpu_cuComplex*cjzxn,cpu_cuComplex*cjzyn,cpu_cuComplex*cmxyp,cpu_cuComplex*cmyxp,cpu_cuComplex*cmxyn,cpu_cuComplex*cmyxn,float*dev_Ezic,float*dev_Ezip,float*dev_Hx_inc,float*dev_Hy_inc)
//{
//	float freq = *f;
//	int index = threadIdx.x+blockIdx.x*blockDim.x;// should launch 2*nx-8*NF2FFdistfromboundary+2*ny-4 threads. 
//	if(index<=size_NF2FF_total)
//	{
//	const cpu_cuComplex j(0.0,1.0);
//	int x = getxfromthreadIdNF2FF(index);
//	int y = getyfromthreadIdNF2FF(index);
//	
//	float Ez;
//	cpu_cuComplex pi(PI , 0);
//	cpu_cuComplex two(2.0,0.0);
//	cpu_cuComplex negativeone(-1.0,0);
//	cpu_cuComplex deltatime(dt,0);
//	
//		if(isOnyp(x,y))
//		{
//		Ez = (dev_Ez[getCell(x,y+1,nx+1)]+dev_Ez[getCell(x,y,nx+1)])/2;
//		Ez += (dev_Ezic[getCell(x,y+1,nx+1)] + dev_Ezic[getCell(x,y,nx+1)] + dev_Ezip[getCell(x,y+1,nx+1)] + dev_Ezip[getCell(x,y,nx+1)])/4;
//		float Hx = dev_Hx[getCell(x,y,nx)] + dev_Hx_inc[getCell(x,y,nx)];
//		cjzyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Hx*deltatime*cuexp((float)(-1)*j*(float)2*pi*freq*(float)(*timestep)*deltatime);//cjzyp and cmxyp have nx - 2*NF2FFBoundary -2 elements
//		cmxyp[index-(nx+ny-4*NF2FFdistfromboundary-2)] += -1*Ez*deltatime*cuexp((float)-1.0*j*(float)2.0*(float)PI*freq*((float)(*timestep)-0.5)*(float)dt);
//		}
//		else if(isOnxp(x))//X faces override y faces at their intersections
//		{
//		Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x+1,y,nx+1)])/2;
//		Ez += (dev_Ezic[getCell(x+1,y,nx+1)] + dev_Ezic[getCell(x,y,nx+1)] + dev_Ezip[getCell(x+1,y,nx+1)] + dev_Ezip[getCell(x,y,nx+1)])/4;
//			float Hy = dev_Hy[getCell(x,y,nx)] + dev_Hy_inc[getCell(x,y,nx)];
//			
//			cjzxp[index-(nx-2*NF2FFdistfromboundary-2)] += Hy*deltatime*cuexp(-1*j*2*pi*freq*(float)(*timestep)*(float)dt);//cjzxp and cmyxp have ny-2*NF2FFBound elements
//			
//			cmyxp[index-(nx-2*NF2FFdistfromboundary-2)] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*pi*freq*((float)(*timestep)-0.5)*(float)dt);// this is the discrete fourier transform, by the way.
//		}
//		else if(isOnyn(x,y))
//		{  
//			Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x,y+1,nx+1)])/2;
//			Ez += (dev_Ezic[getCell(x,y+1,nx+1)] + dev_Ezic[getCell(x,y,nx+1)] + dev_Ezip[getCell(x,y+1,nx+1)] + dev_Ezip[getCell(x,y,nx+1)])/4;
//			float Hx=dev_Hx[getCell(x,y,nx)]+dev_Hx_inc[getCell(x,y,nx)];
//		
//			cjzyn[index] += Hx*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt);	//cjzyn and cmxyn need to have nx-2*NF2FFbound-2 elements
//			cmxyn[index] += Ez*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*((float)(*timestep)-0.5)*(float)dt);
//		}
//		else if(isOnxn(x))
//		{
//			Ez = (dev_Ez[getCell(x,y,nx+1)]+dev_Ez[getCell(x+1,y,nx+1)])/2;
//			Ez += (dev_Ezic[getCell(x+1,y,nx+1)] + dev_Ezic[getCell(x,y,nx+1)] + dev_Ezip[getCell(x+1,y,nx+1)] + dev_Ezip[getCell(x,y,nx+1)])/4;
//			float Hy = dev_Hy[getCell(x,y,nx)] + dev_Hy_inc[getCell(x,y,nx)];
//			cjzxn[index-(2*nx+ny-6*NF2FFdistfromboundary-4)] += -1*Hy*(float)dt*cuexp((float)(-1)*j*(float)2.0*(float)PI*freq*(float)(*timestep)*(float)dt); // cjzxn and cmyxn must have ny-2*NFdistfromboundary elements
//			cmyxn[index-(2*nx+ny-6*NF2FFdistfromboundary-4)] += -1*Ez*(float)dt*cuexp(-1.0*j*2.0*(float)PI*freq*((float)(*timestep)-0.5)*(float)dt);
//		}
//	}
//	
//}

int cpu_getOptimizationCell(int x, int y)
{
    int x_coord,y_coord;
    x_coord = (x-(nx/2-(int)(breast_radius/dx)))/(2*breast_radius/(9*dx));
    y_coord = (y-(ny/2-breast_radius/dy))/(2*breast_radius/(9*dy));//the optimization space is 216 FDTD cells wide and high. //The optimization space is split into 25 by 25 optimization cells. 
    //each optimization cell has 24 by 24 FDTD cells within it. That's what the 108, 24 and 25 are about.  
    return x_coord+9*y_coord;//The max return should be, 9*9-1, hopefully.
}
void N2FPostProcess (float* D,float f, cpu_cuComplex *N,cpu_cuComplex *L,cpu_cuComplex *cjzxp,cpu_cuComplex *cjzyp,cpu_cuComplex *cjzxn,cpu_cuComplex *cjzyn,cpu_cuComplex *cmxyp,cpu_cuComplex *cmyxp,cpu_cuComplex *cmxyn,cpu_cuComplex *cmyxn)
{
    int indexofleg1 = nx-2*NF2FFdistfromboundary-2;
    int indexofleg2 = nx+ny-4*NF2FFdistfromboundary-2;
    int indexofleg3 = 2*nx+ny-6*NF2FFdistfromboundary-4;
    int maxindex = 2*nx-8*NF2FFdistfromboundary+2*ny-4;
    int x,y;

    float rhoprime;
    float Psi;
    int Phi_index;
    cpu_cuComplex  Mphi(0,0);
    float Phi;


    float k = 2*PI*f/c0;
    cpu_cuComplex  negativeone(-1.0,0.0);
    int index = 0;
    cpu_cuComplex jcmpx(0,1);
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
void CJ_Init(cpu_cuComplex * cjzyn,int size);
void cpu_scattered_parameter_init(float*eps_r_z,float*sigma_e_z,float*Cezeic,float*Cezeip);

double FDTD_CPU(const vector<double> &arguments)
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

    vector<float> image;
    for(int lerp = 0; lerp<81;lerp++)//This is setting the material parameters of the optimization cells.
    {
        image.push_back((float)arguments.at(lerp));
    }
    for(int lerp = 81; lerp<81*2;lerp++)
    {
        image.push_back((float)arguments.at(lerp));
    }

    float*Ceze,*Cezhy,*Cezhx,*dev_Cezeic,*dev_Cezeip,*Ez,*eps_r_z,*sigma_e_z,*Hy,*Hx,
        *kex,*aex,*bex,*amx,*bmx,*alpha_e,*alpha_m,*sigma_e_pml,*sigma_m_pml
            ,*Psi_ezy,*Psi_ezx,*Psi_hyx,*Psi_hxy,*kmx;
    float* dev_sigma_e_z,*dev_eps_r_z;
    float freq = center_freq;
    float *dev_freq,*D;

    cpu_cuComplex *hcjzxp,*hcjzyp,*hcjzxn,*hcjzyn,*hcmxyp,*hcmyxp,*hcmxyn,*hcmyxn;


    dev_Cezeic = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    dev_Cezeip = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    Ceze = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    Cezhy = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    Cezhx = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));

    Ez = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    eps_r_z =  (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    sigma_e_z = (float*)malloc(sizeof(float)*(1+nx)*(1+ny));
    D = (float*)malloc(sizeof(float)*numberofobservationangles);
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

    hcjzyp = (cpu_cuComplex*)malloc(sizeof(cpu_cuComplex )*size_cjzy);
    hcjzyn = (cpu_cuComplex *)malloc(sizeof(cpu_cuComplex )*size_cjzy);
    hcjzxp = (cpu_cuComplex *)malloc(sizeof(cpu_cuComplex )*size_cjzx);
    hcjzxn = (cpu_cuComplex *)malloc(sizeof(cpu_cuComplex )*size_cjzx);
    hcmxyn = (cpu_cuComplex *)malloc(sizeof(cpu_cuComplex )*size_cjzy);
    hcmxyp = (cpu_cuComplex *)malloc(sizeof(cpu_cuComplex )*size_cjzy);
    hcmyxp = (cpu_cuComplex *)malloc(sizeof(cpu_cuComplex )*size_cjzx);
    hcmyxn = (cpu_cuComplex *)malloc(sizeof(cpu_cuComplex )*size_cjzx);

    CJ_Init(hcjzyp,size_cjzy);//C**** coefficients are for surface current/ field duality for NF2FF processing.
    CJ_Init(hcjzyn,size_cjzy);
    CJ_Init(hcjzxp,size_cjzx);
    CJ_Init(hcjzxn,size_cjzx);
    CJ_Init(hcmxyn,size_cjzy);
    CJ_Init(hcmxyp,size_cjzy);
    CJ_Init(hcmyxp,size_cjzx);
    CJ_Init(hcmyxn,size_cjzx);
    Psi_ezy_init(Psi_ezy);
    Psi_ezx_init(Psi_ezx);
    Psi_hyx_init(Psi_hyx);
    Psi_hxy_init(Psi_hxy);
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



    cpu_scattered_parameter_init(eps_r_z,sigma_e_z,dev_Cezeic,dev_Cezeip);

    cpu_cuComplex *L,*N;

    L  = (cpu_cuComplex*)malloc(sizeof(cpu_cuComplex)*size_NF2FF_total);
    N = (cpu_cuComplex*)malloc(sizeof(cpu_cuComplex)*size_NF2FF_total);

    float measurement[numberofobservationangles*numberofexcitationangles] = {0.544912 , 0.518606 , 0.439233 , 0.330533 , 0.219116 , 0.135115 , 0.0923969 , 0.0774134 , 0.0740459 , 0.0739238 , 0.0660047 , 0.0465372 , 0.0248307 , 0.00913681 , 0.00186162 , 0.0038402 , 0.0130785 , 0.0238094 , 0.0312918 , 0.035705 , 0.0388307 , 0.039513 , 0.0368443 , 0.0338221 , 0.0324815 , 0.0305907 , 0.0270149 , 0.0239178 , 0.0224438 , 0.021849 , 0.0217346 , 0.0222152 , 0.023146 , 0.0245181 , 0.0267161 , 0.0286964 , 0.0276803 , 0.0235098 , 0.0197177 , 0.0183168 , 0.0196998 , 0.0261493 , 0.0375584 , 0.0479223 , 0.0511598 , 0.0461443 , 0.035713 , 0.0249863 , 0.0203708 , 0.0260456 , 0.0395441 , 0.054163 , 0.0660136 , 0.0763823 , 0.0935922 , 0.132053 , 0.201299 , 0.299247 , 0.410792 , 0.504467 ,
        0.0490085 , 0.0278468 , 0.0123693 , 0.00899709 , 0.0196632 , 0.0401112 , 0.0623734 , 0.0809561 , 0.096057 , 0.113814 , 0.145125 , 0.200388 , 0.283438 , 0.386362 , 0.486139 , 0.549594 , 0.547993 , 0.475775 , 0.358033 , 0.230962 , 0.118935 , 0.039843 , 0.00700227 , 0.0112335 , 0.0300356 , 0.0494414 , 0.0605159 , 0.0585777 , 0.0503323 , 0.045704 , 0.0474064 , 0.0523123 , 0.0558987 , 0.0545722 , 0.0475098 , 0.0366045 , 0.0248037 , 0.0155752 , 0.0115322 , 0.0127167 , 0.0176523 , 0.0243556 , 0.0310764 , 0.037444 , 0.0432292 , 0.0469609 , 0.0471761 , 0.0435653 , 0.0369347 , 0.0293987 , 0.0235478 , 0.0206039 , 0.020754 , 0.0247748 , 0.0336772 , 0.047007 , 0.0618746 , 0.0734482 , 0.0763332 , 0.0674785 ,
        0.0463129 , 0.0448933 , 0.0398454 , 0.0319834 , 0.0239428 , 0.0174267 , 0.0129155 , 0.0116624 , 0.0154122 , 0.0247183 , 0.0376821 , 0.0494142 , 0.0552493 , 0.0544909 , 0.0501016 , 0.0466044 , 0.047395 , 0.0522298 , 0.0576919 , 0.0588555 , 0.0504011 , 0.0311956 , 0.0107719 , 0.00755493 , 0.0394798 , 0.116099 , 0.232324 , 0.36478 , 0.478314 , 0.541685 , 0.541186 , 0.484009 , 0.391878 , 0.291105 , 0.204554 , 0.145352 , 0.113254 , 0.0973423 , 0.0835717 , 0.0637299 , 0.0397899 , 0.0189781 , 0.00814281 , 0.0118845 , 0.0291142 , 0.0513172 , 0.0680543 , 0.0744519 , 0.0718442 , 0.0622228 , 0.0473734 , 0.0329352 , 0.0245156 , 0.0212818 , 0.0204027 , 0.0228792 , 0.0298908 , 0.0380399 , 0.0432513 , 0.0455291 ,
        0.0469428 , 0.049667 , 0.0453111 , 0.0370016 , 0.0278006 , 0.0201062 , 0.0173687 , 0.020228 , 0.0242543 , 0.0264199 , 0.0275476 , 0.027771 , 0.0262174 , 0.0237332 , 0.0219206 , 0.0212424 , 0.0214967 , 0.0226845 , 0.0248514 , 0.0275874 , 0.0300439 , 0.0318892 , 0.0340621 , 0.0369823 , 0.0388068 , 0.0379494 , 0.0350817 , 0.030462 , 0.0230471 , 0.0133404 , 0.00457234 , 0.00152755 , 0.00874873 , 0.0260448 , 0.0463293 , 0.0633742 , 0.0751071 , 0.0775575 , 0.0756597 , 0.0916989 , 0.141021 , 0.22185 , 0.328433 , 0.44207 , 0.524772 , 0.544711 , 0.498668 , 0.407614 , 0.29953 , 0.199594 , 0.128704 , 0.0929922 , 0.0772499 , 0.0654169 , 0.0536587 , 0.0399619 , 0.0255793 , 0.0193488 , 0.0253531 , 0.0373143 ,
    };//I've just hardcoded the measurement values. 

    CJ_Init(L,size_NF2FF_total);
    CJ_Init(N,size_NF2FF_total);
    float test_Ez;
    float Phi;
    float fit =0;
    /* The calculation part! */
    for(int Phi_index = 0; Phi_index<numberofexcitationangles;Phi_index++)
    {
        Phi = (float)Phi_index*2*PI/numberofexcitationangles;

        for(int i=0;i<number_of_time_steps;i++)
        {
            cpu_H_field_update(Hy,Hx,Ez,bmx,Psi_hyx,amx,bmx,amx,Psi_hxy,kmx);

            E_field_update(&i,Ez,Hy,Hx,Psi_ezx,aex,aex,bex,bex,Psi_ezy,kex,Cezhy,Cezhy,Ceze,dev_Cezeip,dev_Cezeic,Phi);

            calculate_JandM(&freq, &i,Ez,Hy,Hx,hcjzxp,hcjzyp,hcjzxn,hcjzyn,hcmxyp,hcmyxp,hcmxyn,hcmyxn);

            //	    cout << "[" << i << "/" << number_of_time_steps << "] " << Ez[getCell(nx/2,ny/2,nx+1)] << std::endl;
        }


        N2FPostProcess(D, freq,N,L,hcjzxp,hcjzyp,hcjzxn,hcjzyn,hcmxyp,hcmyxp,hcmxyn,hcmyxn);
        for(int angle_index=0;angle_index < numberofobservationangles;angle_index++)
        {
            fit -=pow(D[angle_index]-measurement[angle_index+Phi_index*numberofobservationangles],2)/pow(measurement[angle_index+Phi_index*numberofobservationangles],2);

        }
        CJ_Init(L,size_NF2FF_total);
        CJ_Init(N,size_NF2FF_total);
        float_Init(Hy,(nx*ny));
        float_Init(Hx,nx*ny);
        float_Init(Ez,(nx+1)*(ny+1));
        float_Init(Psi_hyx,20*ny);
        float_Init(Psi_hxy,20*nx);
        float_Init(Psi_ezx,20*nx);
        float_Init(Psi_ezy,20*nx);
        CJ_Init(hcjzyp,size_cjzy);//C**** coefficients are for surface current/ field duality for NF2FF processing.
        CJ_Init(hcjzyn,size_cjzy);
        CJ_Init(hcjzxp,size_cjzx);
        CJ_Init(hcjzxn,size_cjzx);
        CJ_Init(hcmxyn,size_cjzy);
        CJ_Init(hcmxyp,size_cjzy);
        CJ_Init(hcmyxp,size_cjzx);
        CJ_Init(hcmyxn,size_cjzx);
    }






    //	cout<<" fitness = "<<fit<<endl;

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
    free(dev_Cezeip);
    free(dev_Cezeic);
    free(L);
    free(N);

    //	system("pause");

    return (double)fit;
}
void cpu_scattered_parameter_init(float*eps_r_z,float*sigma_e_z,float*Cezeic,float*Cezeip)
{
    for(int x = 0; x<(nx+1);x++)
    {
        for(int y = 0; y<(ny+1);y++)
        {

            Cezeic[getCell(x,y,nx+1)] = (2*(eps0-eps0*eps_r_z[getCell(x,y,nx+1)])-sigma_e_z[getCell(x,y,nx+1)]*dt)/(2*eps0*eps_r_z[getCell(x,y,nx+1)]+sigma_e_z[getCell(x,y,nx+1)]*dt);
            Cezeip[getCell(x,y,nx+1)] = -1*(2*(eps0-eps0*eps_r_z[getCell(x,y,nx+1)])+sigma_e_z[getCell(x,y,nx+1)]*dt)/(2*eps0*eps_r_z[getCell(x,y,nx+1)]+sigma_e_z[getCell(x,y,nx+1)]*dt);

        }
    }
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
                eps_r_z[getCell(i,j,size)] = (float)argument.at(cpu_getOptimizationCell(i,j));
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
                sigma_e_z[getCell(i,j,size)] = (float)argument.at(cpu_getOptimizationCell(i,j)+9*9);
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
void CJ_Init(cpu_cuComplex * cjzyn,int size)
{
    cpu_cuComplex nullComplex(0,0);
    for( int i =0; i<size;i++)
    {
        cjzyn[i] = nullComplex;
    }
}

