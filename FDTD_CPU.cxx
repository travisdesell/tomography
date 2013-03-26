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
     // image.push_back(5);
    //  if(lerp == 81/2){
    //  image.push_back(40);
    //  }

    }
    for(int lerp = 81; lerp<81*2;lerp++)
    {
       image.push_back((float)arguments.at(lerp));
   //    image.push_back(0.005);
   //    if(lerp == 81/2){
    //   image.push_back(1.1);
    //   }
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
    D = (float*)malloc(sizeof(float)*numberofobservationangles*numberofexcitationangles*numberoffrequencies);
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

   
  float measurement[numberoffrequencies*numberofexcitationangles*numberofobservationangles] = {  0.0368949 , 0.0368559 , 0.0367595 , 0.0366061 , 0.0363965 , 0.036132 , 0.0358152 , 0.0354486 , 0.0350362 , 0.0345823 , 0.0340917 , 0.0335699 , 0.0330223 , 0.0324543 , 0.031871 , 0.0312776 , 0.030678 , 0.0300763 , 0.0294755 , 0.0288784 , 0.0282873 , 0.0277044 , 0.0271314 , 0.0265704 , 0.0260232 , 0.0254919 , 0.0249788 , 0.0244862 , 0.0240164 , 0.0235718 , 0.0231545 , 0.0227665 , 0.0224092 , 0.0220835 , 0.0217896 , 0.0215276 , 0.021296 , 0.0210935 , 0.0209179 , 0.0207667 , 0.0206373 , 0.0205271 , 0.0204336 , 0.0203545 , 0.0202882 , 0.0202333 , 0.020189 , 0.0201548 , 0.0201308 , 0.0201168 , 0.0201134 , 0.0201209 , 0.0201395 , 0.0201697 , 0.020212 , 0.0202668 , 0.0203346 , 0.0204162 , 0.0205126 , 0.020625 , 0.0207548 , 0.0209038 , 0.0210739 , 0.0212668 , 0.0214847 , 0.021729 , 0.0220013 , 0.0223024 , 0.0226331 , 0.0229933 , 0.0233826 , 0.0238001 , 0.0242444 , 0.0247139 , 0.0252067 , 0.0257209 , 0.0262544 , 0.0268051 , 0.0273711 , 0.0279501 , 0.02854 , 0.0291383 , 0.0297425 , 0.0303495 , 0.0309556 , 0.0315569 , 0.032149 , 0.0327269 , 0.0332855 , 0.0338196 , 0.034324 , 0.0347938 , 0.0352243 , 0.0356118 , 0.0359526 , 0.0362442 , 0.036484 , 0.0366702 , 0.0368014 , 0.0368765 , 0.14288 , 0.142009 , 0.13988 , 0.136562 , 0.132154 , 0.126784 , 0.120595 , 0.11374 , 0.106366 , 0.0986116 , 0.0906011 , 0.082443 , 0.0742347 , 0.0660669 , 0.0580294 , 0.0502142 , 0.0427154 , 0.0356273 , 0.0290393 , 0.0230309 , 0.0176672 , 0.0129955 , 0.0090456 , 0.00583043 , 0.00334961 , 0.0015932 , 0.000545888 , 0.000190665 , 0.000511585 , 0.00149506 , 0.0031294 , 0.0054025 , 0.00829768 , 0.0117884 , 0.0158334 , 0.0203724 , 0.0253256 , 0.0305956 , 0.0360727 , 0.0416418 , 0.0471886 , 0.0526039 , 0.0577865 , 0.0626425 , 0.0670847 , 0.0710327 , 0.0744125 , 0.0771573 , 0.0792117 , 0.0805309 , 0.0810846 , 0.0808558 , 0.0798418 , 0.0780536 , 0.0755144 , 0.0722618 , 0.0683483 , 0.063845 , 0.0588421 , 0.0534493 , 0.0477937 , 0.0420135 , 0.0362497 , 0.030637 , 0.0252957 , 0.0203261 , 0.0158062 , 0.0117935 , 0.00832962 , 0.00544492 , 0.00316456 , 0.00151185 , 0.000510402 , 0.000183898 , 0.00055459 , 0.00164087 , 0.00345469 , 0.00599956 , 0.00926927 , 0.0132478 , 0.0179098 , 0.0232225 , 0.0291465 , 0.0356371 , 0.0426436 , 0.0501092 , 0.0579675 , 0.0661402 , 0.074535 , 0.0830424 , 0.0915361 , 0.0998744 , 0.107903 , 0.115458 , 0.122378 , 0.128504 , 0.13369 , 0.137808 , 0.140756 , 0.142461 , 0.312472 , 0.308745 , 0.299428 , 0.285123 , 0.266631 , 0.244849 , 0.220681 , 0.194978 , 0.168529 , 0.14209 , 0.116418 , 0.0922796 , 0.0704159 , 0.0514732 , 0.0359173 , 0.0239777 , 0.0156374 , 0.0106746 , 0.0087327 , 0.00938943 , 0.0122032 , 0.0167283 , 0.0225085 , 0.0290647 , 0.0358927 , 0.0424783 , 0.0483255 , 0.0529947 , 0.0561345 , 0.0575064 , 0.0569966 , 0.0546192 , 0.050516 , 0.0449543 , 0.038321 , 0.031105 , 0.0238581 , 0.0171387 , 0.0114472 , 0.00717492 , 0.00457355 , 0.00374602 , 0.0046478 , 0.00709259 , 0.0107605 , 0.0152147 , 0.0199313 , 0.0243486 , 0.0279308 , 0.0302407 , 0.0310022 , 0.030138 , 0.0277737 , 0.0242089 , 0.0198671 , 0.0152403 , 0.0108411 , 0.00716555 , 0.00466237 , 0.00369775 , 0.00451352 , 0.00718463 , 0.0115927 , 0.0174309 , 0.0242443 , 0.031495 , 0.0386312 , 0.045141 , 0.0505834 , 0.0546033 , 0.0569389 , 0.0574359 , 0.0560634 , 0.052925 , 0.0482579 , 0.0424158 , 0.0358367 , 0.0290076 , 0.0224359 , 0.0166315 , 0.0121007 , 0.00934174 , 0.00883497 , 0.01102 , 0.0162612 , 0.0248113 , 0.0367833 , 0.052141 , 0.0707033 , 0.0921578 , 0.116069 , 0.141876 , 0.168883 , 0.196253 , 0.223009 , 0.248072 , 0.270324 , 0.288689 , 0.302242 , 0.310295 , 0.490798 , 0.481086 , 0.456034 , 0.418325 , 0.371346 , 0.318661 , 0.263707 , 0.209724 , 0.159743 , 0.116406 , 0.0816059 , 0.0560944 , 0.0393568 , 0.029886 , 0.0257449 , 0.0251173 , 0.0265955 , 0.0291539 , 0.0319574 , 0.0342056 , 0.0351257 , 0.0340982 , 0.0308268 , 0.0254637 , 0.0186428 , 0.011414 , 0.00508582 , 0.000987973 , 0.000194484 , 0.00329069 , 0.010281 , 0.0206646 , 0.0335904 , 0.0479564 , 0.0624108 , 0.0753471 , 0.0850664 , 0.0901865 , 0.0901476 , 0.0854589 , 0.0774423 , 0.0676346 , 0.0573093 , 0.0473806 , 0.0385559 , 0.0314367 , 0.02643 , 0.0235745 , 0.02247 , 0.0224238 , 0.0227607 , 0.0231267 , 0.0236303 , 0.0247769 , 0.0272607 , 0.031718 , 0.0385129 , 0.0475761 , 0.0583021 , 0.0695468 , 0.0797806 , 0.0874071 , 0.0911504 , 0.0903422 , 0.0849916 , 0.0756633 , 0.0633016 , 0.0491193 , 0.0345432 , 0.0211187 , 0.0102999 , 0.00316698 , 0.000190973 , 0.00115523 , 0.00525722 , 0.0113373 , 0.0181471 , 0.024579 , 0.0298077 , 0.0333319 , 0.0349489 , 0.0347126 , 0.0329208 , 0.0301316 , 0.0271856 , 0.0252114 , 0.0256065 , 0.0299761 , 0.0400018 , 0.0572164 , 0.0827203 , 0.116935 , 0.159465 , 0.209059 , 0.263596 , 0.320056 , 0.374595 , 0.422833 , 0.460424 , 0.483793 , 0.370158 , 0.355927 , 0.318773 , 0.265129 , 0.203198 , 0.141226 , 0.0863911 , 0.044169 , 0.0176395 , 0.00669789 , 0.00797982 , 0.0161989 , 0.0263423 , 0.035208 , 0.04142 , 0.044591 , 0.0448463 , 0.0429856 , 0.0405759 , 0.0394241 , 0.0407266 , 0.0445806 , 0.0501528 , 0.056237 , 0.0617629 , 0.066016 , 0.0685801 , 0.0691333 , 0.0672532 , 0.0624118 , 0.0542988 , 0.0433837 , 0.0312543 , 0.0202208 , 0.012298 , 0.00840934 , 0.00842638 , 0.0117146 , 0.0174405 , 0.0244411 , 0.0311703 , 0.0362155 , 0.0390325 , 0.0400738 , 0.0401951 , 0.0400433 , 0.0399137 , 0.0399027 , 0.0400191 , 0.0401745 , 0.0402027 , 0.0399882 , 0.0396097 , 0.0393436 , 0.0394893 , 0.0401256 , 0.0409303 , 0.0411731 , 0.0399461 , 0.0366013 , 0.0311632 , 0.0244142 , 0.0175844 , 0.0119435 , 0.00862112 , 0.0085998 , 0.0125547 , 0.0204532 , 0.0312649 , 0.0431681 , 0.0541873 , 0.0628184 , 0.0682885 , 0.0704607 , 0.0696182 , 0.066299 , 0.0612112 , 0.0551893 , 0.0491583 , 0.04407 , 0.0407526 , 0.0396597 , 0.0406227 , 0.0427885 , 0.0448057 , 0.0451628 , 0.0425544 , 0.0362959 , 0.0268359 , 0.0162063 , 0.00806029 , 0.0070702 , 0.017836 , 0.0436581 , 0.0854602 , 0.141035 , 0.204839 , 0.26859 , 0.322744 , 0.358559 , 1.0196 , 0.995792 , 0.930291 , 0.830029 , 0.704524 , 0.567427 , 0.435496 , 0.322567 , 0.233356 , 0.16454 , 0.112194 , 0.0760858 , 0.0560618 , 0.0476747 , 0.0445579 , 0.0436921 , 0.045494 , 0.049439 , 0.0522519 , 0.0506685 , 0.044197 , 0.0346295 , 0.0240318 , 0.0139247 , 0.00569666 , 0.000816103 , 0.000439157 , 0.0051178 , 0.0149409 , 0.0295689 , 0.0476754 , 0.0664623 , 0.0825927 , 0.0941162 , 0.101064 , 0.10405 , 0.102744 , 0.0962619 , 0.0853772 , 0.0732744 , 0.0632541 , 0.0563854 , 0.0515001 , 0.0471031 , 0.0431124 , 0.0405127 , 0.0398697 , 0.0406677 , 0.0417648 , 0.0422558 , 0.041956 , 0.0411863 , 0.0403362 , 0.0397411 , 0.0398222 , 0.0410562 , 0.0436067 , 0.0471286 , 0.051321 , 0.0567711 , 0.0647158 , 0.075412 , 0.0869978 , 0.0966032 , 0.102501 , 0.104542 , 0.102617 , 0.0957387 , 0.0831858 , 0.0660364 , 0.0470654 , 0.0293088 , 0.0149366 , 0.00513799 , 0.000437307 , 0.000822738 , 0.00571422 , 0.0140181 , 0.0242708 , 0.0347556 , 0.0436336 , 0.0493212 , 0.0511691 , 0.0499245 , 0.0473417 , 0.0452027 , 0.0449214 , 0.0482907 , 0.0582359 , 0.0783844 , 0.112235 , 0.163152 , 0.23441 , 0.327863 , 0.441914 , 0.5706 , 0.704026 , 0.829204 , 0.931518 , 0.997719 , 1.56145 , 1.51034 , 1.36516 , 1.14686 , 0.886904 , 0.623504 , 0.392075 , 0.213825 , 0.0932826 , 0.025457 , 0.000977688 , 0.00539734 , 0.0218678 , 0.0384618 , 0.0500334 , 0.054048 , 0.0497192 , 0.0403845 , 0.031607 , 0.0264021 , 0.0244731 , 0.0249787 , 0.0277699 , 0.0323437 , 0.0370983 , 0.0397912 , 0.0384856 , 0.0326657 , 0.0239511 , 0.015168 , 0.00843287 , 0.00504587 , 0.00717547 , 0.0173926 , 0.0347454 , 0.0538605 , 0.070558 , 0.0839177 , 0.0911281 , 0.0866927 , 0.0698295 , 0.0474007 , 0.0265803 , 0.0104582 , 0.0012696 , 0.00157218 , 0.0109013 , 0.0246855 , 0.0378159 , 0.0473432 , 0.0516075 , 0.0491309 , 0.0395407 , 0.0251207 , 0.0107338 , 0.00168816 , 0.00133616 , 0.0104282 , 0.0277634 , 0.0501158 , 0.0718446 , 0.0867578 , 0.0915146 , 0.0864343 , 0.0733839 , 0.0548091 , 0.0346413 , 0.0177613 , 0.00757505 , 0.00478715 , 0.00825093 , 0.0159005 , 0.0250096 , 0.0327749 , 0.0373846 , 0.0385143 , 0.0368226 , 0.0332706 , 0.0289983 , 0.0254315 , 0.0240464 , 0.0259979 , 0.0318089 , 0.040652 , 0.0495559 , 0.0541714 , 0.0511493 , 0.0397458 , 0.0222789 , 0.00535058 , 0.000906292 , 0.0245131 , 0.0916792 , 0.2142 , 0.395887 , 0.628144 , 0.888923 , 1.1463 , 1.3644 , 1.51035 , 1.10574 , 1.03796 , 0.849071 , 0.586605 , 0.321137 , 0.120883 , 0.0181954 , 0.00478013 , 0.0531787 , 0.125656 , 0.178239 , 0.186447 , 0.15765 , 0.108266 , 0.0529578 , 0.0122993 , 0.00116054 , 0.0152062 , 0.0421577 , 0.0721469 , 0.0942524 , 0.0993513 , 0.0881285 , 0.0684241 , 0.0466481 , 0.0259783 , 0.00979752 , 0.00172645 , 0.00227009 , 0.00818636 , 0.0153428 , 0.0216945 , 0.0261055 , 0.0269329 , 0.0264301 , 0.0318723 , 0.0466452 , 0.0650844 , 0.0853338 , 0.110377 , 0.131634 , 0.133128 , 0.115591 , 0.0927116 , 0.0699425 , 0.0468049 , 0.0290241 , 0.0231607 , 0.0263421 , 0.0306373 , 0.0312102 , 0.0281203 , 0.0241774 , 0.0231878 , 0.0295707 , 0.0459534 , 0.0697731 , 0.0954567 , 0.118699 , 0.133666 , 0.131906 , 0.113017 , 0.0881323 , 0.0657161 , 0.0462132 , 0.0321561 , 0.0274059 , 0.02799 , 0.0266331 , 0.021788 , 0.0153896 , 0.00835031 , 0.00253475 , 0.00199543 , 0.00956314 , 0.0248621 , 0.0457925 , 0.0692676 , 0.0898361 , 0.0998681 , 0.0938488 , 0.0725717 , 0.0426527 , 0.0144574 , 0.000997607 , 0.0134166 , 0.0531086 , 0.1092 , 0.162683 , 0.191564 , 0.179229 , 0.126332 , 0.0555714 , 0.00541569 , 0.0183551 , 0.124714 , 0.325604 , 0.586353 , 0.845632 , 1.03566 , 1.33722 , 1.24266 , 0.980164 , 0.635131 , 0.324595 , 0.131758 , 0.061955 , 0.0759617 , 0.124891 , 0.152205 , 0.133374 , 0.0957568 , 0.0667776 , 0.0615568 , 0.094264 , 0.146281 , 0.180657 , 0.185639 , 0.161058 , 0.109165 , 0.0511372 , 0.0124104 , 0.00138924 , 0.0142383 , 0.0416717 , 0.0703599 , 0.0900128 , 0.0980686 , 0.0957661 , 0.0859061 , 0.0745076 , 0.0663919 , 0.0597532 , 0.0519789 , 0.0442915 , 0.0356463 , 0.0237703 , 0.0126075 , 0.00634477 , 0.0045452 , 0.00689396 , 0.011691 , 0.0145755 , 0.0144488 , 0.0139757 , 0.016338 , 0.024927 , 0.0410789 , 0.0608163 , 0.0765562 , 0.0821287 , 0.0758242 , 0.0602517 , 0.041166 , 0.0253377 , 0.0165027 , 0.0136847 , 0.0141222 , 0.0145158 , 0.0116707 , 0.00689131 , 0.00459172 , 0.00643416 , 0.0131181 , 0.0244119 , 0.0358053 , 0.0443962 , 0.0520697 , 0.0594106 , 0.0662886 , 0.0751351 , 0.0863504 , 0.0955672 , 0.0978742 , 0.0901116 , 0.0705114 , 0.0418229 , 0.0143177 , 0.00143793 , 0.0127457 , 0.0506006 , 0.106245 , 0.158695 , 0.186909 , 0.181642 , 0.145545 , 0.0964726 , 0.064055 , 0.0664593 , 0.0981522 , 0.136498 , 0.151409 , 0.126343 , 0.0794313 , 0.0616058 , 0.134109 , 0.333049 , 0.638832 , 0.974667 , 1.23637 , 2.72234 , 2.57338 , 2.08249 , 1.4217 , 0.819214 , 0.38078 , 0.120462 , 0.0351464 , 0.0530067 , 0.103208 , 0.170721 , 0.211638 , 0.216798 , 0.207748 , 0.156942 , 0.0807151 , 0.0261321 , 0.00849963 , 0.0419786 , 0.112608 , 0.188983 , 0.25553 , 0.282329 , 0.243979 , 0.163604 , 0.0865055 , 0.0353142 , 0.00807869 , 0.00264792 , 0.020461 , 0.0461863 , 0.0611143 , 0.06892 , 0.0719983 , 0.0619352 , 0.056617 , 0.0655902 , 0.0673406 , 0.0692805 , 0.0792797 , 0.0767654 , 0.0672071 , 0.0633004 , 0.0567049 , 0.0463586 , 0.0376671 , 0.0307955 , 0.0261519 , 0.0235923 , 0.0233707 , 0.0241095 , 0.02373 , 0.0236312 , 0.0266808 , 0.0323184 , 0.0371607 , 0.0432824 , 0.0551827 , 0.064491 , 0.0678286 , 0.0765674 , 0.0807016 , 0.0721794 , 0.0687834 , 0.065612 , 0.0586526 , 0.0647776 , 0.0728308 , 0.0693375 , 0.0617567 , 0.0456759 , 0.0199822 , 0.00287618 , 0.00685794 , 0.0347995 , 0.0903787 , 0.167674 , 0.239872 , 0.274148 , 0.256695 , 0.196631 , 0.114891 , 0.0417574 , 0.00915785 , 0.02836 , 0.0869272 , 0.157999 , 0.210303 , 0.229044 , 0.214543 , 0.171975 , 0.112538 , 0.0542835 , 0.0380693 , 0.125162 , 0.379558 , 0.839082 , 1.44804 , 2.05878 , 2.52194 , 0.0264341 , 0.0269738 , 0.0275215 , 0.028074 , 0.0286283 , 0.0291814 , 0.0297301 , 0.0302716 , 0.0308034 , 0.031323 , 0.0318281 , 0.0323168 , 0.0327868 , 0.0332362 , 0.0336627 , 0.0340633 , 0.0344354 , 0.0347757 , 0.0350811 , 0.0353481 , 0.0355734 , 0.0357543 , 0.0358882 , 0.0359734 , 0.0360088 , 0.0359942 , 0.03593 , 0.0358176 , 0.0356589 , 0.0354563 , 0.035213 , 0.0349318 , 0.0346163 , 0.0342696 , 0.0338949 , 0.0334953 , 0.0330737 , 0.0326325 , 0.0321741 , 0.0317009 , 0.0312151 , 0.0307187 , 0.0302139 , 0.0297031 , 0.0291888 , 0.0286733 , 0.0281594 , 0.0276498 , 0.0271467 , 0.0266527 , 0.0261698 , 0.0256998 , 0.0252439 , 0.0248031 , 0.0243781 , 0.0239691 , 0.0235763 , 0.0231996 , 0.0228392 , 0.0224951 , 0.0221677 , 0.0218575 , 0.0215652 , 0.0212915 , 0.0210375 , 0.0208038 , 0.0205914 , 0.0204004 , 0.0202313 , 0.0200841 , 0.0199583 , 0.0198536 , 0.0197694 , 0.019705 , 0.0196601 , 0.0196343 , 0.0196275 , 0.0196401 , 0.0196726 , 0.0197257 , 0.0198006 , 0.0198983 , 0.0200199 , 0.0201665 , 0.0203389 , 0.0205378 , 0.0207635 , 0.0210161 , 0.0212953 , 0.0216007 , 0.0219315 , 0.0222871 , 0.0226665 , 0.0230689 , 0.0234933 , 0.0239387 , 0.024404 , 0.0248878 , 0.0253887 , 0.0259048 , 0.029514 , 0.0347683 , 0.0405829 , 0.0469499 , 0.0538515 , 0.0612579 , 0.0691263 , 0.0774009 , 0.086014 , 0.0948876 , 0.103937 , 0.11307 , 0.122191 , 0.131199 , 0.139986 , 0.148438 , 0.156433 , 0.163842 , 0.170537 , 0.176391 , 0.181287 , 0.185125 , 0.187817 , 0.189299 , 0.189518 , 0.188439 , 0.186034 , 0.182289 , 0.177206 , 0.170811 , 0.163165 , 0.154376 , 0.144601 , 0.134045 , 0.122951 , 0.111578 , 0.10018 , 0.0889832 , 0.0781711 , 0.0678768 , 0.0581873 , 0.0491541 , 0.0408076 , 0.0331703 , 0.0262661 , 0.0201243 , 0.0147765 , 0.0102518 , 0.00656897 , 0.00373046 , 0.00171942 , 0.000499208 , 1.62661e-05 , 0.000204468 , 0.000990199 , 0.00229682 , 0.0040479 , 0.0061688 , 0.0085868 , 0.0112299 , 0.0140256 , 0.0168996 , 0.0197764 , 0.0225802 , 0.025238 , 0.0276828 , 0.0298562 , 0.0317115 , 0.0332135 , 0.0343385 , 0.0350728 , 0.0354112 , 0.0353557 , 0.0349146 , 0.0341034 , 0.0329455 , 0.0314723 , 0.0297243 , 0.0277492 , 0.0256012 , 0.0233381 , 0.0210189 , 0.0187014 , 0.0164417 , 0.0142931 , 0.0123076 , 0.010536 , 0.00902916 , 0.00783649 , 0.00700527 , 0.00657828 , 0.00659196 , 0.00707505 , 0.00804875 , 0.00952784 , 0.0115226 , 0.0140414 , 0.0170917 , 0.0206819 , 0.0248201 , 0.0693682 , 0.0582337 , 0.0465833 , 0.0350426 , 0.0242363 , 0.0147606 , 0.00718537 , 0.00207786 , 2.64233e-05 , 0.0016311 , 0.00744269 , 0.0178608 , 0.0330317 , 0.0527948 , 0.0767057 , 0.10412 , 0.134292 , 0.166431 , 0.199697 , 0.233157 , 0.265736 , 0.296221 , 0.323316 , 0.34575 , 0.362393 , 0.372364 , 0.375107 , 0.370453 , 0.358648 , 0.340364 , 0.316655 , 0.288859 , 0.258443 , 0.226822 , 0.195217 , 0.164582 , 0.13562 , 0.108866 , 0.084776 , 0.0637786 , 0.0462592 , 0.0324954 , 0.0225868 , 0.0164234 , 0.0137063 , 0.0140053 , 0.0168197 , 0.0216208 , 0.0278702 , 0.0350266 , 0.0425553 , 0.0499465 , 0.0567393 , 0.0625438 , 0.067049 , 0.0700183 , 0.0712808 , 0.0707312 , 0.0683476 , 0.0642229 , 0.0585916 , 0.0518295 , 0.044414 , 0.0368511 , 0.0295961 , 0.022997 , 0.017275 , 0.0125386 , 0.00881329 , 0.00606781 , 0.00422707 , 0.00317761 , 0.00277323 , 0.0028497 , 0.00324908 , 0.00384844 , 0.00458511 , 0.00547196 , 0.00659909 , 0.00812037 , 0.0102282 , 0.0131209 , 0.0169678 , 0.0218828 , 0.027905 , 0.0349896 , 0.0430033 , 0.0517227 , 0.0608357 , 0.0699516 , 0.0786244 , 0.0863885 , 0.0928006 , 0.0974775 , 0.10012 , 0.100526 , 0.0985935 , 0.0943284 , 0.0878503 , 0.0794052 , 0.0814963 , 0.100321 , 0.116537 , 0.128936 , 0.136752 , 0.139621 , 0.13747 , 0.130432 , 0.118911 , 0.103772 , 0.0865327 , 0.0693678 , 0.0548739 , 0.0457124 , 0.044311 , 0.0526722 , 0.0721857 , 0.103337 , 0.145369 , 0.196097 , 0.252069 , 0.30904 , 0.362584 , 0.408617 , 0.443694 , 0.465145 , 0.471133 , 0.46079 , 0.43446 , 0.394022 , 0.343033 , 0.286414 , 0.229568 , 0.177161 , 0.13217 , 0.0956784 , 0.0674217 , 0.046612 , 0.03249 , 0.0243548 , 0.0212356 , 0.0216607 , 0.0238592 , 0.0262634 , 0.0278823 , 0.0282897 , 0.0273659 , 0.0250808 , 0.0214619 , 0.0167016 , 0.0112781 , 0.00599121 , 0.00187932 , 3.58739e-05 , 0.00137083 , 0.00637631 , 0.0149724 , 0.0265021 , 0.0398867 , 0.0538629 , 0.0671734 , 0.0786448 , 0.087203 , 0.0919489 , 0.0923451 , 0.0884271 , 0.080879 , 0.0708825 , 0.0598089 , 0.048905 , 0.0391026 , 0.0309686 , 0.0247442 , 0.0204116 , 0.0177706 , 0.0165275 , 0.0163931 , 0.017159 , 0.0187217 , 0.0210404 , 0.0240613 , 0.0276511 , 0.031563 , 0.0354255 , 0.0387335 , 0.0408627 , 0.0411556 , 0.0391073 , 0.0345914 , 0.0279964 , 0.0201866 , 0.0123234 , 0.00566683 , 0.00143872 , 0.000728181 , 0.00436911 , 0.0127685 , 0.0257478 , 0.0424897 , 0.061635 , 0.0373118 , 0.0316568 , 0.0263483 , 0.0219327 , 0.018869 , 0.0177409 , 0.01924 , 0.0238317 , 0.0313634 , 0.0410327 , 0.0518408 , 0.0632109 , 0.0753736 , 0.0894499 , 0.107396 , 0.131813 , 0.165446 , 0.210371 , 0.267147 , 0.334157 , 0.407213 , 0.479475 , 0.541939 , 0.584689 , 0.598778 , 0.578435 , 0.52319 , 0.439218 , 0.338857 , 0.237589 , 0.149325 , 0.0826577 , 0.0403645 , 0.0216516 , 0.0242023 , 0.0438501 , 0.0731492 , 0.10224 , 0.122844 , 0.131814 , 0.1308 , 0.12307 , 0.111022 , 0.0961305 , 0.0802289 , 0.0660509 , 0.0563169 , 0.0524515 , 0.0541366 , 0.0597382 , 0.0669917 , 0.0735744 , 0.0775364 , 0.0776236 , 0.0734659 , 0.065573 , 0.0551408 , 0.043753 , 0.0330968 , 0.0247056 , 0.0196686 , 0.0183228 , 0.0201112 , 0.0237675 , 0.0277398 , 0.0306062 , 0.0313455 , 0.0295174 , 0.0253768 , 0.0198071 , 0.0139968 , 0.00899092 , 0.00536818 , 0.00318867 , 0.00215607 , 0.00185301 , 0.00194104 , 0.00227849 , 0.00293282 , 0.00408088 , 0.00583207 , 0.0080619 , 0.0103623 , 0.0121673 , 0.0129916 , 0.0126421 , 0.0113206 , 0.00962879 , 0.00846253 , 0.00875571 , 0.0111325 , 0.0156657 , 0.0218971 , 0.0290387 , 0.0361474 , 0.0422134 , 0.0462808 , 0.0476788 , 0.0462672 , 0.0425125 , 0.0350825 , 0.0557676 , 0.0805432 , 0.10629 , 0.128696 , 0.143045 , 0.146202 , 0.138153 , 0.121261 , 0.0980234 , 0.0703592 , 0.0414101 , 0.0170747 , 0.00458503 , 0.00962039 , 0.0356702 , 0.0860818 , 0.164961 , 0.274291 , 0.409918 , 0.560961 , 0.71329 , 0.853236 , 0.968593 , 1.04821 , 1.08303 , 1.06898 , 1.00915 , 0.913305 , 0.794787 , 0.667035 , 0.541243 , 0.425582 , 0.325137 , 0.241918 , 0.17515 , 0.122405 , 0.0813557 , 0.0508577 , 0.0304816 , 0.019265 , 0.0152236 , 0.0158641 , 0.0187087 , 0.0214253 , 0.0220992 , 0.0198563 , 0.0151828 , 0.00946737 , 0.00425396 , 0.000853476 , 0.00023664 , 0.00278809 , 0.00802722 , 0.0148328 , 0.0221588 , 0.0293562 , 0.0355901 , 0.0393563 , 0.0393253 , 0.0357434 , 0.0303498 , 0.0246993 , 0.0191274 , 0.0135247 , 0.00840811 , 0.00469611 , 0.00273982 , 0.00205626 , 0.00194625 , 0.00204742 , 0.00231333 , 0.00273436 , 0.00321028 , 0.00364881 , 0.00410592 , 0.00478949 , 0.00595908 , 0.0078727 , 0.0108118 , 0.0150897 , 0.0209751 , 0.0285933 , 0.037918 , 0.0488178 , 0.0609752 , 0.0736453 , 0.0855373 , 0.095061 , 0.100765 , 0.101582 , 0.0968603 , 0.0865612 , 0.0716416 , 0.0541949 , 0.0370474 , 0.0230884 , 0.0147521 , 0.0137042 , 0.0206162 , 0.0481377 , 0.0363029 , 0.0237798 , 0.0154245 , 0.0170639 , 0.0324596 , 0.0606992 , 0.0975596 , 0.138728 , 0.17882 , 0.207829 , 0.214099 , 0.19326 , 0.150816 , 0.0963257 , 0.0415793 , 0.00480562 , 0.00810613 , 0.0666952 , 0.183501 , 0.352385 , 0.559668 , 0.78166 , 0.986349 , 1.14083 , 1.21809 , 1.20232 , 1.09648 , 0.926506 , 0.730144 , 0.536188 , 0.357542 , 0.20395 , 0.0906367 , 0.0265877 , 0.00393742 , 0.00832832 , 0.0300516 , 0.0608781 , 0.0900761 , 0.11013 , 0.120763 , 0.123071 , 0.116862 , 0.103673 , 0.0877305 , 0.0729365 , 0.0603351 , 0.0498489 , 0.0419976 , 0.0369882 , 0.0337179 , 0.0301419 , 0.0244804 , 0.016443 , 0.00781077 , 0.00168443 , 0.000667895 , 0.00594987 , 0.0181082 , 0.0371499 , 0.0606261 , 0.0835757 , 0.102052 , 0.11509 , 0.122137 , 0.121249 , 0.111697 , 0.0962712 , 0.079279 , 0.0634248 , 0.0494378 , 0.037467 , 0.0277058 , 0.0201169 , 0.0143955 , 0.0102391 , 0.00741148 , 0.00568526 , 0.00501994 , 0.00588241 , 0.00908423 , 0.0149274 , 0.0227737 , 0.0315834 , 0.0401366 , 0.0465995 , 0.0490167 , 0.0469147 , 0.0416774 , 0.0352318 , 0.0294084 , 0.0264079 , 0.0282019 , 0.034801 , 0.0438732 , 0.0523694 , 0.0580779 , 0.0596729 , 0.0563556 , 0.085439 , 0.121824 , 0.153547 , 0.172784 , 0.17094 , 0.14582 , 0.104762 , 0.0583919 , 0.0181984 , 0.000774252 , 0.0199085 , 0.0725391 , 0.143648 , 0.214866 , 0.259515 , 0.25208 , 0.19324 , 0.110878 , 0.0427523 , 0.0295002 , 0.104804 , 0.275007 , 0.512107 , 0.763143 , 0.960309 , 1.03863 , 0.969133 , 0.779736 , 0.534275 , 0.293791 , 0.105896 , 0.00939596 , 0.0139972 , 0.0849818 , 0.175888 , 0.256497 , 0.298521 , 0.279787 , 0.214543 , 0.138926 , 0.0728988 , 0.0239459 , 0.00127466 , 0.00479091 , 0.0211018 , 0.03864 , 0.0523184 , 0.0577713 , 0.0522584 , 0.0387691 , 0.0236755 , 0.0120892 , 0.00705093 , 0.0100014 , 0.0202384 , 0.0353054 , 0.0517233 , 0.0638622 , 0.0654917 , 0.0558968 , 0.0403936 , 0.0239217 , 0.00992973 , 0.00231383 , 0.00182143 , 0.00507466 , 0.00885383 , 0.0107836 , 0.00902027 , 0.00502944 , 0.00259284 , 0.00332486 , 0.00637086 , 0.010588 , 0.0146255 , 0.0166392 , 0.0158457 , 0.0141935 , 0.0156436 , 0.0234984 , 0.0386453 , 0.0595719 , 0.0820766 , 0.0993388 , 0.105452 , 0.0994633 , 0.0845316 , 0.0652721 , 0.0470041 , 0.0341111 , 0.0277564 , 0.025923 , 0.024731 , 0.020569 , 0.0129526 , 0.0049242 , 0.00094574 , 0.00555495 , 0.022105 , 0.0502458 , 0.178048 , 0.139587 , 0.0877161 , 0.038882 , 0.00841458 , 0.00447563 , 0.0282006 , 0.066885 , 0.0956452 , 0.0987662 , 0.0761734 , 0.0370098 , 0.00501645 , 0.00541554 , 0.0417199 , 0.102424 , 0.172019 , 0.239704 , 0.322 , 0.466899 , 0.726839 , 1.11925 , 1.59214 , 2.02836 , 2.29551 , 2.30235 , 2.03317 , 1.55431 , 0.993677 , 0.500677 , 0.189512 , 0.0853775 , 0.125015 , 0.211874 , 0.265698 , 0.248768 , 0.175919 , 0.0908335 , 0.0320292 , 0.0176749 , 0.0406762 , 0.075589 , 0.0961327 , 0.0884458 , 0.0586401 , 0.0253329 , 0.00412664 , 0.00260418 , 0.0203067 , 0.0471342 , 0.070058 , 0.0819549 , 0.0819094 , 0.0726293 , 0.0608428 , 0.0553447 , 0.0617714 , 0.0790297 , 0.098838 , 0.110947 , 0.109459 , 0.0924542 , 0.0639271 , 0.036581 , 0.0223036 , 0.0242241 , 0.0379198 , 0.0521638 , 0.0557127 , 0.0466067 , 0.0299665 , 0.0128844 , 0.00269901 , 0.00322886 , 0.0114315 , 0.0198585 , 0.0225425 , 0.0188179 , 0.0123295 , 0.00712679 , 0.00537706 , 0.00695299 , 0.00935877 , 0.0100087 , 0.00959465 , 0.0120013 , 0.0205943 , 0.0343454 , 0.0485696 , 0.0586176 , 0.0608523 , 0.055209 , 0.0486576 , 0.0503941 , 0.0649664 , 0.0923577 , 0.128583 , 0.164499 , 0.189074 , 0.194723 , 0.0221594 , 0.0616771 , 0.108361 , 0.138695 , 0.141448 , 0.113921 , 0.0642643 , 0.0196963 , 0.00484762 , 0.0254807 , 0.0726531 , 0.117917 , 0.136435 , 0.127852 , 0.0954026 , 0.0476084 , 0.00907818 , 0.00664276 , 0.0757339 , 0.275241 , 0.651726 , 1.19808 , 1.85805 , 2.51364 , 2.97368 , 3.06209 , 2.7501 , 2.17352 , 1.50875 , 0.887021 , 0.419929 , 0.163223 , 0.0597654 , 0.0342693 , 0.0444218 , 0.0529715 , 0.0586899 , 0.0641044 , 0.05691 , 0.0430544 , 0.0230272 , 0.0123386 , 0.0378871 , 0.0990999 , 0.188019 , 0.274979 , 0.315468 , 0.303745 , 0.255164 , 0.179158 , 0.0959506 , 0.0356661 , 0.0126542 , 0.0202088 , 0.0457508 , 0.0757291 , 0.0938211 , 0.0934324 , 0.0818577 , 0.0659947 , 0.0512547 , 0.0443509 , 0.0468311 , 0.052544 , 0.0530519 , 0.0480623 , 0.0419124 , 0.0348665 , 0.0277617 , 0.0236519 , 0.0232862 , 0.0242458 , 0.0243896 , 0.0240028 , 0.0236433 , 0.023108 , 0.0224416 , 0.0213582 , 0.0198357 , 0.0186761 , 0.0174552 , 0.0146787 , 0.0115114 , 0.011755 , 0.0173572 , 0.0274209 , 0.0388548 , 0.048122 , 0.0559713 , 0.0657167 , 0.0807652 , 0.103704 , 0.128653 , 0.145305 , 0.147345 , 0.129082 , 0.0921735 , 0.0511935 , 0.0204024 , 0.00822706 , 0.019026 , 0.0190329 , 0.0190605 , 0.0191086 , 0.019177 , 0.0192657 , 0.0193747 , 0.0195045 , 0.0196557 , 0.0198293 , 0.0200264 , 0.0202482 , 0.0204959 , 0.0207709 , 0.021074 , 0.0214057 , 0.0217664 , 0.0221554 , 0.022572 , 0.0230146 , 0.0234817 , 0.023971 , 0.0244802 , 0.025007 , 0.0255491 , 0.0261043 , 0.0266708 , 0.0272465 , 0.0278299 , 0.0284195 , 0.0290134 , 0.0296102 , 0.0302075 , 0.0308031 , 0.0313944 , 0.031978 , 0.0325503 , 0.0331073 , 0.0336449 , 0.0341584 , 0.0346435 , 0.0350956 , 0.0355109 , 0.0358855 , 0.0362164 , 0.036501 , 0.0367373 , 0.0369241 , 0.0370603 , 0.0371456 , 0.0371801 , 0.0371639 , 0.0370976 , 0.0369817 , 0.036817 , 0.0366044 , 0.036345 , 0.0360402 , 0.0356916 , 0.0353016 , 0.0348726 , 0.0344078 , 0.0339108 , 0.0333857 , 0.0328366 , 0.0322681 , 0.0316846 , 0.0310903 , 0.0304891 , 0.0298846 , 0.0292801 , 0.0286781 , 0.0280809 , 0.0274906 , 0.026909 , 0.026338 , 0.0257792 , 0.0252344 , 0.0247056 , 0.0241945 , 0.0237032 , 0.0232335 , 0.0227872 , 0.0223653 , 0.0219693 , 0.0215996 , 0.0212568 , 0.0209406 , 0.020651 , 0.0203872 , 0.0201487 , 0.0199346 , 0.0197442 , 0.0195771 , 0.0194327 , 0.0193105 , 0.0192104 , 0.0191321 , 0.0190753 , 0.0190401 , 0.00776719 , 0.00772489 , 0.00759798 , 0.00739497 , 0.00713193 , 0.00683228 , 0.00652572 , 0.0062465 , 0.00603066 , 0.00591321 , 0.00592534 , 0.00609251 , 0.00643366 , 0.00696167 , 0.00768483 , 0.00860893 , 0.00973865 , 0.011079 , 0.012635 , 0.0144114 , 0.0164117 , 0.0186374 , 0.0210881 , 0.0237619 , 0.026657 , 0.0297727 , 0.0331108 , 0.0366761 , 0.0404765 , 0.0445215 , 0.048821 , 0.053383 , 0.0582103 , 0.0632987 , 0.0686344 , 0.0741922 , 0.079936 , 0.0858195 , 0.0917886 , 0.0977833 , 0.10374 , 0.109592 , 0.115266 , 0.120681 , 0.125746 , 0.130361 , 0.134416 , 0.137799 , 0.140406 , 0.142144 , 0.142944 , 0.142766 , 0.141603 , 0.139476 , 0.13644 , 0.132575 , 0.127982 , 0.122777 , 0.117089 , 0.111048 , 0.104784 , 0.0984227 , 0.092073 , 0.0858288 , 0.0797635 , 0.0739298 , 0.0683602 , 0.0630703 , 0.0580626 , 0.0533308 , 0.0488636 , 0.0446476 , 0.0406699 , 0.0369185 , 0.0333849 , 0.0300624 , 0.026948 , 0.0240412 , 0.0213441 , 0.01886 , 0.0165931 , 0.0145473 , 0.0127254 , 0.0111292 , 0.00975883 , 0.00861274 , 0.00768736 , 0.00697651 , 0.00647037 , 0.00615465 , 0.00600996 , 0.00601189 , 0.00613179 , 0.0063383 , 0.00659907 , 0.00688275 , 0.00716037 , 0.0074064 , 0.0075997 , 0.00772368 , 0.0341296 , 0.033754 , 0.0326798 , 0.0310855 , 0.0292482 , 0.0275124 , 0.0262535 , 0.0258353 , 0.0265661 , 0.0286597 , 0.0322085 , 0.037176 , 0.0434066 , 0.0506447 , 0.0585511 , 0.0667119 , 0.0746446 , 0.0818161 , 0.0876825 , 0.0917511 , 0.0936459 , 0.093159 , 0.0902648 , 0.0851035 , 0.0779402 , 0.0691236 , 0.0590605 , 0.0482124 , 0.0371066 , 0.0263465 , 0.0166027 , 0.00857992 , 0.00296351 , 0.000364936 , 0.0012842 , 0.00609756 , 0.0150619 , 0.0283135 , 0.0458391 , 0.0674173 , 0.0925549 , 0.120461 , 0.150082 , 0.180203 , 0.209562 , 0.23695 , 0.261269 , 0.281551 , 0.296969 , 0.30685 , 0.310706 , 0.308271 , 0.299535 , 0.284772 , 0.264557 , 0.239763 , 0.21154 , 0.181252 , 0.150366 , 0.120311 , 0.0923267 , 0.0673599 , 0.0460317 , 0.0286837 , 0.0154629 , 0.00639663 , 0.00142012 , 0.000355738 , 0.00287443 , 0.00847842 , 0.0165277 , 0.0263036 , 0.037083 , 0.0481944 , 0.0590428 , 0.0691065 , 0.0779248 , 0.0850918 , 0.0902677 , 0.0932066 , 0.0937885 , 0.0920443 , 0.0881618 , 0.0824697 , 0.075405 , 0.0674718 , 0.0592018 , 0.0511161 , 0.0436894 , 0.0373142 , 0.0322709 , 0.0287058 , 0.026625 , 0.0259059 , 0.0263188 , 0.0275564 , 0.0292661 , 0.0310837 , 0.0326699 , 0.0337463 , 0.0224555 , 0.0213797 , 0.0184994 , 0.014366 , 0.0097564 , 0.00549143 , 0.00224602 , 0.00041238 , 4.80569e-05 , 0.000899852 , 0.00247399 , 0.00413928 , 0.00527054 , 0.00542627 , 0.00451123 , 0.00284953 , 0.00112677 , 0.000230148 , 0.00106099 , 0.00438022 , 0.0106993 , 0.0202009 , 0.0326845 , 0.0475496 , 0.0638314 , 0.0802816 , 0.0954802 , 0.107979 , 0.116497 , 0.120152 , 0.118685 , 0.112524 , 0.102648 , 0.0902797 , 0.0766335 , 0.0629107 , 0.0505752 , 0.0416951 , 0.038995 , 0.0453858 , 0.0631179 , 0.0930197 , 0.134244 , 0.184567 , 0.24092 , 0.299802 , 0.357396 , 0.409553 , 0.451905 , 0.480297 , 0.491491 , 0.48386 , 0.457842 , 0.415953 , 0.362387 , 0.302313 , 0.241074 , 0.183466 , 0.133277 , 0.09307 , 0.0641748 , 0.0467533 , 0.0399189 , 0.0419389 , 0.0505458 , 0.0633062 , 0.0779214 , 0.0923794 , 0.104979 , 0.114334 , 0.119428 , 0.119715 , 0.115181 , 0.106317 , 0.0940045 , 0.0793584 , 0.063587 , 0.0478785 , 0.0333225 , 0.0208431 , 0.0111292 , 0.00454893 , 0.00106662 , 0.000211294 , 0.00115196 , 0.00289088 , 0.00451233 , 0.0053843 , 0.00524491 , 0.00418402 , 0.00257436 , 0.000985282 , 6.7183e-05 , 0.000390391 , 0.00226671 , 0.00562504 , 0.010003 , 0.0146584 , 0.0187492 , 0.0215203 , 0.00142748 , 0.00122339 , 0.00079016 , 0.000403502 , 0.000339913 , 0.000770331 , 0.00171343 , 0.00302838 , 0.00444734 , 0.00568927 , 0.00665169 , 0.00756194 , 0.00895093 , 0.0114565 , 0.015589 , 0.0215362 , 0.0289916 , 0.0370694 , 0.0444749 , 0.049964 , 0.0528265 , 0.0530519 , 0.051108 , 0.0475882 , 0.043017 , 0.0378766 , 0.0327062 , 0.0280891 , 0.0244824 , 0.0220156 , 0.0204691 , 0.0195429 , 0.0192161 , 0.0197778 , 0.0214078 , 0.0238141 , 0.0264191 , 0.0288233 , 0.0310459 , 0.0336772 , 0.0382908 , 0.0478189 , 0.066046 , 0.0958623 , 0.137297 , 0.186979 , 0.23938 , 0.288563 , 0.329118 , 0.356373 , 0.36667 , 0.358121 , 0.331429 , 0.290092 , 0.239762 , 0.18705 , 0.138183 , 0.0978006 , 0.0681681 , 0.0490732 , 0.0384661 , 0.0334781 , 0.0313054 , 0.0297623 , 0.0276271 , 0.0247842 , 0.021945 , 0.0199597 , 0.019205 , 0.0195246 , 0.0206366 , 0.022484 , 0.0252025 , 0.0288575 , 0.0332829 , 0.0381326 , 0.0430015 , 0.0474616 , 0.0510044 , 0.0529991 , 0.0527824 , 0.0499081 , 0.0444517 , 0.0371472 , 0.0291927 , 0.0218093 , 0.0158437 , 0.0116181 , 0.00899784 , 0.00753446 , 0.00663124 , 0.00574635 , 0.00458665 , 0.00318303 , 0.00180406 , 0.000775621 , 0.000308439 , 0.000398965 , 0.00082904 , 0.00126606 , 0.0448071 , 0.0442914 , 0.0427991 , 0.0404107 , 0.0373362 , 0.0338784 , 0.0304847 , 0.0279111 , 0.0271975 , 0.0292966 , 0.0347393 , 0.0435939 , 0.0553019 , 0.0681592 , 0.0792706 , 0.0857556 , 0.0862676 , 0.0811527 , 0.07138 , 0.0579923 , 0.04268 , 0.0282622 , 0.0181333 , 0.0152132 , 0.021236 , 0.0363672 , 0.0589271 , 0.0856397 , 0.112758 , 0.1372 , 0.156414 , 0.167274 , 0.166181 , 0.151628 , 0.126523 , 0.0965419 , 0.0664446 , 0.0396399 , 0.0208944 , 0.0172101 , 0.0348304 , 0.0767481 , 0.144564 , 0.240513 , 0.36462 , 0.509929 , 0.661806 , 0.802426 , 0.916082 , 0.99117 , 1.01952 , 0.996361 , 0.92228 , 0.805459 , 0.661444 , 0.509275 , 0.365905 , 0.242595 , 0.145072 , 0.0756169 , 0.0341432 , 0.0179135 , 0.0217844 , 0.0398517 , 0.0669203 , 0.0982349 , 0.128429 , 0.151855 , 0.164476 , 0.165248 , 0.155494 , 0.137363 , 0.113176 , 0.0858153 , 0.058933 , 0.036367 , 0.0212183 , 0.0151636 , 0.018137 , 0.0283548 , 0.0428259 , 0.0583143 , 0.0721623 , 0.0823918 , 0.0874184 , 0.0862693 , 0.0793307 , 0.0685375 , 0.0564571 , 0.0451803 , 0.036026 , 0.0298169 , 0.0269537 , 0.0272165 , 0.0297663 , 0.0334742 , 0.0373098 , 0.0405771 , 0.0429506 , 0.0443571 , 0.00436173 , 0.0053389 , 0.00742572 , 0.0109649 , 0.016975 , 0.0265583 , 0.0394446 , 0.0534821 , 0.066432 , 0.0776818 , 0.0865057 , 0.0897023 , 0.0841811 , 0.0714871 , 0.0568077 , 0.0442808 , 0.0357176 , 0.0321113 , 0.0335839 , 0.0389431 , 0.0466181 , 0.0549056 , 0.0611502 , 0.0623614 , 0.0572742 , 0.0474791 , 0.0368546 , 0.0297927 , 0.0287184 , 0.0333918 , 0.0439572 , 0.0631719 , 0.0925047 , 0.125593 , 0.149246 , 0.155177 , 0.145763 , 0.124765 , 0.0909859 , 0.0478493 , 0.0143347 , 0.0151688 , 0.065584 , 0.173946 , 0.348333 , 0.586997 , 0.863839 , 1.13442 , 1.35678 , 1.50218 , 1.55301 , 1.50163 , 1.35433 , 1.1326 , 0.868114 , 0.597164 , 0.356302 , 0.174356 , 0.0627177 , 0.0148206 , 0.0155735 , 0.0477328 , 0.0916412 , 0.129055 , 0.151055 , 0.157469 , 0.149033 , 0.126309 , 0.0951512 , 0.0657097 , 0.044656 , 0.0323758 , 0.027261 , 0.0287843 , 0.0361428 , 0.0466871 , 0.0565929 , 0.0623719 , 0.0621179 , 0.0563468 , 0.047727 , 0.0394646 , 0.033823 , 0.0322097 , 0.0359507 , 0.0456015 , 0.059365 , 0.0734298 , 0.0842215 , 0.0896858 , 0.0887471 , 0.0811489 , 0.0683246 , 0.0532984 , 0.038972 , 0.0268472 , 0.0173393 , 0.0105668 , 0.00647101 , 0.00460526 , 0.00538071 , 0.00467103 , 0.00242342 , 0.000380721 , 0.000774574 , 0.0045798 , 0.0107847 , 0.0170281 , 0.0205836 , 0.0197189 , 0.0152625 , 0.0103737 , 0.0085895 , 0.0122665 , 0.0212138 , 0.0317951 , 0.0389126 , 0.0391732 , 0.0319133 , 0.019454 , 0.00730329 , 0.00217385 , 0.00875887 , 0.0279269 , 0.0561187 , 0.085729 , 0.10831 , 0.119171 , 0.117742 , 0.103467 , 0.0757283 , 0.0410878 , 0.0140987 , 0.00566398 , 0.0199969 , 0.060996 , 0.123582 , 0.18386 , 0.220492 , 0.229224 , 0.20271 , 0.134712 , 0.053454 , 0.00922706 , 0.0339812 , 0.143713 , 0.346175 , 0.612444 , 0.868679 , 1.04122 , 1.09513 , 1.02827 , 0.85562 , 0.612169 , 0.356275 , 0.150743 , 0.034244 , 0.0106808 , 0.0573939 , 0.135618 , 0.203942 , 0.236515 , 0.228245 , 0.185631 , 0.123317 , 0.0630177 , 0.0222483 , 0.0066959 , 0.0149647 , 0.0414662 , 0.075156 , 0.103874 , 0.120236 , 0.121523 , 0.107968 , 0.0833893 , 0.05447 , 0.0279486 , 0.00924628 , 0.00222323 , 0.00722088 , 0.0194584 , 0.0318503 , 0.0392317 , 0.0391709 , 0.0319047 , 0.0211841 , 0.0123457 , 0.00874813 , 0.0106113 , 0.0154044 , 0.0193567 , 0.0199959 , 0.0168326 , 0.0107948 , 0.00441903 , 0.000662326 , 0.000370858 , 0.00216746 , 0.00427882 , 0.0362544 , 0.038545 , 0.0446832 , 0.0540995 , 0.0654984 , 0.0774472 , 0.0876882 , 0.0913677 , 0.0857841 , 0.0768757 , 0.0738282 , 0.0799868 , 0.0912178 , 0.100184 , 0.102529 , 0.0957469 , 0.077139 , 0.0520147 , 0.0335448 , 0.0316048 , 0.0500343 , 0.0865969 , 0.131211 , 0.169939 , 0.189025 , 0.178482 , 0.140141 , 0.0878824 , 0.0376083 , 0.00501478 , 0.00493218 , 0.036317 , 0.0795599 , 0.116138 , 0.130126 , 0.107842 , 0.0606532 , 0.0189473 , 0.00118346 , 0.0122972 , 0.043716 , 0.0708768 , 0.0795196 , 0.0843492 , 0.11912 , 0.22322 , 0.422958 , 0.706452 , 1.01444 , 1.25661 , 1.34805 , 1.25407 , 1.01244 , 0.708652 , 0.427595 , 0.22492 , 0.118025 , 0.0840035 , 0.0795837 , 0.0704967 , 0.0438696 , 0.0125505 , 0.00120082 , 0.0193362 , 0.0607908 , 0.10757 , 0.130464 , 0.116577 , 0.0790991 , 0.0356527 , 0.00487046 , 0.00489368 , 0.0371889 , 0.0877513 , 0.140279 , 0.178625 , 0.189194 , 0.16969 , 0.130537 , 0.0867462 , 0.0511466 , 0.0325812 , 0.0345786 , 0.052931 , 0.0771141 , 0.0969308 , 0.105996 , 0.102359 , 0.0912232 , 0.0812274 , 0.076901 , 0.0789004 , 0.0848714 , 0.0888788 , 0.0865024 , 0.0781899 , 0.066605 , 0.0544424 , 0.0443392 , 0.0381126 , 0.00905887 , 0.00951508 , 0.0116803 , 0.0155738 , 0.0215347 , 0.0317108 , 0.0470115 , 0.064134 , 0.0811822 , 0.0997188 , 0.113768 , 0.111395 , 0.0938326 , 0.0698824 , 0.0444219 , 0.027424 , 0.0271986 , 0.0429674 , 0.06605 , 0.080632 , 0.0796758 , 0.0665774 , 0.0440271 , 0.0204179 , 0.0100393 , 0.021879 , 0.0566069 , 0.105032 , 0.143192 , 0.146296 , 0.115552 , 0.0670189 , 0.0270249 , 0.0351927 , 0.0942461 , 0.180614 , 0.285 , 0.353599 , 0.353172 , 0.316028 , 0.23463 , 0.124302 , 0.0363222 , 0.00116746 , 0.0874687 , 0.347225 , 0.794074 , 1.4234 , 2.0925 , 2.55966 , 2.69798 , 2.51862 , 2.07155 , 1.44875 , 0.818102 , 0.343731 , 0.0826233 , 0.000882791 , 0.0395876 , 0.135105 , 0.236229 , 0.320544 , 0.369468 , 0.357482 , 0.287666 , 0.190445 , 0.0948201 , 0.0333581 , 0.0269896 , 0.0650642 , 0.118694 , 0.1513 , 0.141262 , 0.101589 , 0.0574481 , 0.0238705 , 0.00936273 , 0.0188509 , 0.0445154 , 0.0682647 , 0.0802421 , 0.0798957 , 0.0656726 , 0.0433533 , 0.0283858 , 0.0288391 , 0.0446883 , 0.0717649 , 0.0982307 , 0.11348 , 0.11423 , 0.102222 , 0.0832012 , 0.0626529 , 0.0445059 , 0.0313552 , 0.0223713 , 0.0158166 , 0.0117057 , 0.00977584 , 0.0265338 , 0.0259893 , 0.0254577 , 0.0249405 , 0.0244394 , 0.0239557 , 0.0234907 , 0.0230457 , 0.0226219 , 0.0222208 , 0.0218438 , 0.0214921 , 0.021167 , 0.0208697 , 0.0206008 , 0.0203609 , 0.02015 , 0.0199677 , 0.0198133 , 0.0196856 , 0.0195834 , 0.0195051 , 0.0194495 , 0.0194152 , 0.0194011 , 0.0194067 , 0.0194317 , 0.0194762 , 0.0195407 , 0.019626 , 0.0197332 , 0.019863 , 0.0200164 , 0.0201942 , 0.0203961 , 0.0206225 , 0.0208725 , 0.0211454 , 0.0214396 , 0.0217541 , 0.0220873 , 0.0224377 , 0.0228042 , 0.0231862 , 0.0235827 , 0.0239937 , 0.0244191 , 0.024859 , 0.0253133 , 0.0257819 , 0.0262646 , 0.0267604 , 0.0272682 , 0.0277863 , 0.0283125 , 0.0288441 , 0.0293783 , 0.0299119 , 0.0304416 , 0.0309641 , 0.0314764 , 0.0319755 , 0.0324588 , 0.0329238 , 0.0333684 , 0.0337904 , 0.0341877 , 0.0345582 , 0.0348995 , 0.0352091 , 0.0354842 , 0.0357221 , 0.03592 , 0.0360753 , 0.0361854 , 0.0362485 , 0.0362634 , 0.0362296 , 0.0361473 , 0.0360175 , 0.0358416 , 0.035622 , 0.035361 , 0.0350613 , 0.0347257 , 0.0343567 , 0.033957 , 0.0335288 , 0.0330744 , 0.0325962 , 0.0320963 , 0.0315771 , 0.0310414 , 0.0304921 , 0.029932 , 0.0293646 , 0.0287934 , 0.0282216 , 0.0276524 , 0.0270889 , 0.030425 , 0.0255748 , 0.0212542 , 0.0174762 , 0.0142516 , 0.0115865 , 0.00947896 , 0.00791877 , 0.00688621 , 0.00635383 , 0.00628897 , 0.00665659 , 0.00742196 , 0.00855176 , 0.0100133 , 0.0117724 , 0.0137898 , 0.0160185 , 0.0184012 , 0.0208712 , 0.0233543 , 0.025773 , 0.0280507 , 0.0301164 , 0.031908 , 0.0333736 , 0.0344735 , 0.0351792 , 0.0354736 , 0.0353519 , 0.0348199 , 0.0338953 , 0.0326056 , 0.0309868 , 0.02908 , 0.0269288 , 0.0245761 , 0.0220636 , 0.0194324 , 0.0167255 , 0.0139916 , 0.0112882 , 0.00868424 , 0.00625928 , 0.00410241 , 0.00230809 , 0.000971364 , 0.00018272 , 2.34276e-05 , 0.000561813 , 0.00185097 , 0.00392802 , 0.00681477 , 0.0105199 , 0.0150425 , 0.0203757 , 0.0265106 , 0.0334381 , 0.0411481 , 0.0496253 , 0.0588408 , 0.068742 , 0.0792421 , 0.0902135 , 0.101487 , 0.112857 , 0.124096 , 0.134969 , 0.145252 , 0.154748 , 0.163291 , 0.170757 , 0.177051 , 0.182111 , 0.1859 , 0.1884 , 0.18961 , 0.189546 , 0.18824 , 0.185741 , 0.182111 , 0.177424 , 0.171769 , 0.165238 , 0.157937 , 0.14998 , 0.141488 , 0.132593 , 0.123433 , 0.114147 , 0.104867 , 0.0957172 , 0.0868017 , 0.078206 , 0.069994 , 0.0622101 , 0.0548829 , 0.0480301 , 0.0416633 , 0.0357916 , 0.0698984 , 0.0798313 , 0.0881458 , 0.0944715 , 0.0985371 , 0.100194 , 0.0994402 , 0.09643 , 0.0914598 , 0.0849249 , 0.0772598 , 0.068882 , 0.0601612 , 0.0514167 , 0.0429366 , 0.0349939 , 0.0278446 , 0.0217006 , 0.0166909 , 0.0128304 , 0.0100156 , 0.00804941 , 0.00668787 , 0.00569341 , 0.00488026 , 0.00414607 , 0.00348505 , 0.00298479 , 0.00280461 , 0.00314037 , 0.0041818 , 0.006075 , 0.00890233 , 0.0126826 , 0.0173845 , 0.0229382 , 0.0292367 , 0.0361208 , 0.0433554 , 0.0506085 , 0.0574526 , 0.0634008 , 0.0679781 , 0.0708031 , 0.0716515 , 0.0704771 , 0.0674001 , 0.0626701 , 0.056626 , 0.0496627 , 0.0422087 , 0.034716 , 0.0276618 , 0.0215506 , 0.0169139 , 0.0142953 , 0.0142202 , 0.0171528 , 0.0234531 , 0.0333462 , 0.0469182 , 0.0641332 , 0.0848616 , 0.108893 , 0.135914 , 0.165461 , 0.196846 , 0.229121 , 0.261081 , 0.291346 , 0.318482 , 0.341158 , 0.358264 , 0.369004 , 0.372929 , 0.369941 , 0.360252 , 0.344353 , 0.322952 , 0.296928 , 0.267279 , 0.235093 , 0.201516 , 0.167724 , 0.134894 , 0.10414 , 0.0764488 , 0.0525936 , 0.033086 , 0.0181627 , 0.00782011 , 0.00187211 , 8.51688e-06 , 0.00183113 , 0.00686499 , 0.0145569 , 0.0242798 , 0.0353543 , 0.0470854 , 0.058804 , 0.0790535 , 0.0596539 , 0.0413823 , 0.0255364 , 0.0131235 , 0.00481364 , 0.000919694 , 0.00135513 , 0.0055554 , 0.0124367 , 0.0205146 , 0.02823 , 0.0343633 , 0.038306 , 0.0400391 , 0.0398796 , 0.0382127 , 0.0353869 , 0.0317685 , 0.0278135 , 0.0240314 , 0.0208559 , 0.01853 , 0.0170999 , 0.0165169 , 0.0167762 , 0.0180127 , 0.0205126 , 0.0246409 , 0.0307143 , 0.0388407 , 0.0487668 , 0.0597829 , 0.070754 , 0.0803309 , 0.0872766 , 0.0907372 , 0.0903209 , 0.0860075 , 0.0780449 , 0.0669525 , 0.0536112 , 0.0393135 , 0.0256343 , 0.0141003 , 0.00580674 , 0.00118864 , 4.47518e-05 , 0.00173967 , 0.0054438 , 0.010314 , 0.0155916 , 0.0206325 , 0.0248967 , 0.0279269 , 0.029362 , 0.0290228 , 0.0270743 , 0.024192 , 0.0216201 , 0.0210172 , 0.0241193 , 0.0323837 , 0.0468328 , 0.0681714 , 0.0970039 , 0.133839 , 0.178698 , 0.230467 , 0.286434 , 0.342377 , 0.39325 , 0.43419 , 0.461382 , 0.472551 , 0.467021 , 0.445546 , 0.410056 , 0.363448 , 0.309399 , 0.25213 , 0.196079 , 0.145418 , 0.103519 , 0.0725272 , 0.0531866 , 0.0449713 , 0.0464098 , 0.055444 , 0.0697188 , 0.0867908 , 0.104295 , 0.120095 , 0.132418 , 0.139957 , 0.141941 , 0.138165 , 0.128978 , 0.115216 , 0.09809 , 0.0379626 , 0.0422652 , 0.0450332 , 0.0459967 , 0.0450384 , 0.0420298 , 0.036861 , 0.0297932 , 0.0218744 , 0.0148449 , 0.0102789 , 0.00858706 , 0.00888498 , 0.00989715 , 0.0109131 , 0.0118235 , 0.0124655 , 0.0123137 , 0.0109408 , 0.00858115 , 0.00604157 , 0.00407438 , 0.00291077 , 0.00232703 , 0.00202002 , 0.00190441 , 0.00218199 , 0.00326235 , 0.00560344 , 0.00945306 , 0.0145483 , 0.020045 , 0.0248927 , 0.0283648 , 0.0301666 , 0.0300949 , 0.0279173 , 0.0239188 , 0.0195851 , 0.0173456 , 0.019131 , 0.0250675 , 0.0337254 , 0.0435033 , 0.0535126 , 0.0632336 , 0.0717144 , 0.0773921 , 0.0787875 , 0.0754546 , 0.0684532 , 0.0601453 , 0.0535368 , 0.0514213 , 0.0554872 , 0.0656667 , 0.080169 , 0.0963629 , 0.111914 , 0.12508 , 0.133812 , 0.13489 , 0.124888 , 0.103013 , 0.0733691 , 0.0441965 , 0.0245983 , 0.0217369 , 0.0403748 , 0.0834823 , 0.1516 , 0.240793 , 0.341407 , 0.439545 , 0.520837 , 0.574364 , 0.594856 , 0.582771 , 0.542987 , 0.483106 , 0.411948 , 0.338324 , 0.269861 , 0.21184 , 0.166465 , 0.133035 , 0.109003 , 0.0912803 , 0.0771632 , 0.064743 , 0.0530654 , 0.042114 , 0.0325303 , 0.0251197 , 0.0204277 , 0.018597 , 0.0194416 , 0.0225343 , 0.0272274 , 0.032678 , 0.0356316 , 0.0210551 , 0.0141067 , 0.0149024 , 0.0226937 , 0.036034 , 0.05274 , 0.0699658 , 0.0847656 , 0.0949912 , 0.0997979 , 0.0993172 , 0.0940392 , 0.0847166 , 0.0726828 , 0.0597646 , 0.0475846 , 0.0369789 , 0.028093 , 0.0208428 , 0.0151611 , 0.0109452 , 0.00798694 , 0.00602586 , 0.00482775 , 0.00416978 , 0.00376868 , 0.00333057 , 0.00276044 , 0.00221935 , 0.00187449 , 0.00174421 , 0.00186209 , 0.00259498 , 0.00474517 , 0.00869484 , 0.0135789 , 0.0184468 , 0.0236247 , 0.0298065 , 0.0361294 , 0.039868 , 0.0389471 , 0.0342137 , 0.0280533 , 0.0217906 , 0.015356 , 0.00876161 , 0.00316842 , 0.00029593 , 0.00101271 , 0.00476617 , 0.0101057 , 0.015404 , 0.0192818 , 0.0209008 , 0.0202414 , 0.018161 , 0.0161503 , 0.0161138 , 0.0203925 , 0.0315952 , 0.0518633 , 0.0822799 , 0.123457 , 0.176792 , 0.244629 , 0.328853 , 0.429428 , 0.544084 , 0.668308 , 0.794637 , 0.912014 , 1.00694 , 1.06654 , 1.0818 , 1.04927 , 0.971111 , 0.854609 , 0.711659 , 0.557492 , 0.407774 , 0.275112 , 0.167179 , 0.0872068 , 0.0352996 , 0.00921353 , 0.00486955 , 0.0171488 , 0.0405607 , 0.0694954 , 0.0985567 , 0.123135 , 0.139783 , 0.14631 , 0.141983 , 0.127799 , 0.106366 , 0.0813134 , 0.0565627 , 0.0464332 , 0.056496 , 0.0617865 , 0.0606538 , 0.0529961 , 0.0417469 , 0.0320898 , 0.0274624 , 0.0269827 , 0.0285378 , 0.0327482 , 0.0403828 , 0.0476591 , 0.0491855 , 0.0446832 , 0.0380883 , 0.0312679 , 0.0233217 , 0.0148422 , 0.00852463 , 0.00580952 , 0.00559767 , 0.00626059 , 0.00737136 , 0.00950067 , 0.0134064 , 0.0196612 , 0.0284051 , 0.0390768 , 0.0508702 , 0.0638672 , 0.0790461 , 0.0961152 , 0.111437 , 0.119979 , 0.120032 , 0.113526 , 0.101556 , 0.0831142 , 0.0594477 , 0.0359508 , 0.0175765 , 0.00578415 , 0.000499245 , 0.00167043 , 0.00793285 , 0.0160609 , 0.0232102 , 0.0286175 , 0.0330682 , 0.0378442 , 0.0439601 , 0.0515085 , 0.0600311 , 0.0703188 , 0.0846869 , 0.103058 , 0.119395 , 0.125918 , 0.121159 , 0.109011 , 0.0901717 , 0.0626472 , 0.0313639 , 0.00836518 , 0.00402246 , 0.0270673 , 0.0896241 , 0.200356 , 0.354283 , 0.537183 , 0.734979 , 0.930511 , 1.09587 , 1.19786 , 1.21344 , 1.13902 , 0.988517 , 0.786988 , 0.565291 , 0.354893 , 0.182231 , 0.0644528 , 0.00738798 , 0.00511766 , 0.0420463 , 0.0981911 , 0.154741 , 0.196935 , 0.215668 , 0.20914 , 0.182206 , 0.142909 , 0.0997359 , 0.0608063 , 0.0325852 , 0.0178892 , 0.015612 , 0.0223539 , 0.0339673 , 0.0830719 , 0.0475646 , 0.0215019 , 0.00612844 , 0.00139868 , 0.00530594 , 0.0124504 , 0.0182557 , 0.0229051 , 0.0266654 , 0.0280894 , 0.0319769 , 0.0461633 , 0.0669041 , 0.0838861 , 0.0957935 , 0.10356 , 0.100209 , 0.0823208 , 0.0583098 , 0.0378647 , 0.0240559 , 0.0166696 , 0.0147988 , 0.0159263 , 0.0165001 , 0.0144458 , 0.0104663 , 0.00634993 , 0.00325898 , 0.00231809 , 0.00481153 , 0.00962493 , 0.0115855 , 0.00861768 , 0.00460913 , 0.00213091 , 0.00211073 , 0.00903978 , 0.0246509 , 0.0414395 , 0.053911 , 0.0622851 , 0.0638232 , 0.0535452 , 0.0353093 , 0.019067 , 0.0102853 , 0.0081757 , 0.0116726 , 0.0219551 , 0.0380329 , 0.053615 , 0.0603218 , 0.0545216 , 0.0395425 , 0.0207393 , 0.00417547 , 0.00166649 , 0.0256341 , 0.0757847 , 0.143159 , 0.219566 , 0.284331 , 0.30292 , 0.261715 , 0.180357 , 0.0866967 , 0.013983 , 0.00949476 , 0.106649 , 0.296594 , 0.539398 , 0.782093 , 0.963493 , 1.02919 , 0.956658 , 0.766825 , 0.516145 , 0.276062 , 0.105641 , 0.0308959 , 0.0439518 , 0.114101 , 0.198187 , 0.254845 , 0.261666 , 0.220287 , 0.149053 , 0.0743027 , 0.0205044 , 0.00101962 , 0.016487 , 0.0575059 , 0.107439 , 0.149124 , 0.172416 , 0.174471 , 0.156251 , 0.122621 , 0.175971 , 0.193947 , 0.190555 , 0.167033 , 0.130172 , 0.0917866 , 0.0631715 , 0.0490251 , 0.0475374 , 0.0537204 , 0.0598525 , 0.0581735 , 0.0476407 , 0.0333651 , 0.0202483 , 0.0117669 , 0.00938816 , 0.0101012 , 0.00961653 , 0.00723904 , 0.00553881 , 0.00734781 , 0.0128997 , 0.019038 , 0.021798 , 0.0191234 , 0.0115979 , 0.00352941 , 0.00265062 , 0.0130065 , 0.0299525 , 0.0467761 , 0.0572989 , 0.0536015 , 0.0375404 , 0.0241167 , 0.0221936 , 0.0350399 , 0.0637135 , 0.0931723 , 0.10747 , 0.108786 , 0.0988829 , 0.0777974 , 0.0590521 , 0.0549524 , 0.062382 , 0.0732551 , 0.0820077 , 0.0827108 , 0.0704874 , 0.0468432 , 0.0202774 , 0.00267522 , 0.00420321 , 0.0262419 , 0.0583061 , 0.0852256 , 0.0937861 , 0.075823 , 0.0407982 , 0.018016 , 0.0324481 , 0.0907631 , 0.178738 , 0.252776 , 0.266028 , 0.21144 , 0.12627 , 0.0869099 , 0.192938 , 0.504068 , 0.990158 , 1.54603 , 2.03111 , 2.31 , 2.30299 , 2.02573 , 1.5837 , 1.11645 , 0.730635 , 0.470036 , 0.322153 , 0.238813 , 0.171875 , 0.103531 , 0.0421363 , 0.00516985 , 0.00516732 , 0.0358164 , 0.075242 , 0.100153 , 0.0956804 , 0.0652262 , 0.0281934 , 0.00476339 , 0.0078745 , 0.0398158 , 0.0893612 , 0.139032 , 0.0231711 , 0.0078234 , 0.0201599 , 0.0553924 , 0.0952216 , 0.125511 , 0.144453 , 0.145665 , 0.12485 , 0.0996986 , 0.0815816 , 0.0641827 , 0.053183 , 0.0483106 , 0.0373097 , 0.0254342 , 0.0177297 , 0.0117056 , 0.0115666 , 0.0154162 , 0.0169893 , 0.0174884 , 0.0192367 , 0.0211416 , 0.0220658 , 0.0224528 , 0.0228552 , 0.023281 , 0.0237892 , 0.0241213 , 0.0243209 , 0.0247054 , 0.0272043 , 0.0341816 , 0.0421676 , 0.0459732 , 0.0503351 , 0.0525507 , 0.0451265 , 0.0422338 , 0.0518857 , 0.0637621 , 0.0785212 , 0.093496 , 0.0920406 , 0.072619 , 0.047196 , 0.0224911 , 0.0122888 , 0.0372905 , 0.0992295 , 0.176235 , 0.248044 , 0.30346 , 0.321235 , 0.277217 , 0.188277 , 0.100832 , 0.0378774 , 0.0113305 , 0.0227678 , 0.0434748 , 0.0595349 , 0.0670312 , 0.0593508 , 0.0527517 , 0.0451821 , 0.0358307 , 0.0610296 , 0.164113 , 0.41988 , 0.887729 , 1.51516 , 2.18078 , 2.74625 , 3.04969 , 2.96892 , 2.52171 , 1.86777 , 1.20004 , 0.648987 , 0.273954 , 0.0755382 , 0.00606083 , 0.0100219 , 0.0488104 , 0.0955997 , 0.13118 , 0.141075 , 0.119185 , 0.0735682 , 0.0266748 , 0.00483322 , 0.0201663 , 0.0646353 , 0.115573 , 0.145705 , 0.139574 , 0.104061 , 0.0592619 }; 




    CJ_Init(L,size_NF2FF_total);
    CJ_Init(N,size_NF2FF_total);
    float test_Ez;
    float Phi;
    /* The calculation part! */
    for(int Phi_index = 0; Phi_index<numberofexcitationangles;Phi_index++)
    {
        Phi = (float)Phi_index*2*PI/numberofexcitationangles;

        for(int i=0;i<number_of_time_steps;i++)
        {
            cpu_H_field_update(Hy,Hx,Ez,bmx,Psi_hyx,amx,bmx,amx,Psi_hxy,kmx);

            E_field_update(&i,Ez,Hy,Hx,Psi_ezx,aex,aex,bex,bex,Psi_ezy,kex,Cezhy,Cezhy,Ceze,dev_Cezeip,dev_Cezeic,Phi);
            cout<<Ez[getCell(nx/2,ny/2,nx+1)]<<endl;
            for(int freq_index = 0; freq_index < numberoffrequencies; freq_index++){
            freq = startfrequency + deltafreq*freq_index;
            calculate_JandM(&freq, &i,Ez,Hy,Hx,hcjzxp+freq_index*size_x_side,hcjzyp+freq_index*size_y_side, hcjzxn+freq_index*size_x_side, hcjzyn+freq_index*size_y_side, hcmxyp+freq_index*size_y_side, hcmyxp+freq_index*size_x_side,hcmxyn+freq_index*size_y_side,hcmyxn+freq_index*size_x_side);
            }
            //	    cout << "[" << i << "/" << number_of_time_steps << "] " << Ez[getCell(nx/2,ny/2,nx+1)] << std::endl;
        }

        for(int freq_index; freq_index < numberoffrequencies; freq_index++)
        {
            freq = startfrequency + deltafreq*freq_index;
            N2FPostProcess(D + Phi_index*numberofobservationangles*numberoffrequencies+freq_index*numberofobservationangles, freq,N,L, hcjzxp+size_x_side*freq_index , hcjzyp+size_y_side*freq_index, hcjzxn+size_x_side*freq_index , hcjzyn+size_y_side*freq_index, hcmxyp+size_y_side*freq_index, hcmyxp+size_x_side*freq_index, hcmxyn+size_y_side*freq_index, hcmyxn+size_x_side*freq_index);
        CJ_Init(L,size_NF2FF_total);
        CJ_Init(N,size_NF2FF_total);   
        }


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
    float fit=0;
    for(int i = 0; i < numberoffrequencies*numberofexcitationangles*numberofobservationangles; i++)
    {
        fit-= pow(D[i]-measurement[i],2)/(pow(measurement[i],2)*numberofexcitationangles*numberoffrequencies);
    }





    cout<<" fitness = "<<fit<<endl;

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

