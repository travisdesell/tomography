#include <cmath>
#include "FDTD_common.hxx"

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

float calc_incident_power(float freq)
{
    return (0.5/eta0)*pow(tau*sqrt(PI)*exp(-tau*tau*2*PI*freq*2*PI*freq/4),2);// just gonna assume gaussian pulse.  This is the fourier transform of the gaussian pulse.
}

float fitness(float* D,int max_index, float* measurement)
{
    float fit = 0;
    for(int i =0;i<max_index;i++)
    {
        fit -= pow((measurement[i]-D[i]),2)/(numberofexcitationangles*pow(measurement[i],2));
    }

    return fit;
}


