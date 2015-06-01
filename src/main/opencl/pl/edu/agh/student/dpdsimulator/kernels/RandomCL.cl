#pragma OPENCL EXTENSION cl_khr_fp64 : enable

ulong MWCAdd(ulong a, ulong b, ulong M)
{
    ulong v=a+b;
    if( (v>=M) || (v<a) )
            v=v-M;
    return v;
}

ulong MWCMul(ulong a, ulong b, ulong M)
{	
    ulong r=0;
    while(a!=0){
            if(a&1)
                    r=MWCAdd(r,b,M);
            b=MWCAdd(b,b,M);
            a=a>>1;
    }
    return r;
}

ulong MWCPow(ulong a, ulong e, ulong M)
{
    ulong sqr=a, acc=1;
    while(e!=0){
            if(e&1)
                    acc=MWCMul(acc,sqr,M);
            sqr=MWCMul(sqr,sqr,M);
            e=e>>1;
    }
    return acc;
}

void MWCStep(uint2 *s)
{
    ulong A = 4294883355U;
    uint X=(*s).x, C=(*s).y;

    uint Xn=A*X+C;
    uint carry=(uint)(Xn<C);
    uint Cn=((A * X) >> 32) + carry;  

    (*s).x=Xn;
    (*s).y=Cn;
}

void MWCSkip(uint2 *s, ulong distance)
{
    ulong A = 4294883355U;
    ulong M = 9223372036854775807UL;
    
    ulong m=MWCPow(A, distance, M);
    ulong x=(*s).x*A+(*s).y;
    x=MWCMul(x, m, M);
    *s = (uint2)((uint)(x/A), (uint)(x%A));
}

void MWCSeed(uint2 *s, ulong baseOffset, ulong perStreamOffset)
{
    ulong A = 4294883355U;
    ulong M = 9223372036854775807UL;
    ulong BASEID = 4077358422479273989UL;

    ulong dist=baseOffset + get_global_id(0)*perStreamOffset;
    ulong m=MWCPow(A, dist, M);

    ulong x=MWCMul(BASEID, m, M);
    *s = (uint2)((uint)(x/A), (uint)(x%A));
}

uint MWCNext(uint2 *s)
{
    uint res=(*s).x ^ (*s).y;
    MWCStep(s);
    return res;
}

kernel void test(global double* a) {    
    int dropletId = get_global_id(0);
    if (dropletId >= 16384) {
        return;
    }
    
    uint2 state;
    MWCSeed(&state, 0, 16384);
    int i;
    for(i = 0; i < 16384; ++i) {
        a[dropletId*16384+i] = MWCNext(&state) / 2147483647.0 - 1.0;
    }
}

kernel void generateRandomNumbers(global float* vector, int numberOfRandoms, int step) { 
    int globalId = get_global_id(0);
    int globalSize = get_global_size(0);
    if (globalId >= globalSize) {
        return;
    }
    
    int randomsPerCore = ceil(numberOfRandoms / (double)globalSize);
    
    uint2 state;
    MWCSeed(&state, step * numberOfRandoms, randomsPerCore);
    int i;
    for(i = 0; i < randomsPerCore; ++i) {
        int index = globalId * randomsPerCore + i;
        if(index < numberOfRandoms) {
            double randNum = MWCNext(&state) / 2147483647.0 - 1.0;   
            vector[index] = (float) randNum;
        }
    }
}