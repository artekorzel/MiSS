// created on 2014-10-12

/**
 * @author Filip
 */


float rand(int* seed, int step) {
    long const a = 16807L;
    long const m = 2147483647L;
    *seed = (*seed * a * step) % m;
    return (float)(*seed) / (m - 1);
}

uint MWC64X(uint2 *state)
{
    enum { A=4294883355U};
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);               // Pack the state back up
    return res;                       // Return the next result
}
/*
Funkcja randomizujaca z rozkladem normalnym na przedziale <-1; 1>
*/
float normalRand(float U1, float U2) {
     float R = -2 * log(U1);
     float fi = 2 * M_PI * U2;
     float Z1 = sqrt(R) * cos(fi);
     return Z1;
     //float Z2 = sqrt(R) * sin(fi);
}

kernel void random(global float* a, const int size) {
    int id = get_global_id(0);
    if (id >= size) {
        return;
    }       
    int seed = id;
    a[id] = MWC64X(&seed);
}