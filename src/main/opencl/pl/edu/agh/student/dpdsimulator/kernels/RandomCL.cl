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

typedef struct DropletParameters {
    float mass;
    float lambda;
} DropletParameters;

typedef struct PairParameters {
    float cutoffRadius;
    float pi;
    float sigma;
    float gamma;
} PairParameters;

typedef struct Parameters {
    DropletParameters droplets[3];
    PairParameters pairs[3][3];
} Parameters;

kernel void test(global Parameters* params, global float* out) {
    int i, j;
    for(i = 0; i < 3; ++i) {
        for(j = 0; j < 3; ++j) {
            out[i * 3 + j] = params[0].pairs[i][j].pi;
        }
    }
}

kernel void normalizePosition(global float3* vector, float boxSize, float boxWidth) {
    float3 changeVector = (float3)(boxSize, boxWidth, boxSize);
    vector[0] = fmod(fmod(vector[0] + changeVector, 2.0f * changeVector) + 2.0f * changeVector, 2.0f * changeVector) - changeVector;
}

kernel void dist(global float3* vector, global float3* vector2, global float* res) {
    res[0] = distance(vector[0], vector2[0]);
}

kernel void calculateCellId(global int* cellNo, global float3* pos, float cellRadius, float boxSize, float boxWidth) {
    float3 position = pos[0];
    cellNo[0] = ((int)((position.x + boxSize) / cellRadius)) + 
            ((int)(2 * boxSize / cellRadius)) * (((int)((position.y + boxWidth) / cellRadius)) + 
                    ((int)(2 * boxWidth / cellRadius)) * ((int)((position.z + boxSize) / cellRadius)));
}

kernel void calculateCellCoordinates(global int3* res, int dropletCellId, float cellRadius, float boxSize, 
        float boxWidth, int cellsNoXZ, int cellsNoY) {
    int dropletCellZ = dropletCellId / (cellsNoXZ * cellsNoY);
    int dropletCellX = dropletCellId - dropletCellZ * cellsNoXZ * cellsNoY;
    int dropletCellY = dropletCellX / cellsNoXZ;
    dropletCellX = dropletCellId % cellsNoXZ;
    res[0] = (int3)(dropletCellX, dropletCellY, dropletCellZ);
}

kernel void test2(global float* a) {
    a[0] = (-5) % 5;
}


typedef struct TestStruct {
    float mass;
} TestStruct;

kernel void test3(constant TestStruct *aStruct, global float* out) {
    out[0] = aStruct->mass;
}

kernel void reduction(global float* data, local float* partialSums, global float* output, int dataLength) {

    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);
    
    float localResult = 0;
    while (global_id < dataLength) {
        localResult += data[global_id];
        global_id += global_size;
    }
    
    partialSums[local_id] = localResult;
    barrier(CLK_LOCAL_MEM_FENCE);
    int offset;
    for(offset = local_size/2; offset > 0; offset >>= 1){
        if(local_id < offset){
            partialSums[local_id] += partialSums[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(local_id == 0){
        output[group_id] = partialSums[0];
    }
}