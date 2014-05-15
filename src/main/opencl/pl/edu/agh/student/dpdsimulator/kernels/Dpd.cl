float weightR(float distanceValue, float cutoffRadius) {
    if(distanceValue > cutoffRadius) {
        return 0.0;
    }
    return 1.0 - distanceValue / cutoffRadius;
}

float weightD(float distanceValue, float cutoffRadius) {
    float weightRValue = weightR(distanceValue, cutoffRadius);
    return weightRValue * weightRValue;
}

float3 normalizePosition(float3 vector, float boxSize) {
    return fmod(fmod(vector + boxSize, 2.0 * boxSize) + 2.0 * boxSize, 2.0 * boxSize) - boxSize;
}

long findIndexOfRandomNumber(int dropletId, int neighbourId) {
    long i1, i2;
    if(dropletId <= neighbourId) {
        i1 = dropletId;
        i2 = neighbourId;
    } else {
        i1 = neighbourId;
        i2 = dropletId;
    }
    return i1 / 2 * (i1 - 1) + i2 - 1;
}

float3 calculateForce(__global float3* positions, __global float3* velocities, __global float* gaussianRandoms,
        float gamma, float sigma, float cutoffRadius, float repulsionParameter, int numberOfDroplets, int dropletId) {

    float3 conservativeForce = (float3)(0.0, 0.0, 0.0);
    float3 dissipativeForce = (float3)(0.0, 0.0, 0.0);
    float3 randomForce = (float3)(0.0, 0.0, 0.0);

    for(int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
        if(neighbourId != dropletId) {
            float distanceValue = distance(positions[neighbourId], positions[dropletId]);
            if(distanceValue < cutoffRadius) {
                float weightRValue = weightR(distanceValue, cutoffRadius);
                float weightDValue = weightD(distanceValue, cutoffRadius);
                float3 normalizedPositionVector = normalize(positions[neighbourId] - positions[dropletId]);

                conservativeForce += repulsionParameter * (1.0 - distanceValue / cutoffRadius) * normalizedPositionVector;

                dissipativeForce += gamma * weightDValue * normalizedPositionVector
                        * dot(normalizedPositionVector, velocities[neighbourId] - velocities[dropletId]);

                randomForce += sigma * weightRValue
                        * gaussianRandoms[findIndexOfRandomNumber(dropletId, neighbourId)] * normalizedPositionVector;
            }
        }
    }

    return conservativeForce + dissipativeForce + randomForce;
}

__kernel void calculateForces(__global float3* positions, __global float3* velocities, __global float3* forces,
        __global float* gaussianRandoms, float gamma, float sigma, float cutoffRadius, float repulsionParameter,
        int numberOfDroplets) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    forces[dropletId] = calculateForce(positions, velocities, gaussianRandoms,
            gamma, sigma, cutoffRadius, repulsionParameter, numberOfDroplets, dropletId);
}

__kernel void calculateNewPositionsAndPredictedVelocities(__global float3* positions, __global float3* velocities,
        __global float3* forces, __global float3* newPositions, __global float3* predictedVelocities,
        float deltaTime, float lambda, int numberOfDroplets, float boxSize) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    float3 newPosition = positions[dropletId]
            + deltaTime * velocities[dropletId] + 0.5 * deltaTime * deltaTime * forces[dropletId];
    newPositions[dropletId] = normalizePosition(newPosition, boxSize);
    predictedVelocities[dropletId] = velocities[dropletId] + lambda * deltaTime * forces[dropletId];
}

__kernel void calculateNewVelocities(__global float3* newPositions, __global float3* velocities,
        __global float3* predictedVelocities, __global float3* newVelocities, __global float3* forces,
        __global float* gaussianRandoms, float deltaTime, float gamma, float sigma, float cutoffRadius,
        float repulsionParameter, int numberOfDroplets) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    float3 predictedForce = calculateForce(newPositions, predictedVelocities, gaussianRandoms, gamma, sigma,
            cutoffRadius, repulsionParameter, numberOfDroplets, dropletId);

    newVelocities[dropletId] = velocities[dropletId] + 0.5 * deltaTime * (forces[dropletId] + predictedForce);
}

__kernel void reductionVector(__global float3* data, __global float3* partialSums, __global float3* output, int dataLength){
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);
    
    partialSums[local_id] = 0.0f;
    
    int i;
    for(i = local_id; i < dataLength; i += group_size){
        partialSums[local_id] += data[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if(local_id == 0){
        output[0] = 0;
        int i;
        for(i = 0; i < group_size; i++){
            output[0] += partialSums[i];
        }
        output[0] /= dataLength;
    }
}

float rand(int* seed) //generuje liczby z zakresu [-1,1];
{
    int const a = 16807;
    int const m = 2147483647;
    *seed = (long)((*seed) * a)%(m+1);
    return ((float)(*seed)/m);
}

float normal_rand(float U1, float U2){   //transformacja Boxa-Mullera zakłada, że U1 U2 rozkład jednostajny na przedziale (0,1]
    float R = -2 * log(U1);
    float fi = 2 * M_PI * U2;

    float Z1 = sqrt(R) * cos(fi);
    return Z1;
    //float Z2 = sqrt(R) * sin(fi);
}

__kernel void random_number_kernel(global int* seed_memory, global float* randoms, int range)
{
    int global_id = get_global_id(1) * get_global_size(0) + get_global_id(0);

    int seed = seed_memory[global_id];
    float U1 = (rand(&seed)+1.0)/2;
    float U2 = (rand(&seed)+1.0)/2;
    randoms[global_id] = normal_rand(U1, U2);

    seed_memory[global_id] = seed; // Save the seed for the next time this kernel gets enqueued.
}


// hashowanie przy pomocy funkcji cantora? http://pl.wikipedia.org/wiki/Funkcja_pary