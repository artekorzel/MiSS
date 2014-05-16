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

int calculateHash(int d1, int d2) {    
    int i1, i2;
    if(d1 <= d2) {
        i1 = d1;
        i2 = d2;
    } else {
        i1 = d2;
        i2 = d1;
    }
    return (i1 + i2) * (i1 + i2 + 1) / 2 + i2;
}

float rand(int seed, int step) {
    long const a = 16807L;
    long const m = 2147483647L;
    seed = ((seed) * a * step) % m;
    return (float)(seed) / (m - 1);
}

float normalRand(float U1, float U2) {
     float R = -2 * log(U1);
     float fi = 2 * M_PI * U2;
     float Z1 = sqrt(R) * cos(fi);
     return Z1;
     //float Z2 = sqrt(R) * sin(fi);
}

float gaussianRandom(int dropletId, int neighbourId, int numberOfDroplets, int step) {
    int seed = calculateHash(dropletId, neighbourId);
    float U1 = (rand(seed, step) + 1.0) / 2;
    float U2 = (rand(seed, step) + 1.0) / 2;
    return normalRand(U1, U2);
}

float3 calculateForce(global float3* positions, global float3* velocities, float gamma, float sigma, 
        float cutoffRadius, float repulsionParameter, int numberOfDroplets, int dropletId, int step) {

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

                randomForce += sigma * weightRValue * normalizedPositionVector
                        * gaussianRandom(dropletId, neighbourId, numberOfDroplets, step);
            }
        }
    }

    return conservativeForce + dissipativeForce + randomForce;
}

kernel void calculateForces(global float3* positions, global float3* velocities, global float3* forces,
        float gamma, float sigma, float cutoffRadius, float repulsionParameter, int numberOfDroplets, int step) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    forces[dropletId] = calculateForce(positions, velocities, gamma, 
            sigma, cutoffRadius, repulsionParameter, numberOfDroplets, dropletId, step);
}

kernel void calculateNewPositionsAndPredictedVelocities(global float3* positions, global float3* velocities,
        global float3* forces, global float3* newPositions, global float3* predictedVelocities,
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

kernel void calculateNewVelocities(global float3* newPositions, global float3* velocities,
        global float3* predictedVelocities, global float3* newVelocities, global float3* forces,
        float deltaTime, float gamma, float sigma, float cutoffRadius, float repulsionParameter, 
        int numberOfDroplets, int step) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    float3 predictedForce = calculateForce(newPositions, predictedVelocities, gamma, sigma,
            cutoffRadius, repulsionParameter, numberOfDroplets, dropletId, step);

    newVelocities[dropletId] = velocities[dropletId] + 0.5 * deltaTime * (forces[dropletId] + predictedForce);
}

kernel void reductionVector(global float3* data, global float3* partialSums, global float3* output, int dataLength) {

    int global_id = get_global_id(0);
    int group_size = get_global_size(0);

    if(global_id < dataLength){
        partialSums[global_id] = data[global_id];
    } else {
        partialSums[global_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    int offset;
    for(offset = get_global_size(0)/2; offset > 0; offset >>= 1){
        if(global_id < offset){
            partialSums[global_id] += partialSums[global_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
    if(global_id == 0){
        output[0] = partialSums[0]/dataLength;
    }
}
