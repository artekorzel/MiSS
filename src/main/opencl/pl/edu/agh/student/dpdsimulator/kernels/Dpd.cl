typedef struct DropletParameter {
    float temperature;
    float density;
    float repulsionParameter;
    float lambda;
    float sigma;
    float gamma;
} DropletParameter;

float weightR(float distanceValue, float cutoffRadius) {
    if(distanceValue > cutoffRadius) {
        return 0.0;
    }
    return 1.0 - distanceValue / cutoffRadius;
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

float rand(int* seed, int step) {
    long const a = 16807L;
    long const m = 2147483647L;
    *seed = (*seed * a * step) % m;
    return (float)(*seed) / (m - 1);
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
    float U1 = (rand(&seed, step) + 1.0) / 2;
    float U2 = (rand(&seed, step) + 1.0) / 2;
    return normalRand(U1, U2);
}

float3 calculateForce(global float3* positions, global float3* velocities, global DropletParameter* params,
    global int* types, float cutoffRadius, int numberOfDroplets, int dropletId, int step) {

    float3 conservativeForce = (float3)(0.0, 0.0, 0.0);
    float3 dissipativeForce = (float3)(0.0, 0.0, 0.0);
    float3 randomForce = (float3)(0.0, 0.0, 0.0);

    float3 dropletPosition = positions[dropletId];
    float3 dropletVelocity = velocities[dropletId];
    
    DropletParameter dropletParameter = params[types[dropletId]];
    
    for(int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
        if(neighbourId != dropletId) {
            float3 neighbourPosition = positions[neighbourId];
            float distanceValue = distance(neighbourPosition, dropletPosition);
            if(distanceValue < cutoffRadius) {
                float weightRValue = weightR(distanceValue, cutoffRadius);
                float weightDValue = weightRValue * weightRValue;
                float3 normalizedPositionVector = normalize(neighbourPosition - dropletPosition);

                conservativeForce += dropletParameter.repulsionParameter * (1.0 - distanceValue / cutoffRadius) * normalizedPositionVector;

                dissipativeForce += dropletParameter.gamma * weightDValue * normalizedPositionVector
                        * dot(normalizedPositionVector, velocities[neighbourId] - dropletVelocity);

                randomForce += dropletParameter.sigma * weightRValue * normalizedPositionVector
                        * gaussianRandom(dropletId, neighbourId, numberOfDroplets, step);
            }
        }
    }

    return conservativeForce + dissipativeForce + randomForce;
}

bool isInsideTube(float3 position, float radius, float height){
    if(fabs(position.y) < height && sqrt(position.x * position.x + position.z * position.z) < radius){
        return true;
    }
    return false;
}

kernel void generateTube(global float3* vector, global int* types, int numberOfDroplets, float range, int type,
    int initialSeed, float radius, float height){
    
    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }    
//    types[dropletId] = type;
    int seed = calculateHash(dropletId, initialSeed);
    float x = rand(&seed, 1) * 2 - 1;
    float y = rand(&seed, 1) * 2 - 1;
    float z = rand(&seed, 1) * 2 - 1;
    vector[dropletId] = ((float3) (x, y, z)) * range;
    if(isInsideTube(vector[dropletId], radius, height)){
        types[dropletId] = type;
    } else {
        types[dropletId] = 0;
    }
    
    
}

kernel void generateTubeFromDroplets(global float3* vector, global int* types, int numberOfDroplets, int type,
    int initialSeed, float radiusIn, float radiusOut, float height){
    
    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }  
    
    types[dropletId] = type;
    int seed = calculateHash(dropletId, initialSeed);
    
    float x = (rand(&seed, 1) * 2 - 1) * (radiusOut);
    float y = rand(&seed, 1) * height - height/2;
    float z;
    float rangeOut = sqrt(radiusOut*radiusOut - x*x);
    float rangeIn = sqrt(radiusIn*radiusIn - x*x);
    if(fabs(x) > radiusIn){        
        z = (rand(&seed, 1) * 2 - 1) * rangeOut;
    } else {
        z = (rand(&seed, 1) * 2 - 1) * rangeOut * (rangeOut - rangeIn);
        if(z < 0){
            z -= rangeIn;
        } else {
            z += rangeIn;
        }
    }
        
    vector[dropletId] = ((float3) (x, y, z));
    
    
}


kernel void generateRandomVector(global float3* vector, float range, int numberOfDroplets, int initialSeed) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }
    
    int seed = calculateHash(dropletId, initialSeed);
    float x = rand(&seed, 1) * 2 - 1;
    float y = rand(&seed, 1) * 2 - 1;
    float z = rand(&seed, 1) * 2 - 1;
    vector[dropletId] = ((float3) (x, y, z)) * range;
}

kernel void calculateForces(global float3* positions, global float3* velocities, global float3* forces, global DropletParameter* params,
        global int* types, float cutoffRadius, int numberOfDroplets, int step) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    forces[dropletId] = calculateForce(positions, velocities, params, types,
             cutoffRadius,  numberOfDroplets, dropletId, step);
}

kernel void calculateNewPositionsAndPredictedVelocities(global float3* positions, global float3* velocities,
        global float3* forces, global float3* newPositions, global float3* predictedVelocities,
        global DropletParameter* params, global int* types, float deltaTime, int numberOfDroplets, float boxSize) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    float3 newPosition = positions[dropletId]
            + deltaTime * velocities[dropletId] + 0.5 * deltaTime * deltaTime * forces[dropletId];
    newPositions[dropletId] = normalizePosition(newPosition, boxSize);
    predictedVelocities[dropletId] = velocities[dropletId] + params[types[dropletId]].lambda * deltaTime * forces[dropletId];
}

kernel void calculateNewVelocities(global float3* newPositions, global float3* velocities,
        global float3* predictedVelocities, global float3* newVelocities, global float3* forces,
        global DropletParameter* params, global int* types, float deltaTime, float cutoffRadius, int numberOfDroplets, int step) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    float3 predictedForce = calculateForce(newPositions, predictedVelocities, params, types,
            cutoffRadius,  numberOfDroplets, dropletId, step);

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
