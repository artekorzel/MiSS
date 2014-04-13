float weightR(float dist, float cutoffRadius) {
    if(dist > cutoffRadius) {
        return 0.0;
    }
    return 1.0 - dist / cutoffRadius;
}

float weightD(float dist, float cutoffRadius) {
    float weight = weightR(dist, cutoffRadius);
    return weight * weight;
}

float4 calculateConservativeForce(__global float4* positions, float cutoffRadius,
        float repulsionParameter, int numberOfDroplets, int dropletId) {

    float4 conservativeForce = (float4)(0.0, 0.0, 0.0, 0.0);
    for(int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
        if(neighbourId != dropletId) {
            float dist = distance(positions[neighbourId], positions[dropletId]);
            if(dist < cutoffRadius) {
                conservativeForce += repulsionParameter * (1.0 - dist / cutoffRadius)
                        * normalize(positions[neighbourId] - positions[dropletId]);
            }
        }
    }
    return conservativeForce;
}

float4 calculateDissipativeForce(__global float4* positions, __global float4* velocities,
        float cutoffRadius, float gamma, int numberOfDroplets, int dropletId) {

    float4 dissipativeForce = (float4)(0.0, 0.0, 0.0, 0.0);
    for(int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
        if(neighbourId != dropletId) {
            float dist = distance(positions[neighbourId], positions[dropletId]);
            if(dist < cutoffRadius) {
                float weight = weightD(dist, cutoffRadius);
                float4 normalizedVector = normalize(positions[neighbourId] - positions[dropletId]);
                dissipativeForce -= gamma * weight * dot(normalizedVector,
                        velocities[neighbourId] - velocities[dropletId]) * normalizedVector;
            }
        }
    }
    return dissipativeForce;
}

float4 calculateRandomForce(__global float4* positions, __global float* gaussianRandoms,
        float cutoffRadius, float sigma, int numberOfDroplets, int dropletId) {

    float4 randomForce = (float4)(0.0, 0.0, 0.0, 0.0);
    for(int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
        if(neighbourId != dropletId) {
            float dist = distance(positions[neighbourId], positions[dropletId]);
            if(dist < cutoffRadius) {
                float weight = weightR(dist, cutoffRadius);
                float4 normalizedVector = normalize(positions[neighbourId] - positions[dropletId]);
                randomForce += sigma * weight * gaussianRandoms[neighbourId
                        * numberOfDroplets + dropletId] * normalizedVector;
            }
        }
    }
    return randomForce;
}

float4 calculateForce(__global float4* positions, __global float4* velocities, __global float* gaussianRandoms, float gamma,
        float sigma, float cutoffRadius, int numberOfDroplets, float repulsionParameter, int dropletId) {

    float4 conservativeForce = calculateConservativeForce(positions, cutoffRadius, repulsionParameter, numberOfDroplets, dropletId);
    float4 dissipativeForce = calculateDissipativeForce(positions, velocities, cutoffRadius, gamma, numberOfDroplets, dropletId);
    float4 randomForce = calculateRandomForce(positions, gaussianRandoms, cutoffRadius, sigma, numberOfDroplets, dropletId);
    return conservativeForce + dissipativeForce + randomForce;
}

__kernel void calculateForces(__global float4* positions, __global float4* velocities, __global float4* forces,
        __global float* gaussianRandoms, float time, float deltaTime, float lambda, float gamma, float sigma,
        float cutoffRadius, int numberOfDroplets, float repulsionParameter) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    forces[dropletId] = calculateForce(positions, velocities, gaussianRandoms, gamma, sigma, cutoffRadius,
            numberOfDroplets, repulsionParameter, dropletId);
}

float4 normalizePosition(float4 vector, float boxSize) {
    return fmod(fmod(vector + boxSize, 2.0 * boxSize) + 2.0 * boxSize, 2.0 * boxSize) - boxSize;
}

__kernel void calculateNewPositionsAndPredictedVelocities(__global float4* positions, __global float4* velocities, __global float4* forces,
        __global float4* newPositions, __global float4* predictedVelocities, float deltaTime, float lambda, int numberOfDroplets, float boxSize) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    float4 newPosition = positions[dropletId] + deltaTime * velocities[dropletId] + 0.5 * deltaTime * deltaTime * forces[dropletId];
    newPositions[dropletId] = normalizePosition(newPosition, boxSize);
    predictedVelocities[dropletId] = velocities[dropletId] + lambda * deltaTime * forces[dropletId];
}

__kernel void calculateNewVelocities(__global float4* newPositions, __global float4* velocities, __global float4* predictedVelocities,
        __global float4* newVelocities, __global float4* forces, __global float* gaussianRandoms, float deltaTime,
        float lambda, float gamma, float sigma, float cutoffRadius, int numberOfDroplets, float repulsionParameter) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    float4 predictedForce = calculateForce(newPositions, predictedVelocities, gaussianRandoms, gamma, sigma,
            cutoffRadius, numberOfDroplets, repulsionParameter, dropletId);

    newVelocities[dropletId] = velocities[dropletId] + 0.5 * deltaTime * (forces[dropletId] + predictedForce);
}