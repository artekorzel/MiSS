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
                float3 normalizedVector = normalize(positions[neighbourId] - positions[dropletId]);

                conservativeForce += repulsionParameter * (1.0 - distanceValue / cutoffRadius)
                        * normalize(positions[neighbourId] - positions[dropletId]);

                dissipativeForce += gamma * weightDValue * normalizedVector
                        * dot(normalizedVector, velocities[neighbourId] - velocities[dropletId]);

                randomForce += sigma * weightRValue * gaussianRandoms[neighbourId
                        * numberOfDroplets + dropletId] * normalizedVector;
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