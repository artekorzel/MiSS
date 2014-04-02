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
                conservativeForce += repulsionParameter * (1.0 - dist / cutoffRadius) * normalize(positions[neighbourId] - positions[dropletId]);
            }
        }
    }
    return conservativeForce;
}

float4 calculateDissipativeForce(__global float4* positions, __global float4* velocities, float cutoffRadius, float gamma, int numberOfDroplets, int dropletId) {
    float4 dissipativeForce = (float4)(0.0, 0.0, 0.0, 0.0);
    for(int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
        if(neighbourId != dropletId) {
            float dist = distance(positions[neighbourId], positions[dropletId]);
            if(dist < cutoffRadius) {
                float weight = weightD(dist, cutoffRadius);
                float4 normalizedVector = normalize(positions[neighbourId] - positions[dropletId]);
                dissipativeForce -= gamma * weight * dot(normalizedVector, velocities[neighbourId] - velocities[dropletId]) * normalizedVector;
            }
        }
    }
    return dissipativeForce;
}

float4 calculateRandomForce(__global float4* positions, __global float* gaussianRandoms, float cutoffRadius, float sigma, int numberOfDroplets, int dropletId) {
    float4 randomForce = (float4)(0.0, 0.0, 0.0, 0.0);
    for(int neighbourId = 0; neighbourId < numberOfDroplets; neighbourId++) {
        if(neighbourId != dropletId) {
            float dist = distance(positions[neighbourId], positions[dropletId]);
            if(dist < cutoffRadius) {
                float weight = weightR(dist, cutoffRadius);
                float4 normalizedVector = normalize(positions[neighbourId] - positions[dropletId]);
                randomForce += sigma * weight * gaussianRandoms[neighbourId * numberOfDroplets + dropletId] * normalizedVector;
            }
        }
    }
    return randomForce;
}

__kernel void initForces(__global float4* positions, __global float4* velocities, __global float4* forces, __global float* gaussianRandoms,
        float time, float deltaTime, float gamma, float sigma, float cutoffRadius, int numberOfDroplets, float repulsionParameter) {

    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    float4 conservativeForce = calculateConservativeForce(positions, cutoffRadius, repulsionParameter, numberOfDroplets, dropletId);
    float4 dissipativeForce = calculateDissipativeForce(positions, velocities, cutoffRadius, gamma, numberOfDroplets, dropletId);
    float4 randomForce = calculateRandomForce(positions, gaussianRandoms, cutoffRadius, sigma, numberOfDroplets, dropletId);
    forces[dropletId] = conservativeForce + dissipativeForce + randomForce;
}