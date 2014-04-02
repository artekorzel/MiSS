float weightR(float distance, __constant float cutoffRadius) {
    if(distance > cutoffRadius) {
        return 0.0;
    }
    return 1.0 - distance / cutoffRadius;
}

float weightD(float distance, __constant float cutoffRadius) {
    float weight = weightR(distance, cutoffRadius);
    return weight * weight;
}

float4 countConservativeForce(__global float4* positions, float cutoffRadius, float repulsionParameter, int numberOfDroplets, int dropletId){
    float4 conservativeForce = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for(int i = 0; i < numberOfDroplets, i++){
        if(i != dropletId){
            float distance = distance(poitions[dropletId], positions[i]);
            if(distance < cutoffRadius){
                conservativeForce += (repulsionParameter*(1 - distance/cutoffRadius)) * ((positions[dropletId] - positions[i])/distance);
            }
        }
    }
    return conservaticeForce;
}

__kernel void initForces(__global float4* positions, __global float4* velocities, __global float4* forces, __global float* gaussianRandoms, float time, float deltaTime, float gamma, float sigma, __constant float cutoffRadius, __constant int numberOfDroplets, float repulsionParameter) {
    for(int i = 0; i < numberOfDroplets, i++){
        forces[i] = countConservativeForce(positions, cutoffRadius, repulsionParameter, numberOfDroplets, i)
    }
}