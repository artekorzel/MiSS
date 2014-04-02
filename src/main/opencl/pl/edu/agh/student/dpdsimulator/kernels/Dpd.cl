__kernel void weightR(float distance, __constant float cutoffRadius, float* weight) {
    if(distance > cutoffRadius) {
        *weight = 0.0;
        return;
    }
    *weight = 1.0 - distance / cutoffRadius;
}

__kernel void weightD(float distance, __constant float cutoffRadius, float* weight) {
    weightR(distance, cutoffRadius, weight);
    *weight = *weight * *weight;
}

__kernel void simulate() {

}