typedef struct Data {
	int number;
} Data;

void manipulate(global const Data* data, global int* out, private int id) {
    out[id] = data[id].number;
}



kernel void return_data(global const Data* data, global int* out, int data_length)
{
    int global_id = get_global_id(0);
    if (global_id >= data_length)
        return;

    manipulate(data, out, global_id);
}

float rand(int* seed, int step) {//generuje liczby z zakresu [-1,1];
    long const a = 16807L;
    long const m = 2147483647L;
    *seed = ((*seed) * a * step) % m;
    return (float)(*seed) / (m - 1);
}

float normal_rand(float U1, float U2) {
    float R = -2 * log(U1);
    float fi = 2 * M_PI * U2;
    float Z1 = sqrt(R) * cos(fi); 
    return Z1;
     //float Z2 = sqrt(R) * sin(fi);
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

kernel void random_number_kernel(global float* randoms, int numberOfDroplets, int neighbourId, int step) {
    int dropletId = get_global_id(0);
    if (dropletId >= numberOfDroplets) {
        return;
    }

    int seed = calculateHash(dropletId, neighbourId);
    float U1 = (rand(&seed, step) + 1.0) / 2;
    float U2 = (rand(&seed, step) + 1.0) / 2;
    randoms[dropletId] = normal_rand(U1, U2);
}
