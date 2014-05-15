typedef struct Data {
	int number;
} Data;

void manipulate(__global const Data* data, __global int* out, __private int id) {
    out[id] = data[id].number;
}



__kernel void return_data(__global const Data* data, __global int* out, int data_length)
{
    int global_id = get_global_id(0);
    if (global_id >= data_length)
        return;

    manipulate(data, out, global_id);
}

float rand(int* seed)
{
    int const a = 16807;
    int const m = 2147483647;
    *seed = (long)((*seed) * a)%(m+1);
    return ((float)(*seed)/m);
}

float normal_rand(float U1, float U2){
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
