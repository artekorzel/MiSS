typedef struct Data {
	int number;
} Data;

__kernel void return_data(__global const Data* data, __global int* out, int data_length)
{
    int global_id = get_global_id(0);
    if (global_id >= data_length)
        return;

    out[global_id] = data[global_id].number;
}