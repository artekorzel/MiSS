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

