kernel void prefix_sum(uint n,
                       global const float* a,
                       global short* status,
                       global float* aggregate_res,
                       global float* prefix_res,
                       global float* a_prefix) {

#define cluster_size 256
#define work_per_thread 16
    uint i = get_local_id(0) * work_per_thread;
    uint global_i = get_group_id(0) * cluster_size + i;
    uint group_i = get_group_id(0);

    local float a_local[cluster_size];
    float aggregation_local[work_per_thread];
    local float prefix_local[cluster_size];

    for (int j = 0; j < work_per_thread; j++) {
        a_local[i + j] = a[global_i + j];
        prefix_local[i + j] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    aggregation_local[0] = a_local[i];
    for (int j = 1; j < work_per_thread; j++) {
        aggregation_local[j] = a_local[i + j] + aggregation_local[j - 1];
    }


    float value = 0;

    for (int j = i - 1; j != -1; j--) {
        if (prefix_local[j] != 0) {
            value += prefix_local[j];
            break;
        } else {
            value += a_local[j];
        }
    }


    for (int j = 0; j < work_per_thread; j++) {
        prefix_local[i + j] = aggregation_local[j] + value;
    }

    if (i == 0) {
        aggregate_res[group_i] = prefix_local[cluster_size - 1];
        status[group_i] = 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    local float accum_value;
    if (i == 0) {
        accum_value = 0;
        for (int j = group_i - 1; j != -1; j--) {
            while (status[j] == 0);
            if (status[j] == 2) {
                accum_value += prefix_res[j];
                break;
            } else {
                accum_value += aggregate_res[j];
            }
        }
        prefix_res[group_i] = prefix_local[cluster_size - 1] + accum_value;
        status[group_i] = 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = 0; j < work_per_thread; j++) {
        prefix_local[i + j] += accum_value;
        a_prefix[global_i + j] = prefix_local[i + j];
    }
}