#ifdef __APPLE__
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>

#endif

#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <cstdio>
#include <string>

cl_device_id get_device_id(int num) {
    std::vector<cl_device_id> devices;
    cl_uint platform_num;

    clGetPlatformIDs(0, nullptr, &platform_num);
    if (!platform_num) { throw std::runtime_error("Number of platforms: 0\n"); }

    auto *platform_ids = new cl_platform_id[platform_num];
    clGetPlatformIDs(platform_num, platform_ids, nullptr);

    for (cl_uint i = 0; i < platform_num; ++i) {
        cl_uint device_num;
        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_num);

        if (!device_num) { continue; }

        auto *device_ids = new cl_device_id[device_num];
        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, device_num, device_ids, nullptr);

        devices.insert(devices.end(), device_ids, device_ids + device_num);
        delete[] device_ids;
    }
    delete[] platform_ids;

    std::sort(devices.begin(), devices.end(), [](auto d1, auto d2) {
        cl_device_type d1_type, d2_type;
        clGetDeviceInfo(d1, CL_DEVICE_TYPE, sizeof(cl_device_type), &d1_type, nullptr);
        clGetDeviceInfo(d2, CL_DEVICE_TYPE, sizeof(cl_device_type), &d2_type, nullptr);
        cl_bool d1_mem, d2_mem;
        clGetDeviceInfo(d1, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &d1_mem, nullptr);
        clGetDeviceInfo(d2, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &d2_mem, nullptr);
        int points1 = (d1_type == CL_DEVICE_TYPE_GPU ? 0 : 1) + (d1_mem == CL_FALSE ? 0 : 1);
        int points2 = (d2_type == CL_DEVICE_TYPE_GPU ? 0 : 1) + (d2_mem == CL_FALSE ? 0 : 1);
        // discrete GPU = 0 points
        // integrate GPU = 1 points
        // CPU = 2 points
        return points1 < points2;
    });

    return devices[num];
}

std::string get_device_name(cl_device_id device) {
    size_t size;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size);
    char *name = new char[size];
    clGetDeviceInfo(device, CL_DEVICE_NAME, size, name, nullptr);
    std::string res(name);
    delete[] name;
    return res;
}

std::string file_to_string(const std::string &filename) {
    std::ifstream t(filename);
    std::string str((std::istreambuf_iterator<char>(t)),
                    std::istreambuf_iterator<char>());
    return str;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Invalid number of arguments";
        return 1;
    }
    std::ifstream in(argv[2]);
    if (!in) {
        std::cerr << "Can't open file: " << argv[2] << '\n';
        return 1;
    }
    std::ofstream out(argv[3]);
    if (!out) {
        std::cerr << "Can't open file " << argv[3] << '\n';
        in.close();
        return 1;
    }

    int device_num;

    try {
        device_num = std::stoi(argv[1]);
    } catch (std::invalid_argument &e) {
        std::cerr << "Invalid 1 argument: " << argv[1] << '\n';
        in.close();
        out.close();
        return 1;
    } catch (std::out_of_range &e) {
        std::cerr << "1 argument out of range: " << argv[1] << '\n';
        in.close();
        out.close();
        return 1;
    }

    size_t n;
    in >> n;
    cl_uint cluster_size = 256;
    cl_uint work_per_thread = 16;
    cl_uint n_round = (n + cluster_size - 1) / cluster_size * cluster_size;
    cl_uint cluster_number = (n + cluster_size - 1) / cluster_size;

    auto *a = new cl_float[n_round];

    for (int i = 0; i < n; i++) {
        in >> a[i];
    }
    for (int i = n; i < n_round; i++) {
        a[i] = 0;
    }

    auto* status = new cl_short[cluster_number];
    for (int i = 0; i < cluster_number; i++) {
        status[i] = 0;
    }

    auto host_time_start = std::chrono::high_resolution_clock::now();

    cl_device_id device = get_device_id(device_num);
    std::cout << "Device: " << get_device_name(device) << '\n';

    int err;

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context) { throw std::runtime_error("Error: Failed to create a context!\n"); }

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!queue) { throw std::runtime_error("Error: Failed to create a command queue!\n"); }

    std::string kernel_source = file_to_string("kernel.cl");
    const char *src = kernel_source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &src, nullptr, &err);
    if (!program) { throw std::runtime_error("Error: Failed to create compute program!\n"); }


    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        throw std::runtime_error("Error: Failed to build program executable!\n" + std::string(buffer));
    }

    cl_kernel kernel = clCreateKernel(program, "prefix_sum", &err);
    if (!kernel || err != CL_SUCCESS) { throw std::runtime_error("Error: Failed to create kernel!\n"); }

    cl_mem cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * n_round, nullptr, nullptr);
    cl_mem cl_status = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_short) * cluster_number, nullptr, nullptr);
    cl_mem cl_aggregation_res = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * cluster_number, nullptr, nullptr);
    cl_mem cl_prefix_res = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * cluster_number, nullptr, nullptr);
    cl_mem cl_a_prefix = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * n_round, nullptr, nullptr);
    if (!cl_a || !cl_status || !cl_aggregation_res || !cl_prefix_res || !cl_a_prefix) { throw std::runtime_error("Error: Failed to allocate device memory!\n"); }

    err = clEnqueueWriteBuffer(queue, cl_a, CL_FALSE, 0, sizeof(cl_float) * n_round, a, 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(queue, cl_status, CL_FALSE, 0, sizeof(cl_short) * cluster_number, status, 0, nullptr, nullptr);

    if (err != CL_SUCCESS) { throw std::runtime_error("Error: Failed to write to source array!\n"); }

    err = clSetKernelArg(kernel, 0, sizeof(cl_uint), &n);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_a);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_status);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_aggregation_res);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_prefix_res);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_a_prefix);
    if (err != CL_SUCCESS) { throw std::runtime_error("Error: Failed to set kernel arguments!"); }

    size_t global[1] = {n_round / work_per_thread};
    size_t local[1] = {cluster_size / work_per_thread};
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global, local, 0, nullptr, &event);
    if (err) { std::cout << err; throw std::runtime_error("Error: Failed to execute kernel!\n"); }
    try {
        err = clFinish(queue);
    } catch (...) {
        out << "aboba";
    }

    auto* a_prefix = new cl_float[n_round];

    err = clEnqueueReadBuffer(queue, cl_a_prefix, CL_TRUE, 0, sizeof(cl_float) * n_round, a_prefix, 0, nullptr, nullptr);

    if (err != CL_SUCCESS) { throw std::runtime_error("Error: Failed to read output array!"); }

    clReleaseMemObject(cl_a);
    clReleaseMemObject(cl_a_prefix);
    clReleaseMemObject(cl_aggregation_res);
    clReleaseMemObject(cl_prefix_res);
    clReleaseMemObject(cl_status);
    cl_a = nullptr;
    cl_a_prefix = nullptr;
    cl_aggregation_res = nullptr;
    cl_prefix_res = nullptr;
    cl_status = nullptr;

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);

    auto host_time_end = std::chrono::high_resolution_clock::now();

    float host_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(host_time_end - host_time_start).count());

    cl_ulong device_time_start;
    cl_ulong device_time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(device_time_start), &device_time_start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(device_time_end), &device_time_end, nullptr);

    auto device_time = static_cast<double>(device_time_end-device_time_start) / 1e6;

    printf("Time:%g\t%g\n", device_time, host_time);
    printf("LOCAL_WORD_SIZE[%i, %i]\n", local[0], work_per_thread);

    delete [] a;

    for (int i = 0; i < n; i++) {
        out << a_prefix[i] << ' ';
    }

    delete [] a_prefix;

    return 0;
}
