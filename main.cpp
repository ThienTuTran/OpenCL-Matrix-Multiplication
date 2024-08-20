#include <OpenCL/OpenCL.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#define CHECK_CL_ERROR(err) if (err != CL_SUCCESS) { std::cerr << "OpenCL error: " << err << std::endl; exit(EXIT_FAILURE); }

void matrixMultiplyOpenCL(float* A, float* B, float* C, int N) {

    cl_int err;

        // Step 1 - Select Device
    // Get all platforms
    cl_uint numPlatforms;
    CHECK_CL_ERROR(clGetPlatformIDs(0, NULL, &numPlatforms));
    std::vector<cl_platform_id> platforms(numPlatforms);
    CHECK_CL_ERROR(clGetPlatformIDs(numPlatforms, platforms.data(), NULL));

    // Select a platform (for simplicity, we'll select the first one)
    cl_platform_id platform = platforms[0];

    // Get all devices for the selected platform
    cl_uint numDevices;
    CHECK_CL_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices));
    std::vector<cl_device_id> devices(numDevices);
    CHECK_CL_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), NULL));

    // Select a device (for simplicity, we'll select the first one)
    cl_device_id device = devices[0];

        // Step 2 - Build Program
    // Create Context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL_ERROR(err);

    // Load kernel source code
    const char* kernelSource = R"(
        __kernel void matrixMultiply(__global float* A, __global float* B, __global float* C, int N) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    )";

    // Create Program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    CHECK_CL_ERROR(err);

    // Build Program
    CHECK_CL_ERROR(clBuildProgram(program, 1, &device, NULL, NULL, NULL));

        // Step 3 - Create Buffers & Copy Memory
    // Create Buffers - create buffers on the device
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &err);
    CHECK_CL_ERROR(err);

    // Create Queue - to which we will push commands for the device
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL_ERROR(err);

    // Enqueue Memory Copy - write arrays A and B to the device
    CHECK_CL_ERROR(clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, N * N * sizeof(float), A, 0, NULL, NULL));
    CHECK_CL_ERROR(clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, N * N * sizeof(float), B, 0, NULL, NULL));

        // Step 4 - Execute Kernel & Retrieve Results
    // Create Kernel
    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", &err);
    CHECK_CL_ERROR(err);

    // Set Kernel Arguments
    CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA));
    CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB));
    CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC));
    CHECK_CL_ERROR(clSetKernelArg(kernel, 3, sizeof(int), &N));

    // Set the work-item dimensions
    size_t globalWorkSize[2] = { static_cast<size_t>(N), static_cast<size_t>(N) };

    // Enqueue Kernel
    CHECK_CL_ERROR(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL));

    // Enqueue Memory Copy - read result C from the device to array C
    CHECK_CL_ERROR(clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, N * N * sizeof(float), C, 0, NULL, NULL));

    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void matrixMultiplySequential(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main() {
    std::vector<int> matrixSizes = {200, 400, 600, 800, 1000, 1200, 1400}; // Matrix sizes to test
    std::ofstream outputFile("scalability_results.csv");

    // Write CSV header
    outputFile << "Matrix Size,Speedup\n";

    for (int N : matrixSizes) {
        std::cout << "Testing matrix size: " << N << "x" << N << std::endl;

        std::vector<float> A(N * N, 1.0f);
        std::vector<float> B(N * N, 1.0f);
        std::vector<float> C_seq(N * N, 0.0f);
        std::vector<float> C_par(N * N, 0.0f);

        // Measure sequential execution time
        auto start_seq = std::chrono::high_resolution_clock::now();
        matrixMultiplySequential(A.data(), B.data(), C_seq.data(), N);
        auto end_seq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_seq = end_seq - start_seq;
        std::cout << "Sequential matrix multiplication took " << duration_seq.count() << " seconds." << std::endl;

        // Measure parallel execution time
        auto start_par = std::chrono::high_resolution_clock::now();
        matrixMultiplyOpenCL(A.data(), B.data(), C_par.data(), N);
        auto end_par = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_par = end_par - start_par;
        std::cout << "OpenCL matrix multiplication took " << duration_par.count() << " seconds." << std::endl;

        // Calculate speedup
        double speedup = duration_seq.count() / duration_par.count();
        std::cout << "Speedup: " << speedup << std::endl;

        // Write results to CSV file
        outputFile << N << "," << speedup << "\n";

        std::cout << std::endl;
    }

    outputFile.close();
    return 0;
}