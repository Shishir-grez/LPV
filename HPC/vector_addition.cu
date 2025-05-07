#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <omp.h>
#include <iomanip>

using namespace std;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel for vector addition
__global__ void vectorAddKernel(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Sequential vector addition for verification
void sequentialVectorAdd(const vector<float>& A, const vector<float>& B, vector<float>& C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Function to generate random vector
vector<float> generateRandomVector(int N) {
    vector<float> vec(N);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 10.0f);
    for (int i = 0; i < N; ++i) {
        vec[i] = dis(gen);
    }
    return vec;
}

int main() {
    int N;
    char choice;

    // Input vector length - Modified for automated testing
    N = 50000000;  // Using a large value for better timing comparison
    choice = 'y'; // Always use random vectors in Kaggle

    vector<float> A, B, C(N), C_seq(N);
    if (choice == 'y' || choice == 'Y') {
        A = generateRandomVector(N);
        B = generateRandomVector(N);
    } else {
        A.resize(N);
        B.resize(N);
        cout << "Enter " << N << " elements for vector A:\n";
        for (int i = 0; i < N; ++i) cin >> A[i];
        cout << "Enter " << N << " elements for vector B:\n";
        for (int i = 0; i < N; ++i) cin >> B[i];
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Set up kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch kernel and measure time
    CUDA_CHECK(cudaEventRecord(start));
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cuda_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&cuda_time_ms, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Sequential vector addition for timing
    double seq_start = omp_get_wtime();
    sequentialVectorAdd(A, B, C_seq, N);
    double seq_end = omp_get_wtime();
    double seq_time = seq_end - seq_start;

    // Output result vector (trimmed if large)
    cout << "\nResult Vector C (first 5 elements):\n";
    for (int i = 0; i < min(5, N); ++i) {
        cout << C[i] << " ";
    }
    if (N > 5) cout << "...";
    cout << endl;

    // Output execution times and stats
    cout << "\nExecution Times:\n";
    cout << "CUDA Vector Addition: " << fixed << setprecision(6) << cuda_time_ms / 1000.0 << " seconds\n";
    cout << "Sequential Vector Addition: " << seq_time << " seconds\n";
    cout << "Speedup (Sequential / CUDA): " << seq_time / (cuda_time_ms / 1000.0) << "x\n";

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}