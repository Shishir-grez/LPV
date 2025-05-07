#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <iomanip> // For setprecision
#include <omp.h> // For omp_get_wtime

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

// Kernel for tiled matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Sequential matrix multiplication for verification
void sequentialMatrixMul(const vector<float>& A, const vector<float>& B, vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Function to generate random matrix
vector<float> generateRandomMatrix(int rows, int cols) {
    vector<float> mat(rows * cols);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 10.0f);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dis(gen);
    }
    return mat;
}

int main() {
    int M = 1024;  // Matrix A: 1024 rows
    int K = 1024;  // Matrix A: 1024 columns, Matrix B: 1024 rows
    int N = 1024;  // Matrix B: 1024 columns
    
    cout << "Using hardcoded matrix dimensions: A(" << M << "x" << K << ") * B(" << K << "x" << N << ") = C(" << M << "x" << N << ")" << endl;
    
    // Always use random matrices
    cout << "Generating random matrices..." << endl;
    vector<float> A = generateRandomMatrix(M, K);
    vector<float> B = generateRandomMatrix(K, N);
    vector<float> C(M * N), C_seq(M * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Set up kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch kernel and measure time
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cuda_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&cuda_time_ms, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Sequential matrix multiplication for timing
    double seq_start = omp_get_wtime();
    sequentialMatrixMul(A, B, C_seq, M, N, K);
    double seq_end = omp_get_wtime();
    double seq_time = seq_end - seq_start;

    // Output result matrix (trimmed if large)
    cout << "\nResult Matrix C (first 5 elements, row-major):\n";
    for (int i = 0; i < min(5, M * N); ++i) {
        cout << C[i] << " ";
    }
    if (M * N > 5) cout << "...";
    cout << endl;

    // Output execution times and stats
    cout << "\nExecution Times:\n";
    cout << "CUDA Matrix Multiplication: " << fixed << setprecision(6) << cuda_time_ms / 1000.0 << " seconds\n";
    cout << "Sequential Matrix Multiplication: " << seq_time << " seconds\n";
    cout << "Speedup (Sequential / CUDA): " << seq_time / (cuda_time_ms / 1000.0) << "x\n";

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}