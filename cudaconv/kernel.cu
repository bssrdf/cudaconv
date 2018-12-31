
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define N 256 //Default matrix size NxN
#define A(i,j) A[(i)*cols+(j)]  // row-major layout
#define C(i,j) C[(i)*cols+(j)]  // row-major layout

__global__ void convolution(int *A, int *C)
{
	//Filter
	int filter[3][3] = { { 1, 2, 1 },{ 2, 4, 2 },{ 1, 2, 1 } };

	//Needs for row-major layout
	int cols = N + 2;
	//int i = blockIdx.y * blockDim.y + threadIdx.y;
	for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < N + 2; row += blockDim.x * gridDim.x) {
		for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < N + 2; col += blockDim.y * gridDim.y) {
			int value = 99999;

			if (0 < row && row < N + 1 && 0 < col && col < N + 1)
			{
				value = 0;
				value = value + A(row - 1, col - 1)	*  filter[0][0];
				value = value + A(row - 1, col)		*  filter[0][1];
				value = value + A(row - 1, col + 1)	*  filter[0][2];
				value = value + A(row, col - 1)		*  filter[1][0];
				value = value + A(row, col)			*  filter[1][1];
				value = value + A(row, col + 1)		*  filter[1][2];
				value = value + A(row + 1, col - 1)	*  filter[2][0];
				value = value + A(row + 1, col)		*  filter[2][1];
				value = value + A(row + 1, col + 1)	*  filter[2][2];
			}
			C(row, col) = value;
		}
	}

}

#define BLOCK_SIZE 16

int main(void)
{
	//Host variables
	int A[N + 2][N + 2] = {};//+2 for padding matrix
	int *C;

	//Device variables
	int *A_d = 0, *C_d = 0;

	//Needs for row-major layout
	int cols = N + 2;

	//Calculate memory size 
	int memorySize = (N + 2) * (N + 2);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Init matrix by 0
	for (int i = 0; i < N + 2; i++) {
		for (int j = 0; j < N + 2; j++) {
			A[i][j] = 0;
		}
	}

	//Generate random values between 0 and 9
	srand(time(NULL));
	for (int i = 1; i < N + 2; i++) {
		for (int j = 1; j < N + 2; j++) {
			A[i][j] = rand() % 10;
		}
	}

	C = (int *)malloc(sizeof(*C)*memorySize);

	cudaMalloc((void**)&A_d, sizeof(*A_d)*memorySize);
	cudaMalloc((void**)&C_d, sizeof(*C_d)*memorySize);

	//Copy from host to device
	cudaMemcpy(A_d, A, sizeof(*A_d)*memorySize, cudaMemcpyHostToDevice);
	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((N + 2) / dimBlock.x, (N + 2) / dimBlock.y);
	printf("%d, %d \n", dimGrid.x, dimGrid.y);
	cudaEventRecord(start);
	convolution << <dimGrid, dimBlock >> > (A_d, C_d);//Block-thread
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//Copy from device to host
	cudaMemcpy(C, C_d, sizeof(*C)*memorySize, cudaMemcpyDeviceToHost);

	////Print result
	for (int i = 0; i < N + 2; i++) {
		for (int j = 0; j < N + 2; j++) {
			printf("%d ", C(i, j));
		}
		printf("\n");
	}

	//Free memory
	cudaFree(C_d);
	cudaFree(A_d);
	free(C);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f", milliseconds);
	return EXIT_SUCCESS;
}