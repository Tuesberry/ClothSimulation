#pragma once

#include <tuple>

#include <glm/glm.hpp>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
//#include "Header/helper_cuda.h"
//#include "Header/helper_string.h"
//#include "Header/helper_math.h"

#include "vector_types.h"
#include "vector_functions.h"
//#include <thrust/device_ptr.h>
//#include <thrust/transform.h>
//#include <thrust/sort.h>

#define CONST(type)				const type const
#define GET_CUDA_ID(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= maxID) return
#define GET_CUDA_ID_NO_RETURN(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x
#define EPSILON					1e-6f

#ifdef __CUDACC__ 
#define CUDA_CALL(func, totalThreads)  \
	if (totalThreads == 0) return; \
	uint func ## _numBlocks, func ## _numThreads; \
	ComputeGridSize(totalThreads, func ## _numBlocks, func ## _numThreads); \
	func <<<func ## _numBlocks, func ## _numThreads >>>
#define CUDA_CALL_S(func, totalThreads, stream)  \
	if (totalThreads == 0) return; \
	uint func ## _numBlocks, func ## _numThreads; \
	ComputeGridSize(totalThreads, func ## _numBlocks, func ## _numThreads); \
	func <<<func ## _numBlocks, func ## _numThreads, stream>>>
#define CUDA_CALL_V(func, ...) \
	func <<<__VA_ARGS__>>>
#else
#define CUDA_CALL(func, totalThreads)
#define CUDA_CALL_S(func, totalThreads, stream) 
#define CUDA_CALL_V(func, ...)
#endif


typedef unsigned int uint;

const uint BLOCK_SIZE = 256;

__device__ inline float length2(glm::vec3 vec)
{
	return glm::dot(vec, vec);
}

inline void ComputeGridSize(const uint& n, uint& numBlocks, uint& numThreads)
{
	if (n == 0)
	{
		//fmt::print("Error(Solver): numParticles is 0\n");
		numBlocks = 0;
		numThreads = 0;
		return;
	}
	numThreads = glm::min(n, BLOCK_SIZE);
	numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
}

template<class T>
inline T* VtAllocBuffer(size_t elementCount)
{
	T* devPtr = nullptr;
	cudaMallocManaged((void**)&devPtr, elementCount * sizeof(T));
	cudaDeviceSynchronize(); // this is necessary, otherwise realloc can cause crash
	return devPtr;
}

inline void VtFreeBuffer(void* buffer)
{
	cudaFree(buffer);
	//checkCudaErrors(cudaFree(buffer));
}

