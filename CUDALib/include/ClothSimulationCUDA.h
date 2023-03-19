// Copyright SCIEMENT, Inc.
// by Hirofumi Seo, M.D., CEO & President

#pragma once

#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include "common.cuh"
#include "Common.h"

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size, std::string* error_message);
cudaError_t addWithCuda2(int4* c, const int4* a, const int4* b, std::string* error_message);

cudaError_t testCollision_VF(float3* face1, float3* face2, float* result, std::string* error_message);
cudaError_t testCollision_EE(float3* face1, float3* face2, float* result, std::string* error_message);

cudaError_t testCollisions_VF(float3* face[3], int numFace, std::vector<std::vector<float>>& result, std::vector<std::vector<int>>& combResult, std::string* error_message);
cudaError_t testCollisions_EE(float3* face[3], int numFace, std::vector<std::vector<float>>& result, std::vector<std::vector<int>>& combResult, std::string* error_message);
