/*
 * Copyright (c) 2018-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

inline __device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void gpuYoloLayerV3(const float* input, float* output, const uint gridSize, const uint numOutputClasses,
                               const uint numBBoxes)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
    uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x_id >= gridSize) || (y_id >= gridSize) || (z_id >= numBBoxes))
    {
        return;
    }

    const int numGridCells = gridSize * gridSize;
    const int bbindex = y_id * gridSize + x_id;

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]);

    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]);

    for (uint i = 0; i < numOutputClasses; ++i)
    {
        output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]
            = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]);
    }
}

cudaError_t cudaYoloLayerV3(const void* input, void* output, const uint& batchSize, const uint& gridSize,
                            const uint& numOutputClasses, const uint& numBBoxes,
                            uint64_t outputSize, cudaStream_t stream);

cudaError_t cudaYoloLayerV3(const void* input, void* output, const uint& batchSize, const uint& gridSize,
                            const uint& numOutputClasses, const uint& numBBoxes,
                            uint64_t outputSize, cudaStream_t stream)
{
    dim3 threads_per_block(16, 16, 4);
    dim3 number_of_blocks((gridSize / threads_per_block.x) + 1,
                          (gridSize / threads_per_block.y) + 1,
                          (numBBoxes / threads_per_block.z) + 1);
    for (unsigned int batch = 0; batch < batchSize; ++batch)
    {
        gpuYoloLayerV3<<<number_of_blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const float*>(input) + (batch * outputSize),
            reinterpret_cast<float*>(output) + (batch * outputSize), gridSize, numOutputClasses,
            numBBoxes);
    }
    return cudaGetLastError();
}
