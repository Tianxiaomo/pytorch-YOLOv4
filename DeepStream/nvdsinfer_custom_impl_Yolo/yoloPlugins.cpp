/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "yoloPlugins.h"
#include "NvInferPlugin.h"
#include <cassert>
#include <iostream>
#include <memory>

namespace {
template <typename T>
void write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void read(const char*& buffer, T& val)
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}
} //namespace

// Forward declaration of cuda kernels
cudaError_t cudaYoloLayerV3 (
    const void* input, void* output, const uint& batchSize,
    const uint& gridSize, const uint& numOutputClasses,
    const uint& numBBoxes, uint64_t outputSize, cudaStream_t stream);

YoloLayerV3::YoloLayerV3 (const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data);
    read(d, m_NumBoxes);
    read(d, m_NumClasses);
    read(d, m_GridSize);
    read(d, m_OutputSize);
};

YoloLayerV3::YoloLayerV3 (
    const uint& numBoxes, const uint& numClasses, const uint& gridSize) :
    m_NumBoxes(numBoxes),
    m_NumClasses(numClasses),
    m_GridSize(gridSize)
{
    assert(m_NumBoxes > 0);
    assert(m_NumClasses > 0);
    assert(m_GridSize > 0);
    m_OutputSize = m_GridSize * m_GridSize * (m_NumBoxes * (4 + 1 + m_NumClasses));
};

nvinfer1::Dims
YoloLayerV3::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    assert(index == 0);
    assert(nbInputDims == 1);
    return inputs[0];
}

bool YoloLayerV3::supportsFormat (
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const {
    return (type == nvinfer1::DataType::kFLOAT &&
            format == nvinfer1::PluginFormat::kNCHW);
}

void
YoloLayerV3::configureWithFormat (
    const nvinfer1::Dims* inputDims, int nbInputs,
    const nvinfer1::Dims* outputDims, int nbOutputs,
    nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
{
    assert(nbInputs == 1);
    assert (format == nvinfer1::PluginFormat::kNCHW);
    assert(inputDims != nullptr);
}

int YoloLayerV3::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace,
    cudaStream_t stream)
{
    CHECK(cudaYoloLayerV3(
              inputs[0], outputs[0], batchSize, m_GridSize, m_NumClasses, m_NumBoxes,
              m_OutputSize, stream));
    return 0;
}

size_t YoloLayerV3::getSerializationSize() const
{
    return sizeof(m_NumBoxes) + sizeof(m_NumClasses) + sizeof(m_GridSize) + sizeof(m_OutputSize);
}

void YoloLayerV3::serialize(void* buffer) const
{
    char *d = static_cast<char*>(buffer);
    write(d, m_NumBoxes);
    write(d, m_NumClasses);
    write(d, m_GridSize);
    write(d, m_OutputSize);
}

nvinfer1::IPluginV2* YoloLayerV3::clone() const
{
    return new YoloLayerV3 (m_NumBoxes, m_NumClasses, m_GridSize);
}

REGISTER_TENSORRT_PLUGIN(YoloLayerV3PluginCreator);
