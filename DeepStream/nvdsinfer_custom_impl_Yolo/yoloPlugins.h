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

#ifndef __YOLO_PLUGINS__
#define __YOLO_PLUGINS__

#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

#include "NvInferPlugin.h"

#define CHECK(status)                                                                              \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }

namespace
{
const char* YOLOV3LAYER_PLUGIN_VERSION {"1"};
const char* YOLOV3LAYER_PLUGIN_NAME {"YoloLayerV3_TRT"};
} // namespace

class YoloLayerV3 : public nvinfer1::IPluginV2
{
public:
    YoloLayerV3 (const void* data, size_t length);
    YoloLayerV3 (const uint& numBoxes, const uint& numClasses, const uint& gridSize);
    const char* getPluginType () const override { return YOLOV3LAYER_PLUGIN_NAME; }
    const char* getPluginVersion () const override { return YOLOV3LAYER_PLUGIN_VERSION; }
    int getNbOutputs () const override { return 1; }

    nvinfer1::Dims getOutputDimensions (
        int index, const nvinfer1::Dims* inputs,
        int nbInputDims) override;

    bool supportsFormat (
        nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;

    void configureWithFormat (
        const nvinfer1::Dims* inputDims, int nbInputs,
        const nvinfer1::Dims* outputDims, int nbOutputs,
        nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override;

    int initialize () override { return 0; }
    void terminate () override {}
    size_t getWorkspaceSize (int maxBatchSize) const override { return 0; }
    int enqueue (
        int batchSize, const void* const* inputs, void** outputs,
        void* workspace, cudaStream_t stream) override;
    size_t getSerializationSize() const override;
    void serialize (void* buffer) const override;
    void destroy () override { delete this; }
    nvinfer1::IPluginV2* clone() const override;

    void setPluginNamespace (const char* pluginNamespace)override {
        m_Namespace = pluginNamespace;
    }
    virtual const char* getPluginNamespace () const override {
        return m_Namespace.c_str();
    }

private:
    uint m_NumBoxes {0};
    uint m_NumClasses {0};
    uint m_GridSize {0};
    uint64_t m_OutputSize {0};
    std::string m_Namespace {""};
};

class YoloLayerV3PluginCreator : public nvinfer1::IPluginCreator
{
public:
    YoloLayerV3PluginCreator () {}
    ~YoloLayerV3PluginCreator () {}

    const char* getPluginName () const override { return YOLOV3LAYER_PLUGIN_NAME; }
    const char* getPluginVersion () const override { return YOLOV3LAYER_PLUGIN_VERSION; }

    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        std::cerr<< "YoloLayerV3PluginCreator::getFieldNames is not implemented" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2* createPlugin (
        const char* name, const nvinfer1::PluginFieldCollection* fc) override
    {
        std::cerr<< "YoloLayerV3PluginCreator::getFieldNames is not implemented.\n";
        return nullptr;
    }

    nvinfer1::IPluginV2* deserializePlugin (
        const char* name, const void* serialData, size_t serialLength) override
    {
        std::cout << "Deserialize yoloLayerV3 plugin: " << name << std::endl;
        return new YoloLayerV3(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override {
        m_Namespace = libNamespace;
    }
    const char* getPluginNamespace() const override {
        return m_Namespace.c_str();
    }

private:
    std::string m_Namespace {""};
};

#endif // __YOLO_PLUGINS__
