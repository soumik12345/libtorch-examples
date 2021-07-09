#ifndef LIBTORCH_EXAMPLES_CIFAR10_HPP
#define LIBTORCH_EXAMPLES_CIFAR10_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <fstream>
#include <torch/types.h>
#include <torch/data/example.h>
#include <torch/data/datasets/base.h>

class CIFAR10: public torch::data::datasets::Dataset<CIFAR10> {

private:

    torch::Tensor _images;
    torch::Tensor _targets;
    bool isTrain;

public:

    const uint32_t trainSize = 50000;
    const uint32_t testSize = 10000;
    const uint32_t sizePerBatch = 10000;
    const uint32_t imageRows = 32;
    const uint32_t imageColumns = 32;
    const uint32_t bytesPerRow = 3073;
    const uint32_t bytesPerChannelPerRow = 1024;
    const uint32_t bytesPerBatchFile = bytesPerRow * sizePerBatch;
    std::vector<std::string> trainDataBatchFiles, testDataBatchFiles;

    CIFAR10(const std::string& rootDirectory, bool isTrain) {
        trainDataBatchFiles = {
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin",
        };
        testDataBatchFiles = {"test_batch.bin"};
        auto data = readData(rootDirectory, isTrain);
        _images = std::move(data.first);
        _targets = std::move(data.second);
        this->isTrain = isTrain;
    }

    std::pair<torch::Tensor, torch::Tensor> readData(const std::string& rootDirectory, bool isTrain) {
        const auto& files = isTrain ? trainDataBatchFiles : testDataBatchFiles;
        const uint32_t numSamples = isTrain ? trainSize : testSize;
        std::vector<char> dataBuffer;
        dataBuffer.reserve(files.size() * bytesPerBatchFile);
        for(auto& file: files) {
            auto path = rootDirectory + "/" + file;
            std::ifstream data(path, std::ios::binary);
            TORCH_CHECK(data, "Error opening data file", path);
            dataBuffer.insert(dataBuffer.end(), std::istreambuf_iterator<char>(data), {});
        }
        TORCH_CHECK(dataBuffer.size() == files.size() * bytesPerBatchFile, "Unexpected file sizes");
        auto images = torch::empty({numSamples, 3, imageRows, imageColumns}, torch::kByte);
        auto targets = torch::empty(numSamples, torch::kByte);
        for(uint32_t i = 0; i != numSamples; ++i) {
            uint32_t startIndex = i * bytesPerRow;
            targets[i] = dataBuffer[startIndex];
            uint32_t imageStart = startIndex + 1;
            uint32_t  imageEnd = imageStart + 3 * bytesPerChannelPerRow;
            std::copy(
                    dataBuffer.begin() + imageStart,
                    dataBuffer.begin() + imageEnd,
                    reinterpret_cast<char*>(images[i].data_ptr()));
        }
        return {
            images.to(torch::kFloat32).div_(255),
            targets.to(torch::kInt64)};
    }

    torch::optional<size_t> size() const {
        return _images.size(0);
    }

    torch::data::Example<> get(size_t index) override {
        return {
            _images[(int64_t)index],
            _targets[(int64_t)index]
        };
    }

    torch::Tensor& getImages() {
        return _images;
    }

    torch::Tensor& getTargets() {
        return _targets;
    }
};

#endif //LIBTORCH_EXAMPLES_CIFAR10_HPP
