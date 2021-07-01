#ifndef CONVNET_HPP
#define CONVNET_HPP

#include <cstring>
#include <torch/torch.h>

#include "utils.hpp"

class ConvNet : public torch::nn::Module {

private:

    torch::nn::Sequential convBlock1 {
            torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(
                            1, 16, 5).stride(1).padding(2)),
            torch::nn::BatchNorm2d(16), torch::nn::ReLU(),
            torch::nn::MaxPool2d(
                    torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential convBlock2 {
            torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(
                            16, 32, 5).stride(1).padding(2)),
            torch::nn::BatchNorm2d(32), torch::nn::ReLU(),
            torch::nn::MaxPool2d(
                    torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Linear denseLayer;

public:

    explicit ConvNet(int64_t numClasses) : denseLayer(7 * 7 * 32, numClasses) {
        register_module("convBlock1", convBlock1);
        register_module("convBlock2", convBlock2);
        register_module("denseLayer", denseLayer);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = convBlock1->forward(x);
        x = convBlock2->forward(x);
        x = x.view({-1, 7 * 7 * 32});
        return denseLayer->forward(x);
    }

    void save(const std::string& checkpointDir, const std::string& checkpointName) {
        createDirectory(checkpointDir + "/model/" + checkpointName);
        torch::save(convBlock1, checkpointDir + "/model/" + checkpointName + "/convBlock1.pt");
        torch::save(convBlock2, checkpointDir + "/model/" + checkpointName + "/convBlock2.pt");
        torch::save(denseLayer, checkpointDir + "/model/" + checkpointName + "/denseLayer.pt");
    }
};

#endif // CONVNET_HPP