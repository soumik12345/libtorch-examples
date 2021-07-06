#ifndef LIBTORCH_EXAMPLES_MNIST_HPP
#define LIBTORCH_EXAMPLES_MNIST_HPP

#include <cstring>
#include <torch/torch.h>

class MNIST {

public:

    static auto getDataset(const std::string& mnistDataPath, bool isTrain) {
        if(isTrain)
            return torch::data::datasets::MNIST(mnistDataPath)
                    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                    .map(torch::data::transforms::Stack<>());
        return torch::data::datasets::MNIST(mnistDataPath)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());
    }
};

#endif //LIBTORCH_EXAMPLES_MNIST_HPP
