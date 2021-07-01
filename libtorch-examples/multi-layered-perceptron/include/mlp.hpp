#ifndef MLP_HPP
#define MLP_HPP

#include <cstring>
#include <torch/torch.h>

#include "utils.hpp"

class MLP : public torch::nn::Module {

private:
    torch::nn::Linear denseLayer1;
    torch::nn::Linear denseLayer2;

public:

    MLP(int64_t inputSize, int64_t hiddenSize, int64_t numClasses) :
            denseLayer1(inputSize, hiddenSize),
            denseLayer2(hiddenSize, numClasses) {
        register_module("denseLayer1", denseLayer1);
        register_module("denseLayer2", denseLayer2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = denseLayer1->forward(x);
        x = torch::nn::functional::relu(x);
        return denseLayer2->forward(x);
    }

    void save(const std::string& checkpointDir, const std::string& checkpointName) {
        createDirectory(checkpointDir + "/model/" + checkpointName);
        torch::save(denseLayer1, checkpointDir + "/model/" + checkpointName + "/denseLayer1.pt");
        torch::save(denseLayer2, checkpointDir + "/model/" + checkpointName + "/denseLayer2.pt");
    }

    void load(const std::string& checkpointDir, const std::string& checkpointName) {
        torch::load(
                denseLayer1, checkpointDir + "/model/" + checkpointName + "/denseLayer1.pt");
        torch::load(
                denseLayer1, checkpointDir + "/model/" + checkpointName + "/denseLayer2.pt");
    }

};

#endif // MLP_HPP