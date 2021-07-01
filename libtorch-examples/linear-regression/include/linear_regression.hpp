#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <iostream>
#include <torch/torch.h>

#include "utils.hpp"

class LinearRegression {

public:

    torch::Tensor xTrain;
    torch::Tensor yTrain;
    torch::Device *device;
    torch::nn::Linear *model{};
    torch::optim::SGD *optimizer{};
    std::vector<double> lossHistory;

    LinearRegression(int64_t dataLow, int64_t dataHigh, int datasetSize) {
        device = new torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        xTrain = torch::randint(dataLow, dataHigh, {datasetSize, 1}).to(*device);
        yTrain = torch::randint(dataLow, dataHigh, {datasetSize, 1}).to(*device);
    }

    void compile(double learningRate) {
        model = new torch::nn::Linear(1, 1);
        (*model)->to(*device);
        optimizer = new torch::optim::SGD((*model)->parameters(), learningRate);
    }

    void train(size_t numEpochs, const std::string& checkpointDirectory) {
        createDirectory(checkpointDirectory);
        createDirectory(checkpointDirectory + "/model");
        createDirectory(checkpointDirectory + "/optimizer");
        for(size_t epoch = 1; epoch <= numEpochs; ++epoch) {
            auto output = (*model)(xTrain);
            auto loss = torch::nn::functional::mse_loss(output, yTrain);
            (*optimizer).zero_grad();
            loss.backward();
            (*optimizer).step();
            std::cout << "Epoch [" << epoch << "/" << numEpochs << "]";
            std::cout << ", Loss: " << loss.item<double>() << std::endl;
            lossHistory.push_back(loss.item<double>());
            std::string modelCheckpoint = checkpointDirectory + "/model/linear_regression_model_" + std::to_string(epoch) + ".pt";
            std::string optimizerCheckpoint = checkpointDirectory + "/optimizer/linear_regression_optimizer_" + std::to_string(epoch) + ".pt";
            torch::save(*model, modelCheckpoint);
            torch::save(*optimizer, optimizerCheckpoint);
        }
        std::cout << "Training Done!!!" << std::endl;
    }
};

#endif //LINEAR_REGRESSION_HPP
