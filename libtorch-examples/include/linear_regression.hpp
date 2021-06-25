//
// Created by geekyrakshit on 6/23/21.
//

#ifndef TORCH_CPP_LINEAR_REGRESSION_HPP
#define TORCH_CPP_LINEAR_REGRESSION_HPP

#include <iostream>
#include <vector>
#include <cstring>
#include <torch/torch.h>

#include "utils.hpp"

class LinearRegression {

public:

    torch::Tensor xTrain;
    torch::Tensor yTrain;
    std::vector<float> lossHistory;
    torch::Device *device;
    torch::nn::Linear *model;
    torch::optim::SGD *optimizer;

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

    void train(size_t numEpochs, std::string checkpointDirectory) {
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
            std::string modelCheckpoint = checkpointDirectory + "/model/" + "model_" + std::to_string(epoch) + ".pt";
            std::string optimizerCheckpoint = checkpointDirectory + "/optimizer/" + "optimizer_" + std::to_string(epoch) + ".pt";
            torch::save(*model, modelCheckpoint);
            torch::save(*optimizer, optimizerCheckpoint);
        }
        std::cout << "Training Done!!!" << std::endl;
    }

};

inline void LinearRegressionDemo() {
    LinearRegression linearRegression(0, 10, 15);
    linearRegression.compile(0.001);
    linearRegression.train(60, "checkpoints");
}

#endif //TORCH_CPP_LINEAR_REGRESSION_HPP
