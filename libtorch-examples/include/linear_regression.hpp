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
    torch::nn::Linear *model;
    torch::optim::SGD *optimizer;

    LinearRegression(int64_t dataLow, int64_t dataHigh, int datasetSize) {
        xTrain = torch::randint(dataLow, dataHigh, {datasetSize, 1});
        yTrain = torch::randint(dataLow, dataHigh, {datasetSize, 1});
    }

    void compile(double learningRate) {
        model = new torch::nn::Linear(1, 1);
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
            torch::save(*model, checkpointDirectory + "/model/" + "model_" + std::to_string(epoch) + ".pt");
            torch::save(*optimizer, checkpointDirectory + "/optimizer/" + "optimizer_" + std::to_string(epoch) + ".pt");
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
