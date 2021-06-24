//
// Created by geekyrakshit on 6/23/21.
//

#ifndef TORCH_CPP_LINEAR_REGRESSION_HPP
#define TORCH_CPP_LINEAR_REGRESSION_HPP

#include <iostream>
#include <vector>
#include <torch/torch.h>

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

    void train(size_t numEpochs) {
        for(size_t epoch = 1; epoch <= numEpochs; ++epoch) {
            auto output = (*model)(xTrain);
            auto loss = torch::nn::functional::mse_loss(output, yTrain);
            (*optimizer).zero_grad();
            loss.backward();
            (*optimizer).step();
            std::cout << "Epoch [" << epoch << "/" << numEpochs << "]";
            std::cout << ", Loss: " << loss.item<double>() << std::endl;
            lossHistory.push_back(loss.item<double>());
        }
        std::cout << "Training Done!!!" << std::endl;
    }

};

#endif //TORCH_CPP_LINEAR_REGRESSION_HPP
