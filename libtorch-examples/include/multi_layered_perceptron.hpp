//
// Created by Soumik Rakshit on 26/06/21.
//

#ifndef LIBTORCH_EXAMPLES_MULTI_LAYERED_PERCEPTRON_HPP
#define LIBTORCH_EXAMPLES_MULTI_LAYERED_PERCEPTRON_HPP

#include <iostream>
#include <vector>
#include <cstring>
#include <torch/torch.h>

#include "utils.hpp"

struct MLP : torch::nn::Module {

    torch::nn::Linear *denseLayer1;
    torch::nn::Linear *denseLayer2;

    MLP(int64_t inputSize, int64_t hiddenSize, int64_t numClasses) {
        denseLayer1 = new torch::nn::Linear(inputSize, hiddenSize);
        denseLayer2 = new torch::nn::Linear(hiddenSize, numClasses);
        register_module("denseLayer1", (*denseLayer1));
        register_module("denseLayer2", (*denseLayer2));
    }

    torch::Tensor forward(torch::Tensor x) const {
        x = (*denseLayer1)->forward(x);
        x = torch::nn::functional::relu(x);
        x = (*denseLayer2)->forward(x);
        return x;
    }
};


class MLPClassifier {

public:

    torch::Device *device;
    MLP* model{};
    torch::optim::SGD *optimizer{};
    std::vector<double> lossHistory;
    std::vector<double> accuracyHistory;

    MLPClassifier() {
        device = new torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    }

    void compile(int64_t inputImageSize, int64_t hiddenSize, double learningRate) {
        model = new MLP(inputImageSize, hiddenSize, 10);
        (*model).to(*device);
        optimizer = new torch::optim::SGD(
                (*model).parameters(),
                torch::optim::SGDOptions(learningRate));
    }

    void train(
            const std::string& mnistDataPath, int64_t batchSize,
            size_t numEpochs, const std::string& checkpointDirectory) {

        createDirectory(checkpointDirectory);
        createDirectory(checkpointDirectory + "/model");
        createDirectory(checkpointDirectory + "/optimizer");

        auto trainDataset = torch::data::datasets::MNIST(mnistDataPath)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());
        unsigned long numTrainSamples = trainDataset.size().value();
        std::cout << "Number of Training Samples: " << numTrainSamples << std::endl;
        auto trainLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(trainDataset), batchSize);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Training Started..." << std::endl;

        for (size_t epoch = 1; epoch <= numEpochs; ++epoch) {

            double batchLoss = 0.0;
            size_t numCorrect = 0;

            for (auto& batch : *trainLoader) {
                auto data = batch.data.view({batchSize, -1}).to(*device);
                auto target = batch.target.to(*device);
                auto output = (*model).forward(data);
                auto loss = torch::nn::functional::cross_entropy(output, target);
                batchLoss += loss.item<double>() * (double)data.size(0);
                auto prediction = output.argmax(1);
                numCorrect += prediction.eq(target).sum().item<int64_t>();
                (*optimizer).zero_grad();
                loss.backward();
                (*optimizer).step();
            }
            auto meanBatchLoss = batchLoss / (double)numTrainSamples;
            auto batchAccuracy = static_cast<double>(numCorrect) / (double)numTrainSamples;
            lossHistory.push_back(meanBatchLoss);
            accuracyHistory.push_back(batchAccuracy);
//            std::string modelCheckpoint = checkpointDirectory + "/model/logistic_regression_model_" + std::to_string(epoch) + ".pt";
//            std::string optimizerCheckpoint = checkpointDirectory + "/optimizer/logistic_regression_optimizer_" + std::to_string(epoch) + ".pt";
//            torch::save(*model, modelCheckpoint);
//            torch::save(*optimizer, optimizerCheckpoint);
            std::cout << "Epoch [" << epoch << "/" << numEpochs << "]";
            std::cout << ", Loss: " << batchLoss << ", Accuracy: " << batchAccuracy << std::endl;
        }

        std::cout << "Training Completed!!!" << std::endl;
    }

    void evaluate(const std::string& mnistDataPath, int64_t batchSize) const {

        auto testDataset = torch::data::datasets::MNIST(mnistDataPath)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());
        unsigned long numTestSamples = testDataset.size().value();
        std::cout << "Number of Training Samples: " << numTestSamples << std::endl;
        auto testLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(testDataset), batchSize);

        (*model).eval();
        torch::NoGradGuard noGrad;

        double totalLoss = 0.0;
        size_t numCorrect = 0;

        std::cout << "Evaluation Started..." << std::endl;

        for(auto& batch: *testLoader) {
            auto data = batch.data.view({batchSize, -1}).to(*device);
            auto target = batch.target.to(*device);
            auto output = (*model).forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);
            totalLoss += loss.item<double>() * (double)data.size(0);
            auto prediction = output.argmax(1);
            numCorrect += prediction.eq(target).sum().item<int64_t>();
        }

        std::cout << "Evaluation Completed!!!" << std::endl;

        auto testAccuracy = static_cast<double>(numCorrect) / (double)numTestSamples;
        auto meanLoss = totalLoss / (double)numTestSamples;

        std::cout << "On Test Dataset, Mean Loss: " << meanLoss << ", Accuracy: " << testAccuracy << std::endl;
    }
};

inline void MLPClassifierDemo() {
    MLPClassifier classifier;
    classifier.compile(784, 500, 0.001);
    classifier.train("../data/mnist", 100, 5, "checkpoints");
    classifier.evaluate("../data/mnist", 100);
}

#endif //LIBTORCH_EXAMPLES_MULTI_LAYERED_PERCEPTRON_HPP
