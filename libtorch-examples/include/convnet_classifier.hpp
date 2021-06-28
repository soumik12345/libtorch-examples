//
// Created by Soumik Rakshit on 28/06/21.
//

#ifndef LIBTORCH_EXAMPLES_CONVNET_CLASSIFIER_H
#define LIBTORCH_EXAMPLES_CONVNET_CLASSIFIER_H

#include <iostream>
#include <vector>
#include <cstring>
#include <ctime>
#include <torch/torch.h>
#include "vendors/cpptqdm/tqdm.h"

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

class ConvNetClassifier {

public:

    torch::Device *device{};
    ConvNet *model{};
    torch::optim::Adam *optimizer{};
    std::vector<double> lossHistory;
    std::vector<double> accuracyHistory;

    ConvNetClassifier() {
        device = new torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    }

    void compile(int64_t numClasses, double learningRate) {
        model = new ConvNet(numClasses);
        (*model).to(*device);
        optimizer = new torch::optim::Adam(
                (*model).parameters(),
                torch::optim::AdamOptions(learningRate));
    }

    void train(const std::string& mnistDataPath, int64_t batchSize,
               size_t numEpochs, const std::string& checkpointDirectory) {

        createDirectory(checkpointDirectory);
        createDirectory(checkpointDirectory + "/model");
        createDirectory(checkpointDirectory + "/optimizer");

        auto trainDataset = torch::data::datasets::MNIST(mnistDataPath)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());
        auto numSamples = trainDataset.size().value();
        std::cout << "Number of Training Samples: " << numSamples << std::endl;
        auto trainLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(trainDataset), batchSize);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Training Started..." << std::endl;

        for(size_t epoch = 1; epoch <= numEpochs; ++epoch) {

            std::cout << "Epoch [" << epoch << "/" << numEpochs << "]" << std::endl;

            double batchLoss = 0.0;
            size_t numCorrect = 0;
            int batchCounter = 0;
            int numBatches = numSamples / batchSize;
            clock_t startTime = std::clock();
            tqdm progressBar;

            for(auto& batch : *trainLoader) {
                auto data = batch.data.to(*device);
                auto target = batch.target.to(*device);
                auto output = (*model).forward(data);
                auto loss = torch::nn::functional::cross_entropy(output, target);
                batchLoss += loss.item<double>() * (double)data.size(0);
                auto prediction = output.argmax(1);
                numCorrect += prediction.eq(target).sum().item<int64_t>();
                (*optimizer).zero_grad();
                loss.backward();
                (*optimizer).step();
                progressBar.progress(++batchCounter, numBatches);
            }

            progressBar.finish();
            auto meanBatchLoss = batchLoss / (double)numSamples;
            auto batchAccuracy = static_cast<double>(numCorrect) / (double)numSamples;
            lossHistory.push_back(meanBatchLoss);
            accuracyHistory.push_back(batchAccuracy);
            (*model).save(checkpointDirectory, "convnet_classification_model_" + std::to_string(epoch));
            std::string optimizerCheckpoint = checkpointDirectory + "/optimizer/logistic_regression_optimizer_" + std::to_string(epoch) + ".pt";
            torch::save(*optimizer, optimizerCheckpoint);
            std::cout << ", Loss: " << meanBatchLoss << ", Accuracy: " << batchAccuracy;
            std::cout << ", Time Taken: " << float(std::clock() - startTime) / CLOCKS_PER_SEC << " seconds" << std::endl;
        }

        std::cout << "Training Completed!!!" << std::endl;
    }

    void evaluate(const std::string& mnistDataPath, int64_t batchSize) const {

        auto testDataset = torch::data::datasets::MNIST(mnistDataPath)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());
        unsigned long numSamples = testDataset.size().value();
        std::cout << "Number of Testing Samples: " << numSamples << std::endl;
        auto testLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(testDataset), batchSize);

        (*model).eval();
        torch::NoGradGuard noGrad;

        std::cout << "Evaluation Started..." << std::endl;

        double totalLoss = 0.0;
        size_t numCorrect = 0;
        tqdm progressBar;
        clock_t startTime = std::clock();
        int batchCounter = 0;
        int numBatches = numSamples / batchSize;

        for(auto& batch: *testLoader) {
            auto data = batch.data.to(*device);
            auto target = batch.target.to(*device);
            auto output = (*model).forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);
            totalLoss += loss.item<double>() * (double)data.size(0);
            auto prediction = output.argmax(1);
            numCorrect += prediction.eq(target).sum().item<int64_t>();
            progressBar.progress(++batchCounter, numBatches);
        }

        progressBar.finish();

        std::cout << "Evaluation Completed!!!" << std::endl;
        std::cout << "Time Taken: " << float(std::clock() - startTime) / CLOCKS_PER_SEC << " seconds" << std::endl;

        auto testAccuracy = static_cast<double>(numCorrect) / (double)numSamples;
        auto meanLoss = totalLoss / (double)numSamples;

        std::cout << "On Test Dataset, Mean Loss: " << meanLoss << ", Accuracy: " << testAccuracy << std::endl;
    }
};

inline void ConvNetClassifierDemo() {
    ConvNetClassifier classifier;
    classifier.compile(10, 0.001);
    classifier.train("../data/mnist", 100, 5, "checkpoints");
    classifier.evaluate("../data/mnist", 100);
}

#endif //LIBTORCH_EXAMPLES_CONVNET_CLASSIFIER_H