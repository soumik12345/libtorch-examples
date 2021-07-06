#ifndef CONVNET_CLASSIFIER_HPP
#define CONVNET_CLASSIFIER_HPP

#include <iostream>
#include <vector>
#include <cstring>
#include <ctime>
#include <torch/torch.h>
#include "vendors/cpptqdm/tqdm.h"

#include "utils.hpp"
#include "mnist.hpp"
#include "convnet.hpp"

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

        auto trainDataset = MNIST::getDataset(mnistDataPath, true);
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

        auto testDataset = MNIST::getDataset(mnistDataPath, false);
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


#endif // CONVNET_CLASSIFIER_HPP