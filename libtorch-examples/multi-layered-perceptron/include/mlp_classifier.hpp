#ifndef MLP_CLASSIFIER_HPP
#define MLP_CLASSIFIER_HPP

#include <iostream>
#include <vector>
#include <cstring>
#include <torch/torch.h>
#include "vendors/cpptqdm/tqdm.h"

#include "mlp.hpp"
#include "mnist.hpp"
#include "utils.hpp"


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
            const std::string& dataPath, int64_t batchSize,
            size_t numEpochs, const std::string& checkpointDirectory) {

        createDirectory(checkpointDirectory);
        createDirectory(checkpointDirectory + "/model");
        createDirectory(checkpointDirectory + "/optimizer");

        MNIST mnistDataset;
        MNIST::DataLoaderType dataLoader = mnistDataset.getDataLoader(dataPath, batchSize, true);
        unsigned long numSamples = mnistDataset.numSamples;
        std::cout << "Number of Training Samples: " << numSamples << std::endl;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Training Started..." << std::endl;

        for (size_t epoch = 1; epoch <= numEpochs; ++epoch) {

            std::cout << "Epoch [" << epoch << "/" << numEpochs << "]" << std::endl;

            double batchLoss = 0.0;
            size_t numCorrect = 0;
            int batchCounter = 0;
            int numBatches = numSamples / batchSize;
            clock_t startTime = std::clock();
            tqdm progressBar;

            for (auto& batch : *dataLoader) {
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
                progressBar.progress(++batchCounter, numBatches);
            }

            progressBar.finish();
            auto meanBatchLoss = batchLoss / (double)numSamples;
            auto batchAccuracy = static_cast<double>(numCorrect) / (double)numSamples;
            lossHistory.push_back(meanBatchLoss);
            accuracyHistory.push_back(batchAccuracy);
            (*model).save(checkpointDirectory, "mlp_classification_model_" + std::to_string(epoch));
            std::string optimizerCheckpoint = checkpointDirectory + "/optimizer/logistic_regression_optimizer_" + std::to_string(epoch) + ".pt";
            torch::save(*optimizer, optimizerCheckpoint);
            std::cout << ", Loss: " << meanBatchLoss << ", Accuracy: " << batchAccuracy;
            std::cout << ", Time Taken: " << float(std::clock() - startTime) / CLOCKS_PER_SEC << " seconds" << std::endl;
        }

        std::cout << "Training Completed!!!" << std::endl;
    }

    void evaluate(const std::string& dataPath, int64_t batchSize) const {

        MNIST mnistDataset;
        MNIST::DataLoaderType dataLoader = mnistDataset.getDataLoader(dataPath, batchSize, true);
        unsigned long numSamples = mnistDataset.numSamples;
        std::cout << "Number of Samples for Evaluation: " << numSamples << std::endl;

        (*model).eval();
        torch::NoGradGuard noGrad;

        std::cout << "Evaluation Started..." << std::endl;

        double totalLoss = 0.0;
        size_t numCorrect = 0;
        tqdm progressBar;
        clock_t startTime = std::clock();
        int batchCounter = 0;
        int numBatches = numSamples / batchSize;

        for(auto& batch: *dataLoader) {
            auto data = batch.data.view({batchSize, -1}).to(*device);
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

#endif // MLP_CLASSIFIER_HPP
