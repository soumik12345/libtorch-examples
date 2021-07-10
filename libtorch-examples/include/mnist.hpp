#ifndef LIBTORCH_EXAMPLES_MNIST_HPP
#define LIBTORCH_EXAMPLES_MNIST_HPP

#include <cstring>
#include <utility>
#include <torch/torch.h>

class MNIST {

public:

    using DatasetType = torch::data::datasets::MapDataset
            <torch::data::datasets::MapDataset
                    <torch::data::datasets::MNIST,
                    torch::data::transforms::Normalize<>>,
                    torch::data::transforms::Stack<>>;

    using DataLoaderType = std::unique_ptr<torch::data::StatelessDataLoader
            <torch::data::datasets::MapDataset
                    <torch::data::datasets::MapDataset
                            <torch::data::datasets::MNIST,
                            torch::data::transforms::Normalize<> >,
                            torch::data::transforms::Stack<torch::data::Example<> > >,
                            torch::data::samplers::RandomSampler>,
                            std::default_delete<torch::data::StatelessDataLoader
                                    <torch::data::datasets::MapDataset
                                            <torch::data::datasets::MapDataset
                                                    <torch::data::datasets::MNIST,
                                                    torch::data::transforms::Normalize<> >,
                                                    torch::data::transforms::Stack<torch::data::Example<> > >,
                                                    torch::data::samplers::RandomSampler> > >;

    unsigned long numSamples;

    MNIST::DataLoaderType getDataLoader(const std::string& mnistDataPath, int64_t batchSize, bool isTrain) {
        MNIST::DatasetType trainDataset = torch::data::datasets::MNIST(
                mnistDataPath, isTrain ? torch::data::datasets::MNIST::Mode::kTrain : torch::data::datasets::MNIST::Mode::kTest)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());
        numSamples = trainDataset.size().value();
        MNIST::DataLoaderType trainLoader = torch::data::make_data_loader
                <torch::data::samplers::RandomSampler>(
                std::move(trainDataset), batchSize);
        return trainLoader;
    }
};

#endif //LIBTORCH_EXAMPLES_MNIST_HPP
