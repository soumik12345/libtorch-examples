#include "convnet_classifier.hpp"

int main() {
    ConvNetClassifier classifier;
    classifier.compile(10, 0.001);
    classifier.train("../../../data/mnist", 100, 5, "../../checkpoints");
    classifier.evaluate("../../../data/mnist", 100);
}
