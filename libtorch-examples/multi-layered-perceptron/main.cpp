#include "mlp.hpp"

int main() {
    MLPClassifier classifier;
    classifier.compile(784, 500, 0.001);
    classifier.train("../../../data/mnist", 100, 5, "../../checkpoints");
    classifier.evaluate("../../../data/mnist", 100);
}
