#include "logistic_regression.hpp"

int main() {
    LogisticRegression logisticRegression;
    logisticRegression.compile(784, 0.001);
    logisticRegression.train(
            "../../../data/mnist", 100,
            5, "../../checkpoints");
    logisticRegression.evaluate("../../../data/mnist", 100);
    return 0;
}
