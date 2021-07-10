#include "logistic_regression.hpp"

int main(int argc, char *argv[]) {
    if (argc > 1) {
        if (std::strcmp(argv[1], "mnist") == 0) {
            LogisticRegression logisticRegression;
            logisticRegression.compile(784, 0.001);
            logisticRegression.train(
                    "../../../data/mnist", 100,
                    5, "../../checkpoints");
            logisticRegression.evaluate("../../../data/mnist", 100);
        }
    } else {
        std::cerr << "Invalid Arguments" << std::endl;
    }
    return 0;
}
