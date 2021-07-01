#include "linear_regression.hpp"

int main() {
    std::cout << "---------------- Linear Regression Demo ----------------" << std::endl;
    LinearRegression linearRegression(0, 10, 15);
    linearRegression.compile(0.001);
    linearRegression.train(60, "../../checkpoints");
    return 0;
}
