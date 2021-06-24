#include "linear_regression.hpp"

int main() {
    LinearRegression linearRegression(0, 10, 15);
    linearRegression.compile(0.001);
    linearRegression.train(60);
    return 0;
}
