# LibTorch Examples [WIP]

<table>
  <tr>
    <th>MacOSX (Clang 12.0)</th>
    <th>Linux (Cuda 11)</th>
    <th>Linux (gcc 9.3)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/soumik12345/libtorch-examples/workflows/build-cmake-macos/badge.svg" alt="build-failing"></td>
    <td><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="build-failing"></td>
    <td><img src="https://github.com/soumik12345/libtorch-examples/workflows/build-cmake-linux/badge.svg" alt="build-failing"></td>
  </tr>
</table>

## Instructions

### Build and Compile

1. `git clone https://github.com/soumik12345/libtorch-examples --recursive`

2. `cmake -DINSTALL_TORCH=true -DCUDA=true -DINSTALL_PYTHON_DEPENDENCIES=true -DDOWNLOAD_MNIST=true ../`

    - `-DINSTALL_TORCH` -> Automatically download and install libtorch during build.
    - `-DCUDA` -> Download GPU Version of LibTorch.
    - `-DINSTALL_PYTHON_DEPENDENCIES` -> Install Python Dependencies during build.
    - `-DDOWNLOAD_MNIST` -> Download the MNIST Dataset during build.
    
3. `make`

### Run Executables

<table>
    <tr>
        <th>Implementation</th>
        <th>Run Instructions</th>
        <th>MNIST</th>
        <th>CIFAR10</th>
    </tr>
    <tr>
        <td><a href="./libtorch-examples/linear-regression">Linear Regression</a></td>
        <td>
            <ol>
                <li><code>cd build</code></li>
                <li><code>./libtorch-examples/linear-regression/linear-regression</code></li>
            </ol>
        </td>
        <td>&#10060;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td><a href="./libtorch-examples/logistic-regression">Logistic Regression</a></td>
        <td>
            <ol>
                <li><code>cd ./build/libtorch-examples/logistic-regression</code></li>
                <li><code>./logistic-regression</code></li>
            </ol>
        </td>
        <td>&#9989;</td>
        <td>&#9202;</td>
    </tr>
    <tr>
        <td><a href="./libtorch-examples/multi-layered-perceptron">Multi-Layered Perceptron</a></td>
        <td>
            <ol>
                <li><code>cd ./build/libtorch-examples/multi-layered-perceptron</code></li>
                <li><code>./multi-layered-perceptron</code></li>
            </ol>
        </td>
        <td>&#9989;</td>
        <td>&#9202;</td>
    </tr>
    <tr>
        <td><a href="./libtorch-examples/convnet">Convolutional Neural Network</a></td>
        <td>
            <ol>
                <li><code>cd ./build/libtorch-examples/convnet</code></li>
                <li><code>./convnet</code></li>
            </ol>
        </td>
        <td>&#9989;</td>
        <td>&#9202;</td>
    </tr>
</table>
