wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu.zip
rm libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu.zip
# shellcheck disable=SC2164
cd libtorch-examples/include/vendors
git clone https://github.com/aminnj/cpptqdm
cd ../../../
mkdir build
# shellcheck disable=SC2164
cd build
cmake -DINSTALL_TORCH=false -DCUDA=false -DINSTALL_PYTHON_DEPENDENCIES=false -DDOWNLOAD_MNIST=false ../
make
