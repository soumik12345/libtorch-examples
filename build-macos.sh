wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip
unzip libtorch-macos-1.8.1.zip
rm libtorch-macos-1.8.1.zip
# shellcheck disable=SC2164
cd libtorch-examples/include/vendors
git clone https://github.com/aminnj/cpptqdm
cd ../../../
mkdir build
# shellcheck disable=SC2164
cd build
cmake -DINSTALL_TORCH=false -DCUDA=false -DINSTALL_PYTHON_DEPENDENCIES=false -DDOWNLOAD_MNIST=false ../
make
