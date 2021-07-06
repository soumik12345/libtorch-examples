//
// Created by geekyrakshit on 6/25/21.
//

#ifndef LIBTORCH_EXAMPLES_UTILS_HPP
#define LIBTORCH_EXAMPLES_UTILS_HPP

#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

void createDirectory(const std::string& directory) {
    std::__fs::filesystem::remove_all(directory);
    if (mkdir(directory.c_str(), 0777) == -1)
        std::cerr << "Unable to create Directory " << directory << " :  " << strerror(errno) << std::endl;
    else
        std::cout << directory << " created" << std::endl;
}

#endif //LIBTORCH_EXAMPLES_UTILS_HPP
