#include "MNIST.hpp"
#include <stdexcept>
#include <memory>
#include <cmath>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iostream>

inline uint32_t byteswap32(uint32_t v) {
    return ((v&0xFF)<<24)|(((v>>8)&0xFF)<<16)|(((v>>16)&0xFF)<<8)|((v>>24)&0xFF);
}

MNIST::MNIST(int argc, char **argv) : two_sigma_squared_ (392.0) {
    struct stat sb;
    fd = open("train-images-idx3-ubyte", O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Cannot open MNIST training image file.");
    }
    if (fstat(fd, &sb) == -1) {
        throw std::runtime_error("Cannot get MNIST training image file size.");
    }

    mem = reinterpret_cast<uint8_t*> (mmap (nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0));
    if (mem == MAP_FAILED) {
        throw std::runtime_error("Cannot map MNIST training image file.");
    }

    header_t &header = *reinterpret_cast<header_t*>(mem);
    if (header.magic == 2051) {
        num_items_ = header.num_items;
        rows_ = header.rows;
        columns_ = header.columns;
    } else if (byteswap32(header.magic) == 2051) {
        num_items_ = byteswap32(header.num_items);
        rows_ = byteswap32(header.rows);
        columns_ = byteswap32(header.columns);
    } else {
        throw std::runtime_error("Invalid magic number.");
    }
    mem += sizeof (header_t);
}

MNIST::~MNIST(void) {
}


Eigen::VectorXf MNIST::column (uint64_t i) const {
    Eigen::VectorXf c(num_items_);
    uint8_t *data_j = mem;
    uint8_t *data_i = mem + i * rows_ * columns_;
    for (auto j = 0; j < num_items_; j++) {
        float dist = 0.0;
        for (auto k = 0; k < rows_*columns_; k++) {
            float d = (1.0f/255.0f) * (float (int(data_i[k]) - int(*data_j++)));
            dist += d*d;
        }
        c(j) = std::exp(-dist/two_sigma_squared_);
    }
    return c;
}

Eigen::RowVectorXf MNIST::diagonal (void) const {
/*    Eigen::RowVectorXf d(num_items_);
    for (uint64_t i = 0; i < num_items_; i++) {
        d(i) = distance(i,i);
    }
    return d;*/
    Eigen::RowVectorXf d(num_items_);
    d.setOnes();
    return d;
}

float MNIST::distance (uint64_t i, uint64_t j) const {
    uint8_t *data_i = mem + i * rows_ * columns_;
    uint8_t *data_j = mem + j * rows_ * columns_;
    float dist = 0.0;
    for (auto k = 0; k < rows_*columns_; k++) {
        float d = (1.0f/255.0f) * (float (int(data_i[k]) - int(data_j[k])));
        dist += d*d;
    }
    return std::exp(-dist/two_sigma_squared_);
}
