#include "MNIST.h"
#include <stdexcept>
#include <memory>
#include <cmath>

inline uint32_t byteswap32(uint32_t v) {
    return ((v&0xFF)<<24)|(((v>>8)&0xFF)<<16)|(((v>>16)&0xFF)<<8)|((v>>24)&0xFF);
}

MNIST::MNIST(int argc, char **argv) : train_images_ ("train-images-idx3-ubyte", std::ios_base::in|std::ios_base::binary),
                                      two_sigma_squared_ (2.0) {
    if (!train_images_.is_open())
        throw std::runtime_error("Cannot open MNIST training image file.");

    header_t header;
    train_images_.read(reinterpret_cast<char *> (&header), sizeof(header));
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
}

MNIST::~MNIST(void) {
}

float MNIST::distance (uint64_t i, uint64_t j) {
    std::unique_ptr<uint8_t[]> data_i(new uint8_t[rows_ * columns_]);
    std::unique_ptr<uint8_t[]> data_j(new uint8_t[rows_ * columns_]);
    train_images_.seekg (sizeof (header_t) + i * rows_ * columns_);
    train_images_.read(reinterpret_cast<char*> (data_i.get()), rows_ * columns_);
    train_images_.seekg (sizeof (header_t) + j * rows_ * columns_);
    train_images_.read(reinterpret_cast<char*> (data_j.get()), rows_ * columns_);
    double dist = 0.0;
    for (auto i = 0; i < rows_*columns_; i++) {
        double d = (1.0/255.0) * (double (data_i[i]) - double (data_j[i]));
        dist += d*d;
    }
    return std::exp(-dist/two_sigma_squared_);
}
