#include <iostream>
#include <cstdlib>
#include <exception>
#include "MNIST.h"
#include "oASIS.h"

int main(int argc, char *argv[]) {
    try {
        MNIST data (argc, argv);
        oASIS oasis (&data);
        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch(...) {
        std::cerr << "Unknown error." << std::endl;
        return EXIT_FAILURE;
    }
}
