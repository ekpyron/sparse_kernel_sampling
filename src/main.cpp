#include <iostream>
#include <cstdlib>
#include <exception>
#include <memory>
#include "data/Abalone.hpp"
#include "data/MNIST.hpp"
#include "data/TwoMoons.hpp"
#include "oASIS.h"
#include "Nystrom.hpp"

int main(int argc, char *argv[]) {
    try {
        std::unique_ptr<Data> data;
        std::string dataset(argc > 1 ? argv[1] : "TwoMoons");
        if (!dataset.compare("TwoMoons")) {
            data = std::unique_ptr<Data> (new TwoMoons (argc, argv));
        } else if (!dataset.compare("Abalone")) {
            data = std::unique_ptr<Data> (new Abalone (argc, argv));
        } else if (!dataset.compare("MNIST")) {
            data = std::unique_ptr<Data> (new MNIST (argc, argv));
        }

        std::cout << "oASIS:" << std::endl;
        oASIS oasis (data.get());
        std::cout << std::endl << "Nystrom:" << std::endl;
        Nystrom nystrom (data.get(), 200);
        std::cout << std::endl << "Results:" << std::endl;
        std::cout << "oASIS:" << std::endl;
        oasis.CheckResult(data.get());
        std::cout << std::endl << "Nystrom:" << std::endl;
        nystrom.CheckResult(data.get());

        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch(...) {
        std::cerr << "Unknown error." << std::endl;
        return EXIT_FAILURE;
    }
}
