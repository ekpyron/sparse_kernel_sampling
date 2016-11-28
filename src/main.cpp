#include <iostream>
#include <cstdlib>
#include <exception>
#include <memory>
#include <data/Data.hpp>
#include <data/Abalone.hpp>
#include <data/MNIST.hpp>
#include <data/TwoMoons.hpp>
#include <sampling/oASIS.h>
#include <sampling/Nystrom.hpp>
#include <utility/Arguments.hpp>
#include <iomanip>

int main(int argc, char *argv[]) {
    try {
        if (!Arguments::get().parse(argc, argv)) {
            return EXIT_SUCCESS;
        }

        std::unique_ptr<Data> data;

        std::cout << "Fetch input data..." << std::endl;
        switch (Arguments::get().input_data_type()) {
            case InputDataType::TwoMoons:
                data = std::unique_ptr<Data> (new TwoMoons ());
                break;
            case InputDataType::Abalone:
                data = std::unique_ptr<Data> (new Abalone ());
                break;
            case InputDataType::MNIST:
                data = std::unique_ptr<Data> (new MNIST ());
                break;
        }
        std::cout << "DONE." << std::endl;

        std::cout << "Running oASIS..." << std::endl;
        oASIS oasis (data.get());
        std::cout << "DONE." << std::endl;
        std::cout << "Running Nystrom..." << std::endl;
        Nystrom nystrom (data.get(), 200);
        std::cout << "DONE." << std::endl;
        std::cout << "oASIS:   Error: " << std::setprecision(15) << std::fixed << oasis.GetError(data.get()) << " Runtime: " << oasis.GetRuntime() << " s" << std::endl;
        std::cout << "Nystrom: Error: " << std::setprecision(15) << std::fixed << nystrom.GetError(data.get()) << " Runtime: " << nystrom.GetRuntime() << " s" << std::endl;

        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch(...) {
        std::cerr << "Unknown error." << std::endl;
        return EXIT_FAILURE;
    }
}
