#include "Arguments.hpp"
#include <stdexcept>
#include <unordered_map>
#include <iostream>

Arguments::Arguments(void) : input_data_type_ (InputDataType::TwoMoons), verbose_(false) {
}

Arguments::~Arguments() {
}

bool Arguments::parse(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            if (argv[i][1] == '\0' || argv[i][2] != '\0') {
                throw std::runtime_error (std::string ("Invalid argument: ") + argv[i]);
            }
            switch (argv[i][1]) {
                case 'v':
                    verbose_ = true;
                    break;
                case 'd':
                    i++;
                    if (i >= argc) {
                        throw std::runtime_error ("Missing argument for -d");
                    } else {
                        std::unordered_map<std::string, InputDataType> input_data_type_map {
                                { "TwoMoons", InputDataType::TwoMoons },
                                { "Abalone", InputDataType::Abalone },
                                { "MNIST", InputDataType::MNIST }
                        };

                        auto it = input_data_type_map.find(argv[i]);
                        if (it == input_data_type_map.end()) {
                            throw std::runtime_error(std::string("Invalid input data type: ") + argv[i]);
                        }
                        input_data_type_ = it->second;
                    }
                    break;
                case 'h':
                    std::cout << "Usage: " << argv[0] << " [-h] [-v] [-d InputDataType]" << std::endl
                              << "    -h                 displays this help message" << std::endl
                              << "    -v                 verbose output" << std::endl
                              << "    -d InputDataType   specifies input data type" << std::endl;
                    return false;
            }
        }
    }

    return true;
}
