#ifndef SPARSE_KERNEL_SAMPLING_ARGUMENTS_HPP
#define SPARSE_KERNEL_SAMPLING_ARGUMENTS_HPP

enum class InputDataType {
    Abalone,
    MNIST,
    TwoMoons
};

class Arguments {
public:
    static Arguments &get(void) {
        static Arguments arguments;
        return arguments;
    }
    ~Arguments(void);

    const InputDataType &input_data_type(void) const {
        return input_data_type_;
    }
    bool verbose(void) const {
        return verbose_;
    }

    bool parse(int argc, char **argv);

private:
    Arguments(void);
    bool verbose_;
    InputDataType input_data_type_;
};

#endif /* !defined SPARSE_KERNEL_SAMPLING_ARGUMENTS_HPP */
