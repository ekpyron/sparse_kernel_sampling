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
#include <MDS/MDS.hpp>
#include <utility/Arguments.hpp>
#include <iomanip>
#include <fstream>

#ifdef USE_MPFR
#include <mpreal.h>
#endif

int main(int argc, char *argv[]) {
    try {
        if (!Arguments::get().parse(argc, argv)) {
            return EXIT_SUCCESS;
        }

        typedef double float_type;

        std::unique_ptr<Data<float_type>> data;

        std::cout << "Fetch input data..." << std::endl;
        switch (Arguments::get().input_data_type()) {
            case InputDataType::TwoMoons:
                data = std::unique_ptr<Data<float_type>> (new TwoMoons<float_type> ());
                break;
            case InputDataType::Abalone:
                data = std::unique_ptr<Data<float_type>> (new Abalone<float_type> ());
                break;
            case InputDataType::MNIST:
                data = std::unique_ptr<Data<float_type>> (new MNIST<float_type> ());
                break;
        }
        std::cout << "DONE." << std::endl;

        std::cout << "Running oASIS..." << std::endl;
        oASIS<float_type> oasis (data.get());
        std::cout << "DONE." << std::endl;
        std::cout << "Running Nystrom..." << std::endl;
        Nystrom<float_type> nystrom (data.get(), oasis.k());
        std::cout << "DONE." << std::endl;
        std::cout << "oASIS (" << oasis.k() << "):" << std::endl << "  Error: " << std::setprecision(4) << std::scientific << oasis.GetError(data.get()) << " Runtime: " << oasis.GetRuntime() << " s" << std::endl;
        std::cout << "Nystrom (" << nystrom.k() << "):" << std::endl << "  Error: " << std::setprecision(4) << std::scientific << nystrom.GetError(data.get()) << " Runtime: " << nystrom.GetRuntime() << " s" << std::endl;

        MDS<float_type> mds (oasis.W());

        auto Ctransp = oasis.Ctransp();
        auto &&Lt = mds.Lt();

        for (auto i = 0; i < Ctransp.cols(); i++) {
            Ctransp.col(i) -= mds.avg();
        }

        auto Embedd = Lt * Ctransp;

        std::cout << "Ctransp(" << Ctransp.rows() << ", " << Ctransp.cols() << ")" << std::endl;
        std::cout << "Lt(" << Lt.rows() << ", " << Lt.cols() << ")" << std::endl;
        std::cout << "Embedd(" << Embedd.rows() << ", " << Embedd.cols() << ")" << std::endl;
        std::cout << "U(" << oasis.U().rows() << ", " << oasis.U().cols() << ")" << std::endl;

        {
            std::ofstream f ("output1", std::ios_base::out);
            for (auto i = 0; i < Lt.cols(); i++) {
                f << Lt.row(0).col(i) << " " << Lt.row(1).col(i) << std::endl;
                //f << oasis.U().row(oasis.Lambda()[i]).col(0) << " " << oasis.U().row(oasis.Lambda()[i]).col(1) << std::endl;
                //f << U.row(oasis.Lambda()[i]).col(0) << " " << U.row(oasis.Lambda()[i]).col(1) << std::endl;
            }
        }

        {
            std::ofstream f ("output", std::ios_base::out);
            for (auto i = 0; i < Embedd.cols(); i++) {
                f << Embedd.row(0).col(i) << " " << Embedd.row(1).col(i) << std::endl;
                //f << oasis.U().row(i).col(0) << " " << oasis.U().row(i).col(1) << std::endl;
            }
        }

        {
            std::ofstream f ("points", std::ios_base::out);
            for (auto &idx : oasis.Lambda()) {
                f << idx << std::endl;
            }
        }

        return EXIT_SUCCESS;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch(...) {
        std::cerr << "Unknown error." << std::endl;
        return EXIT_FAILURE;
    }
}
