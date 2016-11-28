#include "RuntimeMonitor.hpp"
#include <utility/Arguments.hpp>
#include <iostream>

RuntimeMonitorScope::RuntimeMonitorScope(RuntimeMonitor& monitor, const std::string &name) : monitor_(monitor), name_(name) {
    if (Arguments::get().verbose() && !name_.empty ()) {
        std::cout << "  " << name_ << "..." << std::endl;
    }
    start_time = std::chrono::steady_clock::now();
}

RuntimeMonitorScope::~RuntimeMonitorScope() {
    auto elapsed_time = std::chrono::steady_clock::now ()-start_time;
    monitor_.time += elapsed_time;
    if (Arguments::get().verbose() && !name_.empty ()) {
        std::cout << "  DONE (" << std::chrono::duration<double>(elapsed_time).count() << " s)." << std::endl;
    }
}
