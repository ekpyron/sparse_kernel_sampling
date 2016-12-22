#include "RuntimeMonitor.hpp"
#include <utility/Arguments.hpp>
#include <iostream>
#include <iomanip>

RuntimeMonitorScope::RuntimeMonitorScope(RuntimeMonitor& monitor, const std::string &name) : monitor_(monitor), name_(name) {
    if (Arguments::get().verbose() && !name_.empty ()) {
        std::cout << "  " << name_ << "..." << std::endl;
    }
    start_time_ = std::chrono::steady_clock::now();
}

RuntimeMonitorScope::~RuntimeMonitorScope() {
    auto elapsed_time = std::chrono::steady_clock::now ()-start_time_;
    monitor_.time_ += elapsed_time;
    if (Arguments::get().verbose() && !name_.empty ()) {
        std::cout << "  DONE (" << std::scientific << std::setprecision(2) << std::chrono::duration<double>(elapsed_time).count() << " s)." << std::endl;
    }
}

RuntimeMonitorScopeSuspend::RuntimeMonitorScopeSuspend(RuntimeMonitorScope& scope) : scope_ (scope) {
    start_time_ = std::chrono::steady_clock::now();
}

RuntimeMonitorScopeSuspend::~RuntimeMonitorScopeSuspend() {
    auto elapsed_time = std::chrono::steady_clock::now () - start_time_;
    scope_.start_time_ += elapsed_time;
}