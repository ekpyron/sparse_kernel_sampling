#ifndef SPARSE_KERNEL_SAMPLING_RUNTIMEMONITOR_HPP
#define SPARSE_KERNEL_SAMPLING_RUNTIMEMONITOR_HPP

#include <chrono>

class RuntimeMonitor {
public:
    RuntimeMonitor(void) : time (0.0) {
    }
    ~RuntimeMonitor(void) {
    }
    const std::chrono::duration<double> &get(void) const {
        return time;
    }
private:
    friend class RuntimeMonitorScope;
    std::chrono::duration<double> time;
};

class RuntimeMonitorScope {
public:
    RuntimeMonitorScope(RuntimeMonitor &monitor) : monitor_(monitor), start_time (std::chrono::steady_clock::now()) {
    }
    ~RuntimeMonitorScope(void) {
        auto end_time = std::chrono::steady_clock::now ();
        monitor_.time += (end_time-start_time);
    }
private:
    RuntimeMonitor &monitor_;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_RUNTIMEMONITOR_HPP */
