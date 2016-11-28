#ifndef SPARSE_KERNEL_SAMPLING_RUNTIMEMONITOR_HPP
#define SPARSE_KERNEL_SAMPLING_RUNTIMEMONITOR_HPP

#include <chrono>
#include <string>
#include <sstream>

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
    RuntimeMonitorScope(RuntimeMonitor &monitor, const std::string &name = std::string());
    template<typename... Args>
    RuntimeMonitorScope(RuntimeMonitor &monitor, Args... args) : RuntimeMonitorScope (monitor, ArgsToString (args...)) {
    }
    ~RuntimeMonitorScope(void);
private:
    std::string name_;
    RuntimeMonitor &monitor_;
    std::chrono::time_point<std::chrono::steady_clock> start_time;

    template<typename... Args>
    static std::string ArgsToString(Args... args) {
        std::stringstream stream;
        ArgsToStream(stream, args...);
        return stream.str();
    }
    template<typename T, typename... Args>
    static void ArgsToStream(std::stringstream &stream, T &t, Args... args) {
        stream << t;
        ArgsToStream (stream, args...);
    }
    static void ArgsToStream(std::stringstream &stream) {
    }
};


#endif /* !defined SPARSE_KERNEL_SAMPLING_RUNTIMEMONITOR_HPP */
