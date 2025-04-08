#include <cstddef>
#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>

#include <sycl/sycl.hpp>

// A SYCL-compatible function wrapper (non-virtual version)
template<typename T>
class SyclFunction {
public:
    SyclFunction() {}
    
    // This is the function that will be called from SYCL kernels
    // Remove the virtual keyword to avoid virtual calls in kernels
    SYCL_EXTERNAL T operator()(T val, size_t idx) const {
        return val * 2; // Default implementation, doubles the value
    }
};

// Allow SYCL to copy this class to devices
template<typename T>
struct sycl::is_device_copyable<SyclFunction<T>> : std::true_type {};

template<typename T>
class Lp_parallel_vector_GPU: public std::vector<T>
{
public:
    Lp_parallel_vector_GPU(): std::vector<T>() {
        try {
            // Try to select a GPU device
            q = sycl::queue(sycl::gpu_selector_v);
            is_gpu = true;
        } catch (const sycl::exception& e) {
            std::cerr << "GPU not available, falling back to CPU: " << e.what() << std::endl;
            q = sycl::queue(sycl::cpu_selector_v);
            is_gpu = false;
        }
    };
    
    ~Lp_parallel_vector_GPU() {
        // q.quit();
        // SYCL queue manages resource cleanup automatically
    };

    Lp_parallel_vector_GPU(size_t num_elements) : std::vector<T>(num_elements) {
        try {
            q = sycl::queue(sycl::gpu_selector_v);
            is_gpu = true;
        } catch (const sycl::exception& e) {
            std::cerr << "GPU not available, falling back to CPU: " << e.what() << std::endl;
            q = sycl::queue(sycl::cpu_selector_v);
            is_gpu = false;
        }
    };
    
    Lp_parallel_vector_GPU(const Lp_parallel_vector_GPU& other) : std::vector<T>(other) {
        q = other.q;
    };
    
    Lp_parallel_vector_GPU& operator=(const Lp_parallel_vector_GPU& other) {
        if(this != &other) {
            std::vector<T>::operator=(other);
            q = other.q;
            is_gpu = other.is_gpu;
        }
        return *this;
    }
    
    Lp_parallel_vector_GPU(const std::vector<T>& other) : std::vector<T>(other) {
        try {
            q = sycl::queue(sycl::gpu_selector_v);
            is_gpu = true;
        } catch (const sycl::exception& e) {
            std::cerr << "GPU not available, falling back to CPU: " << e.what() << std::endl;
            q = sycl::queue(sycl::cpu_selector_v);
            is_gpu = false;
        }
    };
    
    Lp_parallel_vector_GPU& operator=(const std::vector<T>& other) {
        std::vector<T>::operator=(other);
        return *this;
    }
    
    Lp_parallel_vector_GPU(const std::initializer_list<T>& init) : std::vector<T>(init) {
        try {
            q = sycl::queue(sycl::gpu_selector_v);
            is_gpu = true;
        } catch (const sycl::exception& e) {
            std::cerr << "GPU not available, falling back to CPU: " << e.what() << std::endl;
            q = sycl::queue(sycl::cpu_selector_v);
            is_gpu = false;
        }
    };
    
    Lp_parallel_vector_GPU& operator=(const std::initializer_list<T>& init) {
        std::vector<T>::operator=(init);
        return *this;
    }

    void fill(T value) {
        if (this->empty()) return;
        
        // Create SYCL buffer from the vector data
        sycl::buffer<T, 1> buf(this->data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffer
            auto acc = buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                acc[idx] = value;
            });
        });
        
        // Wait for operations to complete
        q.wait();
    }

    void fill(T value, size_t size) {
        this->resize(size);
        fill(value);
    }

    // Modify the fill method to avoid virtual function calls
    void fill(const SyclFunction<T>& func) {
        if (this->empty()) return;
        
        // Create SYCL buffer from the vector data
        sycl::buffer<T, 1> buf(this->data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffer
            auto acc = buf.template get_access<sycl::access::mode::read_write>(h);
            
            // Run the kernel in parallel - without calls_indirectly property
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                // Call the functor directly - now it's non-virtual
                acc[idx] = func(acc[idx], idx);
            });
        });
        
        // Wait for operations to complete
        q.wait();
    }

    // Keep the existing fill method with std::function for CPU fallback
    void fill(std::function<T(T, size_t)> func) {
        if (this->empty()) return;
        
        // For CPU fallback, we can just loop through elements
        if (!is_gpu) {
            for (size_t i = 0; i < this->size(); i++) {
                (*this)[i] = func((*this)[i], i);
            }
            return;
        }
        
        // Otherwise, we use a default SyclFunction and warn the user
        std::cerr << "Warning: std::function is not fully SYCL compatible. Using default implementation." << std::endl;
        SyclFunction<T> sycl_func;
        fill(sycl_func);
    }

    void fill(std::function<T(T&, size_t)> func, size_t size) {
        this->resize(size);
        fill(func);
    }

    Lp_parallel_vector_GPU<T> operator+(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] + b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator-(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] - b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator*(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] * b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator/(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                // Add a check to avoid division by zero
                if (b_acc[idx] != 0) {
                    res_acc[idx] = a_acc[idx] / b_acc[idx];
                } else {
                    res_acc[idx] = 0; // or some other appropriate value
                }
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator%(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] % b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator&(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] & b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator|(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] | b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator^(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] ^ b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator<<(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] << b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator>>(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] >> b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator&&(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] && b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator||(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] || b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator==(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] == b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator!=(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] != b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator<(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] < b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator<=(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] <= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator>(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] > b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator>=(const Lp_parallel_vector_GPU<T>& other) {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] >= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    // += -= *= /= %= &= |= ^= <<= >>=
    Lp_parallel_vector_GPU<T>& operator+=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] += b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator-=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] -= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator*=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] *= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator/=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                // Add a check to avoid division by zero
                if (b_acc[idx] != 0) {
                    a_acc[idx] /= b_acc[idx];
                } else {
                    a_acc[idx] = 0; // or some other appropriate value
                }
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator%=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] %= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator&=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] &= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator|=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] |= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator^=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] ^= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator<<=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] <<= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator>>=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] >>= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    

    // Implement rest of operators similarly
    Lp_parallel_vector_GPU<T> operator+(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] + other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator-(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] - other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator*(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] * other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator/(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                // Add a check to avoid division by zero
                if (other != 0) {
                    res_acc[idx] = a_acc[idx] / other;
                } else {
                    res_acc[idx] = 0; // or some other appropriate value
                }
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator%(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] % other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator&(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] & other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator|(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] | other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator^(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] ^ other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator<<(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] << other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator>>(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] >> other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator~() {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = ~a_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    // && || == != < <= > >=
    // Implement rest of operators similarly
    Lp_parallel_vector_GPU<T> operator&&(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] && other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator||(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] || other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator==(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] == other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator!=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] != other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator<(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] < other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator<=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] <= other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator>(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] > other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator>=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] >= other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator!() {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = !a_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator++() {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = ++a_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator--() {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = --a_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator+=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] += other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator-=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] -= other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator*=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] *= other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator/=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                // Add a check to avoid division by zero
                if (other != 0) {
                    res_acc[idx] = a_acc[idx] /= other;
                } else {
                    res_acc[idx] = 0; // or some other appropriate value
                }
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator%=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] %= other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator&=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] &= other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator|=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] |= other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator^=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] ^= other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator<<=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] <<= other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator>>=(const T& other) {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] >>= other;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    

private:
    sycl::queue q;
    bool is_gpu;
};
