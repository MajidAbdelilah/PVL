#include <cstddef>
#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>

template<typename T>
class Lp_parallel_vector: public std::vector<T>
{
public:
    Lp_parallel_vector(): std::vector<T>() {
        num_thread = std::thread::hardware_concurrency();
    };
    ~Lp_parallel_vector() {
        // Only join threads that are joinable (have been started)
        for(size_t i = 0; i < num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
    };

    Lp_parallel_vector(size_t num_elements) : std::vector<T>(num_elements) {
        num_thread = std::thread::hardware_concurrency();
    };
    
    Lp_parallel_vector(const Lp_parallel_vector& other) : std::vector<T>(other) {
        num_thread = other.num_thread;
        // Don't copy threads as they can't be copied
    };
    Lp_parallel_vector& operator=(const Lp_parallel_vector& other) {
        if(this != &other) {
            std::vector<T>::operator=(other);
            num_thread = other.num_thread;
            // Don't copy threads as they can't be copied
        }
        return *this;
    }
    Lp_parallel_vector(const std::vector<T>& other) : std::vector<T>(other) {
        num_thread = std::thread::hardware_concurrency();
    };
    
    Lp_parallel_vector& operator=(const std::vector<T>& other) {
        std::vector<T>::operator=(other);
        return *this;
    }
    Lp_parallel_vector(const std::initializer_list<T>& init) : std::vector<T>(init) {
        num_thread = std::thread::hardware_concurrency();
    };
    
    Lp_parallel_vector& operator=(const std::initializer_list<T>& init) {
        std::vector<T>::operator=(init);
        return *this;
    }

    void fill(T value) {
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, value, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    this->at(j) = value;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
    }

    void fill(T value, size_t size)
    {
        this->resize(size);
        fill(value);
    }

    void fill(std::function<T(T&, size_t)> func) {
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, func, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    this->at(j) = func(this->at(j), j);
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
    }

    void fill(std::function<T(T&, size_t)> func, size_t size)
    {
        this->resize(size);
        fill(func);
    }


    Lp_parallel_vector<T> operator+(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) + other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator-(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) - other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator*(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) * other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator/(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) / other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator&&(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) && other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator||(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) || other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator!()
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = !this->at(j);
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator==(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) == other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator==(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) == other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator!=(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) != other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator!=(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) != other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator<(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) < other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator<(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) < other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator>(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) > other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator>(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) > other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator<=(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) <= other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator<=(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) <= other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator>=(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) >= other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator>=(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) >= other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };

    Lp_parallel_vector<T> operator&(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) & other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator|(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) | other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator^(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) ^ other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator%(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) % other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator<<(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) << other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator>>(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) >> other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator+(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) + other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator-(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) - other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator*(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) * other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator/(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) / other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator%(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) % other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator&(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) & other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator|(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) | other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator^(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) ^ other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator<<(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) << other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator>>(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) >> other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator~()
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = ~this->at(j);
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };

private:
    size_t num_thread;
    std::thread threads[128];
};
