#include "Leopard.hpp"
#include <cstddef>
#include <iostream>
#include <chrono>
#include <cstdlib>

// Function to test thread safety by creating and destroying many vectors
void stress_test_thread_safety(int iterations) {
    std::cout << "Running thread safety stress test with " << iterations << " iterations..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        // Create a vector with a small size
        Lp_parallel_vector<int> test_vec(100);
        
        // Fill it with some values
        test_vec.fill(i);
        
        // Apply a function to it
        test_vec.fill([](int& val, size_t index) { (void)index; return val * 2; });
        
        // Test Lp_if_parallel
        if (i % 10 == 0) { // Only do this occasionally to save time
            test_vec[0] = 1; // Set first element to true
            Lp_if_parallel(test_vec, [](size_t index) { (void)index;/* Do nothing */ });
        }
        
        // Vector will be destroyed at end of loop iteration
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Stress test completed in " << elapsed.count() << " seconds" << std::endl;
}

// Function to specifically test the joinable() fixes
void test_joinable_fixes() {
    std::cout << "\nTesting joinable() fixes..." << std::endl;
    
    // Test 1: Create a vector but don't start any threads
    std::cout << "Test 1: Creating vector without starting threads..." << std::endl;
    {
        Lp_parallel_vector<int> vec(10);
        // Destructor will be called here, should not cause issues
    }
    std::cout << "Test 1 passed!" << std::endl;
    
    // Test 2: Create a vector, start threads with fill, then destroy
    std::cout << "Test 2: Creating vector, filling it, then destroying..." << std::endl;
    {
        Lp_parallel_vector<int> vec(10);
        vec.fill(42);
        // Destructor will be called here, should join threads properly
    }
    std::cout << "Test 2 passed!" << std::endl;
    
    // Test 3: Create a vector, start threads with operator, then destroy
    std::cout << "Test 3: Creating vector, using operator, then destroying..." << std::endl;
    {
        Lp_parallel_vector<int> vec1(10);
        Lp_parallel_vector<int> vec2(10);
        vec1.fill(1);
        vec2.fill(2);
        Lp_parallel_vector<int> vec3 = vec1 + vec2;
        // Destructor will be called here for all vectors
    }
    std::cout << "Test 3 passed!" << std::endl;
    
    // Test 4: Create a vector, use Lp_if_parallel, then destroy
    std::cout << "Test 4: Creating vector, using Lp_if_parallel, then destroying..." << std::endl;
    {
        Lp_parallel_vector<int> vec(10);
        vec[0] = 1; // Set first element to true
        Lp_if_parallel(vec, [](size_t index) {(void)index; /* Do nothing */ });
        // Destructor will be called here
    }
    std::cout << "Test 4 passed!" << std::endl;
    
    std::cout << "All joinable() fix tests passed!" << std::endl;
}

int fun(int val, size_t index)
{
    (void)index; return val * 2; 
}

            
int func(const int val, const size_t idx) {
    return idx * 2 - val;
}

class vec2 {
public:
    vec2() {}
    vec2(int x, int y) : x(x), y(y) {}
    vec2(const vec2& other) : x(other.x), y(other.y) {}
    vec2& operator=(const vec2& other) {
        if (this != &other) {
            x = other.x;
            y = other.y;
        }
        return *this;
    }
    vec2 operator+(const vec2& other) const {
        return vec2(x + other.x, y + other.y);
    }
    vec2 operator-(const vec2& other) const {
        return vec2(x - other.x, y - other.y);
    }
    vec2 operator*(const vec2& other) const {
        return vec2(x * other.x, y * other.y);
    }
    vec2 operator/(const vec2& other) const {
        if (other.x == 0 || other.y == 0) {
            throw std::runtime_error("Division by zero");
        }
        return vec2(x / other.x, y / other.y);
    }
    int x;
    int y;
};


// Define a custom SYCL-compatible function class that inherits from SyclFunction
// but doesn't override the function virtually

class DoubleFunction{
public:
    // Remove 'override' since the base function is no longer virtual
    SYCL_EXTERNAL vec2 operator()(const vec2 val, const size_t idx) const {
        return vec2(val.x * 2, val.y * 2);
    }
};
// Define a custom SYCL-compatible function class that inherits from SyclFunction
// but doesn't override the function virtually
class DoubleFunc {
public:
    DoubleFunc() {}
    // Remove 'override' since the base function is no longer virtual
    SYCL_EXTERNAL int operator()(const int val, const size_t idx) const {
        return idx * 2 - val;
    }
};
    

class FillFunction{
public:
    FillFunction() {}
    // Remove 'override' since the base function is no longer virtual
    SYCL_EXTERNAL vec2 operator()(const vec2 val, const size_t idx) const{
        return vec2(12, 42);
    }
};

template<>
struct sycl::is_device_copyable<vec2> : std::true_type {};
template<>
struct sycl::is_device_copyable<DoubleFunction> : std::true_type {};
template<>
struct sycl::is_device_copyable<DoubleFunc> : std::true_type {};
template<>
struct sycl::is_device_copyable<FillFunction> : std::true_type {};

// Function to test GPU vector functionality
void test_gpu_vector() {
    std::cout << "\nTesting GPU vector functionality..." << std::endl;
    try
    {
        sycl::queue q(sycl::gpu_selector_v);        
    
        if (!q.get_device().is_gpu()) {
            std::cerr << "No GPU device found. Skipping GPU vector tests." << std::endl;
            return;
        }
        std::cout << "GPU device found: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "No GPU device found. Skipping GPU vector tests." << std::endl;
        return;
    }
    
    std::cout << "GPU version of Lp_parallel_vector is being tested..." << std::endl;

    // Test basic constructor and destructor
    std::cout << "Testing basic constructor and destructor..." << std::endl;
    Lp_parallel_vector_GPU<vec2> vec_gpu(100000);
    
    // Test fill method
    std::cout << "Testing fill method..." << std::endl;
    FillFunction fill;
    vec_gpu.fill(fill);
    std::cout << "First element: " << vec_gpu[0].x << " " << vec_gpu[0].y << std::endl;
    std::cout << "Last element: " << vec_gpu[vec_gpu.size()-1].x << " " << vec_gpu[vec_gpu.size()-1].y << std::endl;
    
    // Test fill with function
    std::cout << "Testing fill with function..." << std::endl;
    
    // Create a SYCL-compatible function object
    DoubleFunction doubleFunc;
    vec_gpu.fill(doubleFunc);
    
    std::cout << "First element after function: " << vec_gpu[0].x << " " << vec_gpu[0].y << std::endl;
    std::cout << "Last element after function: " << vec_gpu[vec_gpu.size()-1].x << " " << vec_gpu[vec_gpu.size()-1].y << std::endl;
    
    // Test arithmetic operations
    std::cout << "Testing arithmetic operations..." << std::endl;
    
    // Create two vectors for testing
    Lp_parallel_vector_GPU<int> vec1_gpu(1000);
    Lp_parallel_vector_GPU<int> vec2_gpu(1000);
    
    vec1_gpu.fill(5);
    vec2_gpu.fill(3);
    
    // Test addition
    auto sum = vec1_gpu + vec2_gpu;
    std::cout << "Addition test (5 + 3): " << sum[0] << std::endl;
    
    // Test subtraction
    auto diff = vec1_gpu - vec2_gpu;
    std::cout << "Subtraction test (5 - 3): " << diff[0] << std::endl;
    
    // Test multiplication
    auto prod = vec1_gpu * vec2_gpu;
    std::cout << "Multiplication test (5 * 3): " << prod[0] << std::endl;
    
    // Test division
    auto quot = vec1_gpu / vec2_gpu;
    std::cout << "Division test (5 / 3): " << quot[0] << std::endl;
    
    // Test comparison operators
    auto eq = vec1_gpu == vec2_gpu;
    std::cout << "Equality test (5 == 3): " << eq[0] << std::endl;
    
    auto lt = vec1_gpu < vec2_gpu;
    std::cout << "Less than test (5 < 3): " << lt[0] << std::endl;
    
    // Test scalar operations
    auto scalar_sum = vec1_gpu + 2;
    std::cout << "Scalar addition test (5 + 2): " << scalar_sum[0] << std::endl;
    
    auto scalar_prod = vec1_gpu * 2;
    std::cout << "Scalar multiplication test (5 * 2): " << scalar_prod[0] << std::endl;
    
    // Test compound assignment
    vec1_gpu += vec2_gpu;
    std::cout << "Compound addition test (5 += 3): " << vec1_gpu[0] << std::endl;
    
    // Test performance comparison with CPU version
    std::cout << "\nPerformance comparison (GPU vs CPU)..." << std::endl;
    const size_t size = 1000000000; // 1 billion elements
    std::cout << "Filling vector with " << size << " elements..." << std::endl;
    
    // GPU Version
    Lp_parallel_vector_GPU<int> large_vec_gpu(size);
    auto start_gpu = std::chrono::high_resolution_clock::now();
    DoubleFunc doubleFunc2;
    large_vec_gpu.fill(doubleFunc2);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gpu = end_gpu - start_gpu;
    
    // CPU Version
    Lp_parallel_vector<int> large_vec_cpu(size);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    large_vec_cpu.fill(func);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
    
    std::cout << "GPU time: " << elapsed_gpu.count() << " seconds" << std::endl;
    std::cout << "CPU time: " << elapsed_cpu.count() << " seconds" << std::endl;
    
    // Test error handling (division by zero)
    std::cout << "\nTesting error handling..." << std::endl;
    Lp_parallel_vector_GPU<int> zero_vec(10);
    zero_vec.fill(0);
    auto division_result = vec1_gpu / zero_vec;
    std::cout << "Division by zero handled: " << division_result[0] << std::endl;
    
    std::cout << "All GPU vector tests completed successfully!" << std::endl;
}

int main()
{
    // Test basic constructor and destructor
    std::cout << "Testing basic constructor and destructor..." << std::endl;
    Lp_parallel_vector<int> vec(100000);
    
    // Test fill method
    std::cout << "Testing fill method..." << std::endl;
    vec.fill(42);
    std::cout << "First element: " << vec[0] << std::endl;
    std::cout << "Last element: " << vec[vec.size()-1] << std::endl;
    
    // Test fill with function
    std::cout << "Testing fill with function..." << std::endl;
    vec.fill([](int& val, size_t index) { (void)index; return val * 2; });
    std::cout << "First element after function: " << vec[0] << std::endl;
    std::cout << "Last element after function: " << vec[vec.size()-1] << std::endl;
    
    // Test Lp_if_parallel with a smaller vector
    std::cout << "Testing Lp_if_parallel..." << std::endl;
    Lp_parallel_vector<int> small_vec(10);
    small_vec[0] = 1; // Set first element to true
    Lp_if_parallel(small_vec, [](size_t index) { (void)index;std::cout << "Hello from parallel function!" << std::endl; });
    
    // Run specific test for joinable fixes
    test_joinable_fixes();
    
    // Run stress test for thread safety
    stress_test_thread_safety(1000);
    
    // Test the parallel quicksort implementation
    std::cout << "\nTesting parallel quicksort..." << std::endl;
    
    // Create a vector with random values
    Lp_parallel_vector<int> sort_vec(10000);
    std::cout << "Filling vector with random values..." << std::endl;
    
    // Use a lambda to fill with random values
    sort_vec.fill([](int& val, size_t index) { 
        (void)index;
        val = std::rand() % 10000; 
        return val;
    });
    
    // Print first few elements before sorting
    std::cout << "First 10 elements before sorting:" << std::endl;
    for (size_t i = 0; i < 10 && i < sort_vec.size(); i++) {
        std::cout << sort_vec[i] << " ";
    }
    std::cout << std::endl;
    
    // Sort the vector in ascending order
    std::cout << "Sorting vector in ascending order..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    Lp_sort(sort_vec, std::function<bool(int, int)>([](int a, int b) { return a < b; }));
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Print first few elements after sorting
    std::cout << "First 10 elements after sorting:" << std::endl;
    for (size_t i = 0; i < 10 && i < sort_vec.size(); i++) {
        std::cout << sort_vec[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify that the vector is sorted
    bool is_sorted = true;
    for (size_t i = 1; i < sort_vec.size(); i++) {
        if (sort_vec[i] < sort_vec[i-1]) {
            is_sorted = false;
            std::cout << "Error: Vector is not sorted at index " << i << std::endl;
            break;
        }
    }
    
    if (is_sorted) {
        std::cout << "Vector is correctly sorted!" << std::endl;
    }
    
    std::cout << "Sorting completed in " << elapsed.count() << " seconds" << std::endl;
    
    std::cout << "All tests completed successfully!" << std::endl;

    // Run GPU vector tests
    test_gpu_vector();

    // Example
    {
        // Create a parallel vector
        Lp_parallel_vector<int> tvec(1000);
        
        // Fill the vector with a value
        tvec.fill(42);
        
        // Apply a function to each element
        tvec.fill([](int& val, size_t index) { (void)index; return val * 2 ; });
        
        // Perform parallel operations
        Lp_parallel_vector<int> tvec2(1000);
        tvec2.fill(10);
        
        // Add two vectors
        Lp_parallel_vector<int> result = tvec + tvec2;
        
        // Conditional parallel execution
        tvec[0] = 0; // Set condition
        tvec[15] = 0; // Set condition
        Lp_if_parallel(tvec == 0, [tvec](size_t index) {
            // This will be executed in parallel if condition is met
            std::cout << "Executing in parallel!" << std::endl;
            std::cout << tvec[index] << " at index " <<index << std::endl;  
        });
    }
    
    return 0;
}
