#include "vector_cpu.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <chrono>
#include <random>
#include <numeric>
#include <cmath>
#include <functional>
#include <memory>

// Custom assertion macro that prints more information
#define ASSERT_TRUE(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion failed: " << message << " in " << __FILE__ << " line " << __LINE__ << std::endl; \
            assert(condition); \
        } \
    } while (false)

// Helper function to measure execution time
template<typename F, typename... Args>
double measureExecutionTime(F&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    std::forward<F>(func)(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

// Helper function to generate a random vector
template<typename T>
Lp_parallel_vector<T> generateRandomVector(size_t size, T min_val, T max_val) {
    Lp_parallel_vector<T> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if constexpr(std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < size; i++) {
            vec[i] = dist(gen);
        }
    } else if constexpr(std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (size_t i = 0; i < size; i++) {
            vec[i] = dist(gen);
        }
    } else {
        // Default case for other types
        for (size_t i = 0; i < size; i++) {
            vec[i] = T();
        }
    }
    
    return vec;
}

// Test constructor functionality
void testConstructors() {
    std::cout << "Testing constructors..." << std::endl;
    
    // Test default constructor
    Lp_parallel_vector<int> vec1;
    ASSERT_TRUE(vec1.size() == 0, "Default constructor should create empty vector");
    
    // Test constructor with size
    size_t test_size = 100;
    Lp_parallel_vector<int> vec2(test_size);
    ASSERT_TRUE(vec2.size() == test_size, "Size constructor failed");
    
    // Test copy constructor from another Lp_parallel_vector
    vec2[0] = 42;
    Lp_parallel_vector<int> vec3(vec2);
    ASSERT_TRUE(vec3.size() == vec2.size() && vec3[0] == 42, "Copy constructor from Lp_parallel_vector failed");
    
    // Test copy constructor from std::vector
    std::vector<int> std_vec = {1, 2, 3, 4, 5};
    Lp_parallel_vector<int> vec4(std_vec);
    ASSERT_TRUE(vec4.size() == std_vec.size() && vec4[0] == std_vec[0], "Copy constructor from std::vector failed");
    
    // Test initializer list constructor
    Lp_parallel_vector<int> vec5 = {10, 20, 30, 40, 50};
    ASSERT_TRUE(vec5.size() == 5 && vec5[2] == 30, "Initializer list constructor failed");
    
    std::cout << "Constructor tests passed!" << std::endl;
}

// Test assignment operators
void testAssignmentOperators() {
    std::cout << "Testing assignment operators..." << std::endl;
    
    // Test assignment from another Lp_parallel_vector
    Lp_parallel_vector<int> vec1 = {1, 2, 3};
    Lp_parallel_vector<int> vec2;
    vec2 = vec1;
    ASSERT_TRUE(vec2.size() == vec1.size() && vec2[0] == 1 && vec2[1] == 2 && vec2[2] == 3, 
                "Assignment from Lp_parallel_vector failed");
    
    // Test assignment from std::vector
    std::vector<int> std_vec = {4, 5, 6};
    Lp_parallel_vector<int> vec3;
    vec3 = std_vec;
    ASSERT_TRUE(vec3.size() == std_vec.size() && vec3[0] == 4 && vec3[1] == 5 && vec3[2] == 6, 
                "Assignment from std::vector failed");
    
    // Test assignment from initializer list
    Lp_parallel_vector<int> vec4;
    vec4 = {7, 8, 9};
    ASSERT_TRUE(vec4.size() == 3 && vec4[0] == 7 && vec4[1] == 8 && vec4[2] == 9, 
                "Assignment from initializer list failed");
    
    std::cout << "Assignment operator tests passed!" << std::endl;
}

// Test fill methods
void testFillMethods() {
    std::cout << "Testing fill methods..." << std::endl;
    
    // Test fill with value
    Lp_parallel_vector<int> vec1(100);
    vec1.fill(42);
    bool all_42 = true;
    for (size_t i = 0; i < vec1.size(); i++) {
        if (vec1[i] != 42) {
            all_42 = false;
            break;
        }
    }
    ASSERT_TRUE(all_42, "fill(value) method failed");
    
    // Test fill with value and size
    Lp_parallel_vector<int> vec2;
    vec2.fill(99, 50);
    ASSERT_TRUE(vec2.size() == 50, "fill(value, size) method failed to resize");
    bool all_99 = true;
    for (size_t i = 0; i < vec2.size(); i++) {
        if (vec2[i] != 99) {
            all_99 = false;
            break;
        }
    }
    ASSERT_TRUE(all_99, "fill(value, size) method failed to set values");
    
    // Test fill with function
    Lp_parallel_vector<int> vec3(100);
    vec3.fill([](int& val, size_t idx) { return idx * 2; });
    bool function_fill_correct = true;
    for (size_t i = 0; i < vec3.size(); i++) {
        if (vec3[i] != static_cast<int>(i * 2)) {
            function_fill_correct = false;
            break;
        }
    }
    ASSERT_TRUE(function_fill_correct, "fill(function) method failed");
    
    // Test fill with function and size
    Lp_parallel_vector<int> vec4;
    vec4.fill([](int& val, size_t idx) { return idx * 3; }, 75);
    ASSERT_TRUE(vec4.size() == 75, "fill(function, size) method failed to resize");
    bool function_size_fill_correct = true;
    for (size_t i = 0; i < vec4.size(); i++) {
        if (vec4[i] != static_cast<int>(i * 3)) {
            function_size_fill_correct = false;
            break;
        }
    }
    ASSERT_TRUE(function_size_fill_correct, "fill(function, size) method failed to set values");
    
    std::cout << "Fill method tests passed!" << std::endl;
}

// Test vector-vector binary operators
void testVectorVectorBinaryOperators() {
    std::cout << "Testing vector-vector binary operators..." << std::endl;
    
    Lp_parallel_vector<int> vec1 = {10, 20, 30, 40, 50};
    Lp_parallel_vector<int> vec2 = {5, 10, 15, 20, 25};
    
    // Test addition
    auto add_result = vec1 + vec2;
    ASSERT_TRUE(add_result.size() == 5 && add_result[0] == 15 && add_result[4] == 75, 
                "Vector-vector addition failed");
    
    // Test subtraction
    auto sub_result = vec1 - vec2;
    ASSERT_TRUE(sub_result.size() == 5 && sub_result[0] == 5 && sub_result[4] == 25, 
                "Vector-vector subtraction failed");
    
    // Test multiplication
    auto mul_result = vec1 * vec2;
    ASSERT_TRUE(mul_result.size() == 5 && mul_result[0] == 50 && mul_result[4] == 1250, 
                "Vector-vector multiplication failed");
    
    // Test division
    auto div_result = vec1 / vec2;
    ASSERT_TRUE(div_result.size() == 5 && div_result[0] == 2 && div_result[4] == 2, 
                "Vector-vector division failed");
    
    // Test logical AND
    Lp_parallel_vector<bool> bool_vec1 = {true, false, true, false, true};
    Lp_parallel_vector<bool> bool_vec2 = {true, true, false, false, true};
    auto and_result = bool_vec1 && bool_vec2;
    ASSERT_TRUE(and_result.size() == 5 && and_result[0] == true && and_result[1] == false && 
                and_result[2] == false && and_result[4] == true, 
                "Vector-vector logical AND failed");
    
    // Test logical OR
    auto or_result = bool_vec1 || bool_vec2;
    ASSERT_TRUE(or_result.size() == 5 && or_result[0] == true && or_result[1] == true && 
                or_result[2] == true && or_result[3] == false, 
                "Vector-vector logical OR failed");
    
    // Test bitwise operations
    Lp_parallel_vector<int> bits1 = {0b1010, 0b1100, 0b1111};
    Lp_parallel_vector<int> bits2 = {0b0101, 0b1010, 0b0000};
    
    // Bitwise AND
    auto bitand_result = bits1 & bits2;
    ASSERT_TRUE(bitand_result.size() == 3 && bitand_result[0] == 0b0000 && 
                bitand_result[1] == 0b1000 && bitand_result[2] == 0b0000, 
                "Vector-vector bitwise AND failed");
    
    // Bitwise OR
    auto bitor_result = bits1 | bits2;
    ASSERT_TRUE(bitor_result.size() == 3 && bitor_result[0] == 0b1111 && 
                bitor_result[1] == 0b1110 && bitor_result[2] == 0b1111, 
                "Vector-vector bitwise OR failed");
    
    // Bitwise XOR
    auto bitxor_result = bits1 ^ bits2;
    ASSERT_TRUE(bitxor_result.size() == 3 && bitxor_result[0] == 0b1111 && 
                bitxor_result[1] == 0b0110 && bitxor_result[2] == 0b1111, 
                "Vector-vector bitwise XOR failed");
    
    // Modulus
    Lp_parallel_vector<int> mod1 = {10, 20, 30};
    Lp_parallel_vector<int> mod2 = {3, 7, 4};
    auto mod_result = mod1 % mod2;
    ASSERT_TRUE(mod_result.size() == 3 && mod_result[0] == 1 && 
                mod_result[1] == 6 && mod_result[2] == 2, 
                "Vector-vector modulus failed");
    
    // Left shift
    Lp_parallel_vector<int> shift1 = {1, 2, 4};
    Lp_parallel_vector<int> shift2 = {2, 1, 3};
    auto lshift_result = shift1 << shift2;
    ASSERT_TRUE(lshift_result.size() == 3 && lshift_result[0] == 4 && 
                lshift_result[1] == 4 && lshift_result[2] == 32, 
                "Vector-vector left shift failed");
    
    // Right shift
    auto rshift_result = shift1 >> shift2;
    ASSERT_TRUE(rshift_result.size() == 3 && rshift_result[0] == 0 && 
                rshift_result[1] == 1 && rshift_result[2] == 0, 
                "Vector-vector right shift failed");
    
    std::cout << "Vector-vector binary operator tests passed!" << std::endl;
}

// Test vector-scalar binary operators
void testVectorScalarBinaryOperators() {
    std::cout << "Testing vector-scalar binary operators..." << std::endl;
    
    Lp_parallel_vector<int> vec = {10, 20, 30, 40, 50};
    
    // Test addition
    auto add_result = vec + 5;
    ASSERT_TRUE(add_result.size() == 5 && add_result[0] == 15 && add_result[4] == 55, 
                "Vector-scalar addition failed");
    
    // Test subtraction
    auto sub_result = vec - 5;
    ASSERT_TRUE(sub_result.size() == 5 && sub_result[0] == 5 && sub_result[4] == 45, 
                "Vector-scalar subtraction failed");
    
    // Test multiplication
    auto mul_result = vec * 2;
    ASSERT_TRUE(mul_result.size() == 5 && mul_result[0] == 20 && mul_result[4] == 100, 
                "Vector-scalar multiplication failed");
    
    // Test division
    auto div_result = vec / 2;
    ASSERT_TRUE(div_result.size() == 5 && div_result[0] == 5 && div_result[4] == 25, 
                "Vector-scalar division failed");
    
    // Test modulus
    auto mod_result = vec % 3;
    ASSERT_TRUE(mod_result.size() == 5 && mod_result[0] == 1 && mod_result[1] == 2 && mod_result[2] == 0, 
                "Vector-scalar modulus failed");
    
    // Test bitwise AND
    Lp_parallel_vector<int> bits = {0b1010, 0b1100, 0b1111};
    auto bitand_result = bits & 0b0101;
    ASSERT_TRUE(bitand_result.size() == 3 && bitand_result[0] == 0b0000 && 
                bitand_result[1] == 0b0100 && bitand_result[2] == 0b0101, 
                "Vector-scalar bitwise AND failed");
    
    // Test bitwise OR
    auto bitor_result = bits | 0b0101;
    ASSERT_TRUE(bitor_result.size() == 3 && bitor_result[0] == 0b1111 && 
                bitor_result[1] == 0b1101 && bitor_result[2] == 0b1111, 
                "Vector-scalar bitwise OR failed");
    
    // Test bitwise XOR
    auto bitxor_result = bits ^ 0b0101;
    ASSERT_TRUE(bitxor_result.size() == 3 && bitxor_result[0] == 0b1111 && 
                bitxor_result[1] == 0b1001 && bitxor_result[2] == 0b1010, 
                "Vector-scalar bitwise XOR failed");
    
    // Test left shift
    Lp_parallel_vector<int> shift = {1, 2, 4};
    auto lshift_result = shift << 2;
    ASSERT_TRUE(lshift_result.size() == 3 && lshift_result[0] == 4 && 
                lshift_result[1] == 8 && lshift_result[2] == 16, 
                "Vector-scalar left shift failed");
    
    // Test right shift
    auto rshift_result = shift >> 1;
    ASSERT_TRUE(rshift_result.size() == 3 && rshift_result[0] == 0 && 
                rshift_result[1] == 1 && rshift_result[2] == 2, 
                "Vector-scalar right shift failed");
    
    std::cout << "Vector-scalar binary operator tests passed!" << std::endl;
}

// Test unary operators
void testUnaryOperators() {
    std::cout << "Testing unary operators..." << std::endl;
    
    // Test logical NOT
    Lp_parallel_vector<bool> bool_vec = {true, false, true, false};
    auto not_result = !bool_vec;
    ASSERT_TRUE(not_result.size() == 4 && not_result[0] == false && not_result[1] == true && 
                not_result[2] == false && not_result[3] == true, 
                "Logical NOT failed");
    
    // Test bitwise NOT
    Lp_parallel_vector<int> bits = {0b1010, 0b1100, 0b1111, 0b0000};
    auto bitnot_result = ~bits;
    // Check if the bits are correctly inverted (this depends on the bit width of int)
    ASSERT_TRUE(bitnot_result.size() == 4 && (bitnot_result[0] & 0xF) == 0b0101 && 
                (bitnot_result[1] & 0xF) == 0b0011 && (bitnot_result[3] & 0xF) == 0b1111, 
                "Bitwise NOT failed");
    
    std::cout << "Unary operator tests passed!" << std::endl;
}

// Test comparison operators (vector with vector)
void testVectorVectorComparisonOperators() {
    std::cout << "Testing vector-vector comparison operators..." << std::endl;
    
    Lp_parallel_vector<int> vec1 = {10, 20, 30, 40, 50};
    Lp_parallel_vector<int> vec2 = {10, 20, 35, 40, 50};
    Lp_parallel_vector<int> vec3 = {10, 20, 30, 40, 50};
    
    // Test equality
    auto eq_result = vec1 == vec2;
    ASSERT_TRUE(eq_result.size() == 5 && eq_result[0] == true && eq_result[2] == false, 
                "Vector-vector equality comparison failed");
    
    auto eq_result2 = vec1 == vec3;
    bool all_true = true;
    for (size_t i = 0; i < eq_result2.size(); i++) {
        if (!eq_result2[i]) {
            all_true = false;
            break;
        }
    }
    ASSERT_TRUE(all_true, "Vector-vector equality with identical vectors failed");
    
    // Test inequality
    auto neq_result = vec1 != vec2;
    ASSERT_TRUE(neq_result.size() == 5 && neq_result[0] == false && neq_result[2] == true, 
                "Vector-vector inequality comparison failed");
    
    // Test less than
    auto lt_result = vec1 < vec2;
    ASSERT_TRUE(lt_result.size() == 5 && lt_result[0] == false && lt_result[2] == true, 
                "Vector-vector less than comparison failed");
    
    // Test greater than
    auto gt_result = vec1 > vec2;
    ASSERT_TRUE(gt_result.size() == 5 && gt_result[0] == false && gt_result[2] == false, 
                "Vector-vector greater than comparison failed");
    
    // Test less than or equal
    auto lte_result = vec1 <= vec2;
    ASSERT_TRUE(lte_result.size() == 5 && lte_result[0] == true && lte_result[2] == true, 
                "Vector-vector less than or equal comparison failed");
    
    // Test greater than or equal
    auto gte_result = vec1 >= vec2;
    ASSERT_TRUE(gte_result.size() == 5 && gte_result[0] == true && gte_result[2] == false, 
                "Vector-vector greater than or equal comparison failed");
    
    std::cout << "Vector-vector comparison operator tests passed!" << std::endl;
}

// Test comparison operators (vector with scalar)
void testVectorScalarComparisonOperators() {
    std::cout << "Testing vector-scalar comparison operators..." << std::endl;
    
    Lp_parallel_vector<int> vec = {10, 20, 30, 40, 30};
    
    // Test equality
    auto eq_result = vec == 30;
    ASSERT_TRUE(eq_result.size() == 5 && eq_result[0] == false && eq_result[2] == true && 
                eq_result[3] == false && eq_result[4] == true, 
                "Vector-scalar equality comparison failed");
    
    // Test inequality
    auto neq_result = vec != 30;
    ASSERT_TRUE(neq_result.size() == 5 && neq_result[0] == true && neq_result[2] == false && 
                neq_result[3] == true && neq_result[4] == false, 
                "Vector-scalar inequality comparison failed");
    
    // Test less than
    auto lt_result = vec < 30;
    ASSERT_TRUE(lt_result.size() == 5 && lt_result[0] == true && lt_result[2] == false && 
                lt_result[3] == false && lt_result[4] == false, 
                "Vector-scalar less than comparison failed");
    
    // Test greater than
    auto gt_result = vec > 30;
    ASSERT_TRUE(gt_result.size() == 5 && gt_result[0] == false && gt_result[2] == false && 
                gt_result[3] == true && gt_result[4] == false, 
                "Vector-scalar greater than comparison failed");
    
    // Test less than or equal
    auto lte_result = vec <= 30;
    ASSERT_TRUE(lte_result.size() == 5 && lte_result[0] == true && lte_result[2] == true && 
                lte_result[3] == false && lte_result[4] == true, 
                "Vector-scalar less than or equal comparison failed");
    
    // Test greater than or equal
    auto gte_result = vec >= 30;
    ASSERT_TRUE(gte_result.size() == 5 && gte_result[0] == false && gte_result[2] == true && 
                gte_result[3] == true && gte_result[4] == true, 
                "Vector-scalar greater than or equal comparison failed");
    
    std::cout << "Vector-scalar comparison operator tests passed!" << std::endl;
}

// Test utility functions
void testUtilityFunctions() {
    std::cout << "Testing utility functions..." << std::endl;
    
    // Test Lp_if_parallel
    Lp_parallel_vector<bool> conditions = {true, false, true, false, true};
    Lp_parallel_vector<int> values = {1, 2, 3, 4, 5};
    int sum = 0;
    
    Lp_if_parallel(conditions, [&sum, &values](bool val, size_t idx) -> void {
        sum += values[idx];
    });
    
    ASSERT_TRUE(sum == 9, "Lp_if_parallel failed"); // 1 + 3 + 5 = 9
    
    // Test Lp_if_single_threaded
    sum = 0;
    Lp_if_single_threaded(conditions, [&sum, &values](bool val, size_t idx) {
        sum += values[idx];
    });
    
    ASSERT_TRUE(sum == 9, "Lp_if_single_threaded failed"); // 1 + 3 + 5 = 9
    
    // Test Lp_sort
    Lp_parallel_vector<int> to_sort = {5, 3, 8, 1, 9, 4, 2, 7, 6};
    Lp_sort(to_sort, [](int a, int b) { return a < b; });
    
    bool is_sorted = true;
    for (size_t i = 1; i < to_sort.size(); i++) {
        if (to_sort[i-1] > to_sort[i]) {
            is_sorted = false;
            break;
        }
    }
    ASSERT_TRUE(is_sorted, "Lp_sort failed to sort in ascending order");
    
    // Test Lp_sort with custom comparator (descending)
    Lp_sort(to_sort, [](int a, int b) { return a > b; });
    
    bool is_reverse_sorted = true;
    for (size_t i = 1; i < to_sort.size(); i++) {
        if (to_sort[i-1] < to_sort[i]) {
            is_reverse_sorted = false;
            break;
        }
    }
    ASSERT_TRUE(is_reverse_sorted, "Lp_sort failed to sort in descending order");
    
    std::cout << "Utility function tests passed!" << std::endl;
}

// Test edge cases
void testEdgeCases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Test with empty vectors
    Lp_parallel_vector<int> empty1;
    Lp_parallel_vector<int> empty2;
    
    auto empty_add = empty1 + empty2;
    ASSERT_TRUE(empty_add.size() == 0, "Addition of empty vectors should produce empty vector");
    
    // Test with vectors of different sizes
    Lp_parallel_vector<int> small = {1, 2, 3};
    Lp_parallel_vector<int> large = {4, 5, 6, 7, 8};
    
    auto diff_size_add = small + large;
    ASSERT_TRUE(diff_size_add.size() == 3, "Addition of different sized vectors should use min size");
    ASSERT_TRUE(diff_size_add[0] == 5 && diff_size_add[2] == 9, "Addition with different sized vectors failed");
    
    // Test with very large vectors (performance test)
    size_t large_size = 1000000;
    Lp_parallel_vector<int> large_vec(large_size);
    
    double sequential_time = measureExecutionTime([&large_vec]() {
        for (size_t i = 0; i < large_vec.size(); i++) {
            large_vec[i] = i;
        }
    });
    
    double parallel_time = measureExecutionTime([&large_vec]() {
        large_vec.fill([](int& val, size_t idx) { return idx; });
    });
    
    std::cout << "Sequential fill time: " << sequential_time << " ms" << std::endl;
    std::cout << "Parallel fill time: " << parallel_time << " ms" << std::endl;
    
    bool large_vec_correct = true;
    for (size_t i = 0; i < std::min(static_cast<size_t>(100), large_vec.size()); i++) {
        if (large_vec[i] != static_cast<int>(i)) {
            large_vec_correct = false;
            break;
        }
    }
    ASSERT_TRUE(large_vec_correct, "Large vector parallel fill failed");
    
    std::cout << "Edge case tests passed!" << std::endl;
}

// Test performance comparison between parallel and sequential operations
void testPerformanceComparison() {
    std::cout << "Testing performance comparison between parallel and sequential operations..." << std::endl;
    
    size_t size = 10000000;
    Lp_parallel_vector<int> vec1(size);
    Lp_parallel_vector<int> vec2(size);
    
    // Initialize vectors
    vec1.fill(1);
    vec2.fill(2);
    
    // Measure parallel vector addition
    double parallel_time = measureExecutionTime([&vec1, &vec2]() {
        auto result = vec1 + vec2;
    });
    
    // Measure sequential vector addition
    double sequential_time = measureExecutionTime([&vec1, &vec2]() {
        std::vector<int> result(vec1.size());
        for (size_t i = 0; i < vec1.size(); i++) {
            result[i] = vec1[i] + vec2[i];
        }
    });
    
    std::cout << "Parallel addition time: " << parallel_time << " ms" << std::endl;
    std::cout << "Sequential addition time: " << sequential_time << " ms" << std::endl;
    std::cout << "Speedup: " << sequential_time / parallel_time << "x" << std::endl;
    
    std::cout << "Performance comparison tests passed!" << std::endl;
}

// Test with different data types
void testDifferentDataTypes() {
    std::cout << "Testing with different data types..." << std::endl;
    
    // Test with float
    Lp_parallel_vector<float> float_vec = {1.5f, 2.5f, 3.5f};
    auto float_result = float_vec * 2.0f;
    ASSERT_TRUE(float_result.size() == 3 && 
                std::abs(float_result[0] - 3.0f) < 0.001f && 
                std::abs(float_result[2] - 7.0f) < 0.001f, 
                "Float vector operations failed");
    
    // Test with double
    Lp_parallel_vector<double> double_vec = {1.5, 2.5, 3.5};
    auto double_result = double_vec * 2.0;
    ASSERT_TRUE(double_result.size() == 3 && 
                std::abs(double_result[0] - 3.0) < 0.001 && 
                std::abs(double_result[2] - 7.0) < 0.001, 
                "Double vector operations failed");
    
    // Test with char
    Lp_parallel_vector<char> char_vec = {'a', 'b', 'c'};
    auto char_result = char_vec + 1;
    ASSERT_TRUE(char_result.size() == 3 && 
                char_result[0] == 'b' && 
                char_result[1] == 'c' && 
                char_result[2] == 'd', 
                "Char vector operations failed");
    
    std::cout << "Different data type tests passed!" << std::endl;
}

int main() {
    std::cout << "Starting comprehensive tests for Lp_parallel_vector..." << std::endl;
    
    // Run all tests
    testConstructors();
    testAssignmentOperators();
    testFillMethods();
    testVectorVectorBinaryOperators();
    testVectorScalarBinaryOperators();
    testUnaryOperators();
    testVectorVectorComparisonOperators();
    testVectorScalarComparisonOperators();
    testUtilityFunctions();
    testEdgeCases();
    testPerformanceComparison();
    testDifferentDataTypes();
    
    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}
