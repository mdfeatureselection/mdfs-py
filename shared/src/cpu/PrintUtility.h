#ifndef PRINT_UTILITY_H
#define PRINT_UTILITY_H

#include <iostream>

// Variadic template function to print anything
template<typename... Args>
void print(const Args&... args) {
    (std::cout << ... << args) << std::endl;
}

// Print a single value
template<typename T>
void printValue(const T& value) {
    std::cout << value << "|";
}

// Print an array
template<typename T>
void printArray(const T* arr, size_t size) {
    std::cout << "[";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0) std::cout << ", ";
        printValue(arr[i]);
    }
    std::cout << "]\n";
}

template<typename T>
void printArray(const T* arr, size_t size,int max) {
    std::cout << "[";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0 && i < max) std::cout << ", ";
        printValue(arr[i]);
    }
    std::cout << "]\n";
}

template <typename T>
void printVector(const std::vector<T>& vec) {
    std::cout << "[";
    for (const T& value : vec) {
        std::cout << value << ",";
    }
    std::cout << "]\n";
}

#endif // PRINT_UTILITY_H
