#include "fuzzy_art.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Example usage of FuzzyART class
int main() {
    try {
        // Parameters
        size_t inputDim = 16;  // Use 16 for AVX-512 alignment
        float rho = 0.8f;      // Vigilance parameter
        float alpha = 0.01f;   // Choice parameter
        float beta = 1.0f;     // Learning rate
        
        // Create FuzzyART instance
        FuzzyART art(inputDim, rho, alpha, beta);
        
        // Generate random input data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        // Create some sample vectors
        std::vector<std::vector<float>> samples;
        for (int i = 0; i < 100; i++) {
            std::vector<float> sample(inputDim);
            for (size_t j = 0; j < inputDim; j++) {
                sample[j] = dist(gen);
            }
            samples.push_back(sample);
        }
        
        // Train the model
        std::cout << "Training the model...\n";
        auto startTime = std::chrono::high_resolution_clock::now();
        
        for (const auto& sample : samples) {
            auto [weights, category] = art.fit(sample);
            std::cout << "Sample assigned to category: " << category << std::endl;
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "Training completed in " << duration.count() << " ms\n";
        
        // Test prediction
        std::cout << "\nTesting prediction...\n";
        for (int i = 0; i < 5; i++) {
            auto [weights, category] = art.predict(samples[i], false);
            std::cout << "Sample " << i << " predicted category: " << category << std::endl;
        }
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}