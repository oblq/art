#include "fuzzy_art_wrapper.h"
#include "fuzzy_art.hpp"
#include <vector>
#include <cstring>
#include <stdexcept>

extern "C" {

FuzzyARTHandle FuzzyART_New(int inputLen, float rho, float alpha, float beta) {
    try {
        FuzzyART* model = new FuzzyART(inputLen, rho, alpha, beta);
        return static_cast<FuzzyARTHandle>(model);
    } catch (const std::exception& e) {
        // Handle any exceptions from the constructor
        return nullptr;
    }
}

void FuzzyART_Free(FuzzyARTHandle handle) {
    if (handle) {
        FuzzyART* model = static_cast<FuzzyART*>(handle);
        delete model;
    }
}

int FuzzyART_Fit(FuzzyARTHandle handle, const float* input, int inputLen, float* weights, int weightsLen) {
    if (!handle || !input || !weights) {
        return -1;
    }
    
    FuzzyART* model = static_cast<FuzzyART*>(handle);
    
    try {
        // Convert C array to C++ vector
        std::vector<float> inputVec(input, input + inputLen);
        
        // Call the fit method
        auto [weightVec, categoryIdx] = model->fit(inputVec);
        
        // Copy the weights to the output array if there's enough space
        if (weightsLen >= static_cast<int>(weightVec.size())) {
            std::memcpy(weights, weightVec.data(), weightVec.size() * sizeof(float));
        }
        
        return static_cast<int>(categoryIdx);
    } catch (const std::exception& e) {
        return -1;
    }
}

int FuzzyART_Predict(FuzzyARTHandle handle, const float* input, int inputLen, float* weights, int weightsLen, int learn) {
    if (!handle || !input || !weights) {
        return -1;
    }
    
    FuzzyART* model = static_cast<FuzzyART*>(handle);
    
    try {
        // Convert C array to C++ vector
        std::vector<float> inputVec(input, input + inputLen);
        
        // Call the predict method
        auto [weightVec, categoryIdx] = model->predict(inputVec, learn != 0);
        
        // Copy the weights to the output array if there's enough space
        if (weightsLen >= static_cast<int>(weightVec.size())) {
            std::memcpy(weights, weightVec.data(), weightVec.size() * sizeof(float));
        }
        
        return static_cast<int>(categoryIdx);
    } catch (const std::exception& e) {
        return -1;
    }
}

int FuzzyART_GetNumCategories(FuzzyARTHandle handle) {
    if (!handle) {
        return -1;
    }
    
    FuzzyART* model = static_cast<FuzzyART*>(handle);
    
    // Return the number of categories
    return model->getNumCategories();
}

int FuzzyART_GetInputDim(FuzzyARTHandle handle) {
    if (!handle) {
        return -1;
    }
    
    FuzzyART* model = static_cast<FuzzyART*>(handle);
    
    // Return the input dimension
    return model->getInputDim();
}

}