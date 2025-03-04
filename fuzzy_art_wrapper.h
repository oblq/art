#ifndef FUZZY_ART_WRAPPER_H
#define FUZZY_ART_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque type for the FuzzyART model pointer
typedef void* FuzzyARTHandle;

// Create a new FuzzyART model
FuzzyARTHandle FuzzyART_New(int inputLen, float rho, float alpha, float beta);

// Free the FuzzyART model
void FuzzyART_Free(FuzzyARTHandle handle);

// Fit a single input vector to the model
// Returns the category index
// The weights array must be pre-allocated with at least inputLen*2 elements
int FuzzyART_Fit(FuzzyARTHandle handle, const float* input, int inputLen, float* weights, int weightsLen);

// Predict the category for a single input vector
// If learn is true, the model will be updated
// Returns the category index
// The weights array must be pre-allocated with at least inputLen*2 elements
int FuzzyART_Predict(FuzzyARTHandle handle, const float* input, int inputLen, float* weights, int weightsLen, int learn);

// Get the number of categories in the model
int FuzzyART_GetNumCategories(FuzzyARTHandle handle);

// Get the input dimension (before complement coding)
int FuzzyART_GetInputDim(FuzzyARTHandle handle);

#ifdef __cplusplus
}
#endif

#endif // FUZZY_ART_WRAPPER_H