#include "fuzzy_art.hpp"
#include <iostream>
#include <numeric>
#include <mutex>
#include <cmath>

// Check if the CPU supports AVX-512
bool FuzzyART::checkAVX512Support() {
    #ifdef __AVX512F__
    return true;
    #else
    return false;
    #endif
}

FuzzyART::FuzzyART(size_t inputLen, float rho, float alpha, float beta) 
    : batchSize(16), rho(rho), alpha(alpha), beta(beta), inputDim(inputLen) {
    
    if (rho < 0 || rho > 1) {
        throw std::invalid_argument("Vigilance parameter (rho) must be between 0 and 1");
    }
    if (alpha <= 0) {
        throw std::invalid_argument("Choice parameter (alpha) must be positive");
    }
    if (beta <= 0 || beta > 1) {
        throw std::invalid_argument("Learning rate (beta) must be between 0 and 1");
    }
    
    // Check for AVX-512 support
    hasAVX512 = checkAVX512Support();
    
    // Initialize thread pool with hardware concurrency
    numThreads = std::thread::hardware_concurrency();
    pool = std::make_unique<ThreadPool>(numThreads);
    
    // Pre-allocate activation objects for the pool
    for (size_t i = 0; i < numThreads * batchSize; ++i) {
        activationPool.push_back(std::make_shared<Activation>(inputLen * 2));
    }
}

FuzzyART::~FuzzyART() {
    // Thread pool cleanup happens in its destructor
}

std::shared_ptr<Activation> FuzzyART::getActivation() {
    std::lock_guard<std::mutex> lock(poolMutex);
    if (activationPool.empty()) {
        // If pool is empty, create a new activation object
        return std::make_shared<Activation>(inputDim * 2);
    }
    
    auto activation = activationPool.back();
    activationPool.pop_back();
    return activation;
}

void FuzzyART::returnActivations(std::vector<std::shared_ptr<Activation>>& T) {
    std::lock_guard<std::mutex> lock(poolMutex);
    for (auto& t : T) {
        activationPool.push_back(t);
    }
    T.clear();
}

std::vector<float> FuzzyART::complementCode(const std::vector<float>& a) {
    std::vector<float> A(a.size() * 2);
    
    if (hasAVX512 && a.size() % 16 == 0) {
        // Use AVX-512 for complement coding if vector size is aligned
        __m512 ones = _mm512_set1_ps(1.0f);
        
        for (size_t i = 0; i < a.size(); i += 16) {
            // Load 16 elements from input vector
            __m512 x = _mm512_loadu_ps(&a[i]);
            
            // Store original values in first half
            _mm512_storeu_ps(&A[i], x);
            
            // Calculate and store complement (1-x) in second half
            __m512 complement = _mm512_sub_ps(ones, x);
            _mm512_storeu_ps(&A[i + a.size()], complement);
        }
    } else {
        // Fallback to scalar implementation
        for (size_t i = 0; i < a.size(); i++) {
            A[i] = a[i];                   // Original value
            A[i + a.size()] = 1.0f - a[i]; // Complement
        }
    }
    
    return A;
}

void FuzzyART::fuzzyIntersectionAVX512(const std::vector<float>& A, 
                                       const std::vector<float>& W, 
                                       std::vector<float>& result) {
    for (size_t i = 0; i < A.size(); i += 16) {
        // Load 16 elements from each vector
        __m512 a_vec = _mm512_loadu_ps(&A[i]);
        __m512 w_vec = _mm512_loadu_ps(&W[i]);
        
        // Compute element-wise minimum
        __m512 min_vec = _mm512_min_ps(a_vec, w_vec);
        
        // Store the result
        _mm512_storeu_ps(&result[i], min_vec);
    }
}

void FuzzyART::fuzzyIntersection(const std::vector<float>& A, 
                                const std::vector<float>& W, 
                                std::vector<float>& result) {
    if (hasAVX512 && A.size() % 16 == 0) {
        fuzzyIntersectionAVX512(A, W, result);
    } else {
        // Fallback to scalar implementation
        for (size_t i = 0; i < A.size(); i++) {
            result[i] = std::min(A[i], W[i]);
        }
    }
}

float FuzzyART::l1NormAVX512(const std::vector<float>& arr) {
    __m512 sum_vec = _mm512_setzero_ps();
    
    // Process 16 elements at a time
    for (size_t i = 0; i < arr.size(); i += 16) {
        __m512 vec = _mm512_loadu_ps(&arr[i]);
        sum_vec = _mm512_add_ps(sum_vec, vec);
    }
    
    // Horizontal sum of all elements in the vector
    return _mm512_reduce_add_ps(sum_vec);
}

float FuzzyART::l1Norm(const std::vector<float>& arr) {
    if (hasAVX512 && arr.size() % 16 == 0) {
        return l1NormAVX512(arr);
    } else {
        // Fallback to scalar implementation
        return std::accumulate(arr.begin(), arr.end(), 0.0f);
    }
}

std::pair<float, float> FuzzyART::categoryChoice(const std::vector<float>& A, 
                                                const std::vector<float>& W, 
                                                std::vector<float>& fuzzyIntersection) {
    fuzzyIntersection(A, W, fuzzyIntersection);
    float fiNorm = l1Norm(fuzzyIntersection);
    float choice = fiNorm / (alpha + l1Norm(W));
    return {choice, fiNorm};
}

std::vector<std::shared_ptr<Activation>> FuzzyART::activateCategories(const std::vector<float>& A) {
    std::vector<std::shared_ptr<Activation>> T(W.size());
    
    if (W.empty()) {
        return T;
    }
    
    // Synchronization for parallel processing
    std::mutex resultsMutex;
    std::condition_variable cv;
    size_t completed = 0;
    
    // Process categories in batches
    for (size_t jStart = 0; jStart < W.size(); jStart += batchSize) {
        size_t jEnd = std::min(jStart + batchSize, W.size());
        
        pool->enqueue([this, &A, &T, &resultsMutex, &cv, &completed, jStart, jEnd]() {
            std::vector<std::shared_ptr<Activation>> batchResults;
            
            for (size_t j = jStart; j < jEnd; ++j) {
                auto activation = getActivation();
                activation->j = j;
                
                // Compute category choice and fuzzy intersection
                auto [choice, fiNorm] = categoryChoice(A, W[j], activation->fi);
                activation->choice = choice;
                activation->fiNorm = fiNorm;
                
                batchResults.push_back(activation);
            }
            
            // Store results in the output vector
            {
                std::lock_guard<std::mutex> lock(resultsMutex);
                for (auto& act : batchResults) {
                    T[act->j] = act;
                }
                completed += batchResults.size();
            }
            
            cv.notify_one();
        });
    }
    
    // Wait for all batches to complete
    {
        std::unique_lock<std::mutex> lock(resultsMutex);
        cv.wait(lock, [&]{ return completed == W.size(); });
    }
    
    // Sort categories by activation values in descending order
    std::stable_sort(T.begin(), T.end(), 
        [](const std::shared_ptr<Activation>& a, const std::shared_ptr<Activation>& b) {
            // In case of equal activation values, sort by category index
            if (a->choice == b->choice) {
                return a->j < b->j;
            }
            return a->choice > b->choice;
        });
    
    return T;
}

float FuzzyART::matchCriterion(float fiNorm, float aNorm) {
    if (fiNorm == 0 && aNorm == 0) {
        return 1.0f;
    } else {
        return fiNorm / aNorm;
    }
}

std::pair<std::vector<float>, size_t> FuzzyART::resonateOrReset(
    const std::vector<float>& A,
    const std::vector<std::shared_ptr<Activation>>& T) {
    
    float aNorm = l1Norm(A);
    
    // Try to find a resonating category
    for (const auto& t : T) {
        if (!t) continue;  // Skip null activations
        
        float resonance = matchCriterion(t->fiNorm, aNorm);
        if (resonance >= rho) {
            // Update weights for the resonating category
            std::vector<float> newW(A.size());
            
            if (hasAVX512 && A.size() % 16 == 0) {
                __m512 beta_vec = _mm512_set1_ps(beta);
                __m512 one_minus_beta = _mm512_set1_ps(1.0f - beta);
                
                for (size_t k = 0; k < A.size(); k += 16) {
                    __m512 fi_vec = _mm512_loadu_ps(&t->fi[k]);
                    __m512 w_vec = _mm512_loadu_ps(&W[t->j][k]);
                    
                    // newW[k] = beta*t->fi[k] + (1-beta)*W[t->j][k]
                    __m512 beta_fi = _mm512_mul_ps(beta_vec, fi_vec);
                    __m512 old_w = _mm512_mul_ps(one_minus_beta, w_vec);
                    __m512 new_w = _mm512_add_ps(beta_fi, old_w);
                    
                    _mm512_storeu_ps(&newW[k], new_w);
                }
            } else {
                // Fallback to scalar implementation
                for (size_t k = 0; k < newW.size(); k++) {
                    newW[k] = beta * t->fi[k] + (1.0f - beta) * W[t->j][k];
                }
            }
            
            W[t->j] = newW;
            return {newW, t->j};
        }
    }
    
    // If no category meets the vigilance criterion, create a new category
    W.push_back(A);
    return {A, W.size() - 1};
}

std::pair<std::vector<float>, size_t> FuzzyART::fit(const std::vector<float>& a) {
    std::vector<float> A = complementCode(a);
    std::vector<std::shared_ptr<Activation>> T = activateCategories(A);
    auto result = resonateOrReset(A, T);
    returnActivations(T);
    return result;
}

std::pair<std::vector<float>, size_t> FuzzyART::predict(const std::vector<float>& a, bool learn) {
    std::vector<float> A = complementCode(a);
    std::vector<std::shared_ptr<Activation>> T = activateCategories(A);
    
    std::pair<std::vector<float>, size_t> result;
    if (!learn && !T.empty() && T[0]) {
        result = {W[T[0]->j], T[0]->j};
    } else {
        result = resonateOrReset(A, T);
    }
    
    returnActivations(T);
    return result;
}