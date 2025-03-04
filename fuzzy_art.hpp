#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <algorithm>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <immintrin.h> // For AVX-512 intrinsics

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    
public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i)
            workers.emplace_back(
                [this] {
                    while(true) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock,
                                [this] { return this->stop || !this->tasks.empty(); });
                            if(this->stop && this->tasks.empty())
                                return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        task();
                    }
                }
            );
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }
};

struct Activation {
    std::vector<float> fi;      // fuzzy intersection
    float fiNorm;               // L1 norm of the fuzzy intersection
    float choice;               // activation value, choice function value
    size_t j;                   // index of the category weights
    
    Activation(size_t size) : fi(size), fiNorm(0), choice(0), j(0) {}
};

class FuzzyART {
private:
    std::unique_ptr<ThreadPool> pool;
    size_t batchSize;
    
    // Vigilance parameter - controls category granularity
    float rho;
    
    // Choice parameter - influences category competition
    float alpha;
    
    // Learning rate - controls weight update speed
    float beta;
    
    // Weight matrix - stores category prototypes
    std::vector<std::vector<float>> W;
    
    // Number of threads to use for parallel processing
    size_t numThreads;
    
    // Is AVX-512 available
    bool hasAVX512;
    
    // Dimension of input vectors
    size_t inputDim;

    // Activation object pool
    std::vector<std::shared_ptr<Activation>> activationPool;
    std::mutex poolMutex;

    // Complement code the input vector
    std::vector<float> complementCode(const std::vector<float>& a);
    
    // Compute fuzzy intersection using AVX-512 if available
    void fuzzyIntersection(const std::vector<float>& A, const std::vector<float>& W, 
                          std::vector<float>& fuzzyIntersection);
    
    // Compute fuzzy intersection using AVX-512
    void fuzzyIntersectionAVX512(const std::vector<float>& A, const std::vector<float>& W, 
                                std::vector<float>& fuzzyIntersection);
    
    // Compute L1 norm using AVX-512 if available
    float l1Norm(const std::vector<float>& arr);
    
    // Compute L1 norm using AVX-512
    float l1NormAVX512(const std::vector<float>& arr);
    
    // Calculate category choice
    std::pair<float, float> categoryChoice(const std::vector<float>& A, 
                                          const std::vector<float>& W, 
                                          std::vector<float>& fuzzyIntersection);
    
    // Activate categories
    std::vector<std::shared_ptr<Activation>> activateCategories(const std::vector<float>& A);
    
    // Compute match criterion
    float matchCriterion(float fiNorm, float aNorm);
    
    // Resonate or reset logic
    std::pair<std::vector<float>, size_t> resonateOrReset(
        const std::vector<float>& A,
        const std::vector<std::shared_ptr<Activation>>& T);
    
    // Get an activation object from the pool
    std::shared_ptr<Activation> getActivation();
    
    // Return activation to the pool
    void returnActivations(std::vector<std::shared_ptr<Activation>>& T);
    
    // Check if the CPU supports AVX-512
    bool checkAVX512Support();

public:
    FuzzyART(size_t inputLen, float rho, float alpha, float beta);
    ~FuzzyART();
    
    // Train the model with a single input vector
    std::pair<std::vector<float>, size_t> fit(const std::vector<float>& a);
    
    // Predict category for input vector with optional learning
    std::pair<std::vector<float>, size_t> predict(const std::vector<float>& a, bool learn = false);

    // Get number of categories
    size_t getNumCategories() const { return W.size(); }

    // Get input dimension
    size_t getInputDim() const { return inputDim; }
};