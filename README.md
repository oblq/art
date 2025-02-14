# Fuzzy ART in Go

There is an entire family of neural network models that are based on the principles of adaptive resonance theory (ART) and are used for pattern recognition and prediction, this is an highly optimized implementation of a specific algorithm called **Fuzzy ART** in Go, it leverage parallel processing to speed up training and inference time.

This implementation is several order of magnitude faster than the original implementation in Python, it finish the training on the MNIST dataset in less than 9 minutes on my workstation with 48 threads, while any of the python implementations I found would take more than 4 hours.

### Adaptive Resonance Theory
Adaptive Resonance Theory, or ART, is a cognitive and neural theory of how the brain autonomously learns to categorize, recognize, and predict objects and events in a changing world pioneered by Stephen Grossberg and Gail Carpenter.  
You can read all about it in this paper [here](https://www.semanticscholar.org/paper/Adaptive-Resonance-Theory%3A-How-a-brain-learns-to-a-Grossberg/71bc18bcafe1f4909a97b0b17a522dffe306ee6a?p2df).

### Characteristics of the **_Fuzzy ART_** algorithm for classification
- **_Unsupervised_** efficient learning (a single pass is enough).
- **_Explainable_**
- Stable **_On-line learning_** (incremental), no need to retrain from scratch, can learn incrementally and infer at the same time.
- **_Without catastrophic forgetting_**

### Usage

```bash
make run-example
```
This will download the MNIST dataset in the example folder and train the model, then it will run the inference on the test set and print the accuracy and duration of the training and inference.

Depending on your machine this can take a while, both training and inference are parallelized, on my workstation, with 48 threads, it takes around 1 minute to train and 1 minute to infer on the test set.