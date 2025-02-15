# Fuzzy ART

This is a highly optimized implementation of one of the algorithms that belongs to the ART theory by Stephen Grossberg and Gail Carpenter.

There is an entire family of neural network models that are based on the principles of adaptive resonance theory (ART) and are used for pattern recognition and prediction.

The _Fuzzy ART_ algorithm:
- **_Unsupervised_** efficient learning (a single pass is enough)
- **_Explainable_**
- Stable **_On-line learning_** (incremental), no need to retrain from scratch, can learn incrementally and infer at the same time
- **_Without catastrophic forgetting_**

## Why Go?
Any of the python implementations I found need something like 4h30m to complete a training session on the full MNIST dataset.

_**This implementation completes the training in 16 minutes on a MacBook Pro M1 Pro and less than 9 minutes on a 48-threads Xeon.**_

MacBook Pro M1 Pro: 10 cores, 16 threads, 16GB RAM:

![](./resources/MacbookPro_M1_Pro.png)

## How?

It uses a worker-pool of the size of the available threads (minus 2, to leave some room for other tasks) to parallelize the training process, plus a pool of pre-allocated slices that are being rotated and reused at every iteration.
Probably could be optimized even further, mey be I will do a detailed profiling of the code some day but for now I'm happy with the results.

## Usage

Download the NMINST dataset, this makefile target will download the training and test set in the example folder:
```bash
make get-mnist
```

The MNIST dataset contains 60000 samples, open the `example/main.go` file and edit the following constants to test just a subset of the dataset or leave them to -1 to run the complete dataset.

```
TRAIN_SAMPLES_PER_DIGIT = -1
TEST_SAMPLES_PER_DIGIT = -1
```

Now run the following command: `make run-example`

## Adaptive Resonance Theory

Adaptive Resonance Theory, or ART, is a cognitive and neural theory of how the brain autonomously learns to categorize, recognize, and predict objects and events in a changing world pioneered by Stephen Grossberg and Gail Carpenter.  
You can read all about it in this paper [here](https://www.semanticscholar.org/paper/Adaptive-Resonance-Theory%3A-How-a-brain-learns-to-a-Grossberg/71bc18bcafe1f4909a97b0b17a522dffe306ee6a?p2df).
