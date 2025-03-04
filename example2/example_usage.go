package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"art" // Import the art package containing the FuzzyART wrapper
	"art/internal/progress_bar"
)

const (
	TRAIN_SAMPLES_PER_DIGIT = 1000
	TEST_SAMPLES_PER_DIGIT  = 100

	progressBarWidth = 60
)

func main() {
	trainData, err := getData("./mnist/mnist_train.csv", TRAIN_SAMPLES_PER_DIGIT, true)
	if err != nil {
		log.Fatal(err)
	}

	testData, err := getData("./mnist/mnist_test.csv", TEST_SAMPLES_PER_DIGIT, true)
	if err != nil {
		log.Fatal(err)
	}

	// Parameters for the FuzzyART model
	inputDim := 16 // Using 16 for AVX-512 alignment
	rho := 0.8     // Vigilance parameter
	alpha := 0.01  // Choice parameter
	beta := 1.0    // Learning rate

	// Create a new FuzzyART model
	art, err := art.NewFuzzyART(inputDim, rho, alpha, beta)
	if err != nil {
		fmt.Printf("Error creating FuzzyART model: %v\n", err)
		return
	}
	defer art.Close() // Ensure C++ resources are freed

	test(trainData, testData, art.Fit, art.Predict)
}

func test(
	trainData,
	testData map[string][][]float64,
	trainFunc func([]float64) ([]float64, int, error),
	inferFunc func([]float64, bool) ([]float64, int, error),
) {
	startTime := time.Now()

	category2Digit := make(map[int]int)

	epochs := 1
	totalSamples := 0
	for d := range 10 {
		totalSamples += len(trainData[strconv.Itoa(d)])
	}

	fmt.Println("Training progress:")
	pb := progress_bar.New(epochs*totalSamples, progressBarWidth)

	for e := 0; e < epochs; e++ {
		for d := range 10 {
			digitData := trainData[strconv.Itoa(d)]
			for i := range digitData {
				_, k, _ := trainFunc(digitData[i])
				if prevCategoryDigit, ok := category2Digit[k]; ok {
					if prevCategoryDigit != d {
						//log.Printf("category %d identify at least two digits: %d and %d\n",
						//	k, prevCategoryDigit, d)
						//category2Digit[k] = d
					}
				} else {
					category2Digit[k] = d
				}
				pb.Increment()
			}
		}
	}

	trainingTime := time.Since(startTime)
	fmt.Printf("\nTraining completed in %s\n", trainingTime.Round(time.Second))

	testStartTime := time.Now()

	samplesCount := 0
	exactResults := 0

	for digit := range 10 {
		samplesCount += len(testData[strconv.Itoa(digit)])
	}

	fmt.Println("Testing progress:")
	pbTest := progress_bar.New(samplesCount, progressBarWidth)

	for digit := range 10 {
		samples := testData[strconv.Itoa(digit)]
		for _, sample := range samples {
			_, k, _ := inferFunc(sample, false)
			if digit == category2Digit[k] {
				exactResults++
			}
			pbTest.Increment()
		}
	}

	testingTime := time.Since(testStartTime)
	fmt.Printf("\nTesting completed in %s\n", testingTime.Round(time.Second))
	precision := float64(exactResults) / float64(samplesCount)
	fmt.Printf("Accuracy: %.1f%%\n", precision*100)

	totalTime := time.Since(startTime)
	fmt.Printf("Total execution time: %s, learned categories: %d\n", totalTime.Round(time.Second), len(category2Digit))
}

func getData(path string, samplesPerDigit int, shuffle bool) (map[string][][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open mnist_test.csv: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	data, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %v", err)
	}

	dataset := make(map[string][][]float64)

	for i := range 10 {
		key := fmt.Sprintf("%d", i)
		dataset[key] = [][]float64{}
	}

	for _, row := range data {
		label := row[0]
		pixels := make([]float64, len(row)-1)
		for j, val := range row[1:] {
			p, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to parse float: %v", err)
			}
			// normalized values
			pixels[j] = p / 255
		}

		currentSamples := dataset[label]
		if samplesPerDigit == -1 || len(currentSamples) < samplesPerDigit {
			dataset[label] = append(currentSamples, pixels)
		} else {
			continue
		}
	}

	// Shuffle samples
	if shuffle {
		for key := range dataset {
			samples := dataset[key]
			for i := range samples {
				j := i + rand.Intn(len(samples)-i)
				samples[i], samples[j] = samples[j], samples[i]
			}
			if samplesPerDigit != -1 && len(samples) > samplesPerDigit {
				dataset[key] = samples[:samplesPerDigit]
			} else {
				dataset[key] = samples
			}
		}
	}

	return dataset, nil
}
