package main

import (
	"fmt"
	"log"
	"strconv"
	"time"

	"art"
	"art/internal/dataset"
	"art/internal/progress_bar"
)

const (
	TRAIN_SAMPLES_PER_DIGIT = -1
	TEST_SAMPLES_PER_DIGIT  = -1

	progressBarWidth = 60
)

func main() {
	trainData, err := dataset.GetData("../testdata/mnist_train.csv", TRAIN_SAMPLES_PER_DIGIT, false)
	if err != nil {
		log.Fatal(err)
	}

	testData, err := dataset.GetData("../testdata/mnist_test.csv", TEST_SAMPLES_PER_DIGIT, false)
	if err != nil {
		log.Fatal(err)
	}

	model, err := art.NewFuzzyART(28*28, 0.9, 0.01, 1)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	test(trainData, testData, model.Fit, model.Predict)
	fmt.Printf("Learned categories: %d\n", len(model.W))
}

func test(
	trainData,
	testData map[string][][]float64,
	fitFunc func([]float64) (float64, int),
	predictFunc func([]float64, bool) (float64, int),
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
				_, k := fitFunc(digitData[i])
				if _, ok := category2Digit[k]; !ok {
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
			_, k := predictFunc(sample, false)
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
	fmt.Printf("Total execution time: %s\n", totalTime.Round(time.Second))
}
