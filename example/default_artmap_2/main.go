package main

import (
	"fmt"
	"log"
	"strconv"
	"time"

	_ "net/http/pprof"

	"art"
	"art/internal/dataset"
	"art/internal/progress_bar"
)

const (
	TRAIN_SAMPLES_PER_DIGIT = 1000
	TEST_SAMPLES_PER_DIGIT  = 100

	progressBarWidth = 60
)

func main() {
	trainData, err := dataset.GetData("../mnist/mnist_train.csv", TRAIN_SAMPLES_PER_DIGIT, false)
	if err != nil {
		log.Fatal(err)
	}

	testData, err := dataset.GetData("../mnist/mnist_test.csv", TEST_SAMPLES_PER_DIGIT, false)
	if err != nil {
		log.Fatal(err)
	}

	model := art.NewDefaultARTMAP2(28*28, 10) //0.9, 1e-8, 1.0)

	test(trainData, testData, model.Fit, model.Predict)
	fmt.Printf("Learned categories: %d\n", len(model.W))
}

func test(
	trainData,
	testData map[string][][]float64,
	fitFunc func([]float64, int),
	predictFunc func([]float64) int,
) {
	startTime := time.Now()

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
				fitFunc(digitData[i], d)
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
			k := predictFunc(sample)
			if digit == k {
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
