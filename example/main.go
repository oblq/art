package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	_ "net/http/pprof"

	"art"
)

const (
	TRAIN_SAMPLES_PER_DIGIT = 400
	TEST_SAMPLES_PER_DIGIT  = 40

	progressBarWidth = 60
)

func main() {
	//go func() {
	//	log.Fatal(http.ListenAndServe("0.0.0.0:5555", nil))
	//}()

	trainData, err := getData("./mnist/mnist_train.csv", TRAIN_SAMPLES_PER_DIGIT, true)
	if err != nil {
		log.Fatal(err)
	}

	testData, err := getData("./mnist/mnist_test.csv", TEST_SAMPLES_PER_DIGIT, true)
	if err != nil {
		log.Fatal(err)
	}

	//model, err := art.NewFuzzyART(28*28, 0.9, 0.00000001, 1)
	//if err != nil {
	//	log.Fatal(err)
	//}
	model := art.NewFuzzyARTMAP(28*28, 0.9, 1e-8, 1.0)
	//defer model.Close()

	test(trainData, testData, model.Fit, model.Predict)
	fmt.Printf("Learned categories: %d\n", len(model.W))
}

func test(
	trainData,
	testData map[string][][]float64,
	fitFunc func([]float64, string),
	predictFunc func([]float64) ([]float64, string),
) {
	startTime := time.Now()

	//category2Digit := make(map[int]int)

	epochs := 1
	totalSamples := 0
	for d := range 10 {
		totalSamples += len(trainData[strconv.Itoa(d)])
	}

	fmt.Println("Training progress:")
	pb := NewProgressBar(epochs*totalSamples, progressBarWidth)

	for e := 0; e < epochs; e++ {
		for d := range 10 {
			digitData := trainData[strconv.Itoa(d)]
			for i := range digitData {
				fitFunc(digitData[i], strconv.Itoa(d))
				//_, k := fitFunc(digitData[i])
				//if prevCategoryDigit, ok := category2Digit[k]; ok {
				//	if prevCategoryDigit != d {
				//		//log.Printf("category %d identify at least two digits: %d and %d\n",
				//		//	k, prevCategoryDigit, d)
				//		//category2Digit[k] = d
				//	}
				//} else {
				//	category2Digit[k] = d
				//}
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
	pbTest := NewProgressBar(samplesCount, progressBarWidth)

	for digit := range 10 {
		samples := testData[strconv.Itoa(digit)]
		for _, sample := range samples {
			_, k := predictFunc(sample)
			if strconv.Itoa(digit) == k {
				exactResults++
			}
			//if digit == category2Digit[k] {
			//	exactResults++
			//}
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
