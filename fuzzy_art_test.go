package art

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"art/internal/dataset"
)

func loadTestData(filename string, samplesPerDigit int) (map[string][][]float64, error) {
	// Get the current directory
	wd, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("failed to get working directory: %v", err)
	}

	// Construct the path to the test data
	dataPath := filepath.Join(wd, "testdata", filename)

	// Check if the file exists
	if _, err := os.Stat(dataPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("test data file not found: %s - skipping test", dataPath)
	}

	// Load the data
	data, err := dataset.GetData(dataPath, samplesPerDigit, false)
	if err != nil {
		return nil, fmt.Errorf("failed to load test data: %v", err)
	}

	return data, nil
}

func BenchmarkFuzzyART_Fit(b *testing.B) {
	// Load test data
	trainData, err := loadTestData("mnist_train.csv", 100) // Limiting to 100 samples per digit for benchmarking
	if err != nil {
		b.Fatalf("Failed to load test data: %v", err)
	}

	// Test different parameter configurations
	testCases := []struct {
		name  string
		rho   float64
		alpha float64
		beta  float64
	}{
		{"Default", 0.86, 0.01, 1.0},
		{"HighVigilance", 0.95, 0.01, 1.0},
		{"LowVigilance", 0.75, 0.01, 1.0},
		{"HighAlpha", 0.86, 0.1, 1.0},
		{"LowAlpha", 0.86, 0.001, 1.0},
		{"LowBeta", 0.86, 0.01, 0.5},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			// Create a new FuzzyART model
			model, err := NewFuzzyART(28*28, tc.rho, tc.alpha, tc.beta)
			if err != nil {
				b.Fatalf("Failed to create FuzzyART model: %v", err)
			}
			defer model.Close()

			// Reset the timer for the fitting process
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// Reset the model for each iteration
				model, err = NewFuzzyART(28*28, tc.rho, tc.alpha, tc.beta)
				if err != nil {
					b.Fatalf("Failed to create FuzzyART model: %v", err)
				}

				// Train the model on each digit
				for d := range 10 {
					digitData := trainData[strconv.Itoa(d)]
					for i := range digitData {
						_, _ = model.Fit(digitData[i])
					}
				}
			}

			// Report additional metrics
			b.ReportMetric(float64(len(model.W)), "categories")
		})
	}
}

func BenchmarkFuzzyART_Predict(b *testing.B) {
	// Load test data
	trainData, err := loadTestData("mnist_train.csv", 100)
	if err != nil {
		b.Fatalf("Failed to load train data: %v", err)
	}
	testData, err := loadTestData("mnist_test.csv", 100)
	if err != nil {
		b.Fatalf("Failed to load test data: %v", err)
	}

	// Test different parameter configurations
	testCases := []struct {
		name  string
		rho   float64
		alpha float64
		beta  float64
	}{
		{"Default", 0.86, 0.01, 1.0},
		{"HighVigilance", 0.95, 0.01, 1.0},
		{"LowVigilance", 0.75, 0.01, 1.0},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			// Create and train the model
			model, err := NewFuzzyART(28*28, tc.rho, tc.alpha, tc.beta)
			if err != nil {
				b.Fatalf("Failed to create FuzzyART model: %v", err)
			}
			defer model.Close()

			// Build category to digit mapping during training
			category2Digit := make(map[int]int)

			// Train the model (outside the benchmark measurement)
			b.StopTimer()
			for d := range 10 {
				digitData := trainData[strconv.Itoa(d)]
				for i := range digitData {
					_, k := model.Fit(digitData[i])
					if _, ok := category2Digit[k]; !ok {
						category2Digit[k] = d
					}
				}
			}
			b.StartTimer()

			// Flatten the test data for easier benchmarking
			var flatTestData [][]float64
			for d := range 10 {
				flatTestData = append(flatTestData, testData[strconv.Itoa(d)]...)
			}

			// Benchmark prediction
			correct := 0
			total := 0

			for i := 0; i < b.N; i++ {
				// Use modulo to cycle through the test data
				idx := i % len(flatTestData)
				sample := flatTestData[idx]

				// Find the digit for this sample
				var digit int
				for d := range 10 {
					if contains(testData[strconv.Itoa(d)], sample) {
						digit = d
						break
					}
				}

				_, k := model.Predict(sample, false)
				if digit == category2Digit[k] {
					correct++
				}
				total++
			}

			// Report accuracy metrics
			accuracy := float64(correct) / float64(total) * 100
			b.ReportMetric(accuracy, "accuracy%")
			b.ReportMetric(float64(len(model.W)), "categories")
		})
	}
}

func BenchmarkFuzzyART_BatchProcessing(b *testing.B) {
	trainData, err := loadTestData("mnist_train.csv", 100)
	if err != nil {
		b.Fatalf("Failed to load train data: %v", err)
	}

	batchSizes := []int{1, 8, 16, 32, 64, 128, 256}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("BatchSize_%d", batchSize), func(b *testing.B) {
			// Create a new FuzzyART model with the specified batch size
			model, err := NewFuzzyART(28*28, 0.86, 0.01, 1.0)
			if err != nil {
				b.Fatalf("Failed to create FuzzyART model: %v", err)
			}
			defer model.Close()

			// Set the batch size
			model.batchSize = batchSize

			// Flatten the training data
			var flatTrainData [][]float64
			for d := range 10 {
				flatTrainData = append(flatTrainData, trainData[strconv.Itoa(d)]...)
			}

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// Process a complete batch or as many samples as available
				batchEnd := min(batchSize, len(flatTrainData))
				for j := 0; j < batchEnd; j++ {
					_, _ = model.Fit(flatTrainData[j])
				}
			}

			b.ReportMetric(float64(len(model.W)), "categories")
		})
	}
}

// Helper function to check if a slice contains a specific item
func contains(slice [][]float64, item []float64) bool {
	for _, s := range slice {
		// For simplicity, we're checking if the pointers are the same
		// In a real implementation, you might want to compare values
		if &s[0] == &item[0] {
			return true
		}
	}
	return false
}

func TestFuzzyART(t *testing.T) {
	trainData, err := loadTestData("mnist_train.csv", 10) // Limit to 10 samples per digit for faster testing
	if err != nil {
		t.Fatalf("Failed to load train data: %v", err)
	}

	testData, err := loadTestData("mnist_test.csv", 10)
	if err != nil {
		t.Fatalf("Failed to load test data: %v", err)
	}

	model, err := NewFuzzyART(28*28, 0.86, 0.01, 1.0)
	if err != nil {
		t.Fatalf("Failed to create FuzzyART model: %v", err)
	}
	defer model.Close()

	// Build category to digit mapping during training
	category2Digit := make(map[int]int)

	// Train the model
	for d := range 10 {
		digitData := trainData[strconv.Itoa(d)]
		for i := range digitData {
			_, k := model.Fit(digitData[i])
			if _, ok := category2Digit[k]; !ok {
				category2Digit[k] = d
			}
		}
	}

	// Test the model
	correct := 0
	total := 0

	for d := range 10 {
		digitData := testData[strconv.Itoa(d)]
		for i := range digitData {
			_, k := model.Predict(digitData[i], false)
			predictedDigit := category2Digit[k]
			if predictedDigit == d {
				correct++
			}
			total++
		}
	}

	accuracy := float64(correct) / float64(total) * 100
	t.Logf("Accuracy: %.2f%% (%d/%d)", accuracy, correct, total)
	t.Logf("Number of categories created: %d", len(model.W))

	// Verify that the model performs better than random chance (which would be 10%)
	if accuracy < 20 {
		t.Errorf("Model accuracy is too low: %.2f%%", accuracy)
	}
}

// TestParameterSensitivity tests how different parameter settings affect the model's performance
func TestParameterSensitivity(t *testing.T) {
	trainData, err := loadTestData("mnist_train.csv", 10)
	if err != nil {
		t.Fatalf("Failed to load test data: %v", err)
	}

	testData, err := loadTestData("mnist_test.csv", 10)
	if err != nil {
		t.Fatalf("Failed to load test data: %v", err)
	}

	testCases := []struct {
		name  string
		rho   float64
		alpha float64
		beta  float64
	}{
		{"HighVigilance", 0.95, 0.01, 1.0},
		{"MediumVigilance", 0.85, 0.01, 1.0},
		{"LowVigilance", 0.75, 0.01, 1.0},
		{"HighAlpha", 0.85, 0.1, 1.0},
		{"LowAlpha", 0.85, 0.001, 1.0},
		{"MediumBeta", 0.85, 0.01, 0.5},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, err := NewFuzzyART(28*28, tc.rho, tc.alpha, tc.beta)
			if err != nil {
				t.Fatalf("Failed to create FuzzyART model: %v", err)
			}
			defer model.Close()

			// Build category to digit mapping during training
			category2Digit := make(map[int]int)

			// Train the model
			for d := range 10 {
				digitData := trainData[strconv.Itoa(d)]
				for i := range digitData {
					_, k := model.Fit(digitData[i])
					if _, ok := category2Digit[k]; !ok {
						category2Digit[k] = d
					}
				}
			}

			// Test the model
			correct := 0
			total := 0

			for d := range 10 {
				digitData := testData[strconv.Itoa(d)]
				for i := range digitData {
					_, k := model.Predict(digitData[i], false)
					predictedDigit := category2Digit[k]
					if predictedDigit == d {
						correct++
					}
					total++
				}
			}

			accuracy := float64(correct) / float64(total) * 100
			t.Logf("Parameters (rho=%.2f, alpha=%.3f, beta=%.1f) - Accuracy: %.2f%% - Categories: %d",
				tc.rho, tc.alpha, tc.beta, accuracy, len(model.W))
		})
	}
}
