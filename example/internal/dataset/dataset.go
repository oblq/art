package dataset

import (
	"encoding/csv"
	"fmt"
	"math/rand/v2"
	"os"
	"strconv"
)

func GetData(path string, samplesPerDigit int, shuffle bool) (map[string][][]float64, error) {
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
				j := i + rand.IntN(len(samples)-i)
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
