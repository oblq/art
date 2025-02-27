package art

import (
	"math"
)

// DefaultARTMAP2 represents the core structure of the Default ARTMAP 2 neural network
type DefaultARTMAP2 struct {
	// Network parameters
	Rho    float64 // p Vigilance parameter
	RhoBar float64 // ρ - baseline vigilance parameter
	Alpha  float64 // α - choice parameter
	Beta   float64 // β - learning rate

	// Network structure
	M             int // M - dimension of input vectors
	NumCategories int // number of output classes
	C             int // C - number of coding nodes

	// Network weights
	Weights [][]float64 // w_j - weight vectors for coding nodes

	// Category mappings
	MapField []int // maps coding nodes to output categories
}

// NewDefaultARTMAP2 creates a new Default ARTMAP 2 network with default parameters
func NewDefaultARTMAP2(inputDim, numCategories int) *DefaultARTMAP2 {
	// Initialize with default parameters as mentioned in the paper
	network := &DefaultARTMAP2{
		RhoBar:        0.0, // Default value for maximal code compression
		Alpha:         0.1, // Default choice parameter
		Beta:          1.0, // Fast learning
		M:             inputDim,
		NumCategories: numCategories,
		C:             0, // Start with no coding nodes
		Weights:       [][]float64{},
		MapField:      []int{},
	}
	return network
}

// ComplementCode performs complement coding on an input vector
func (am *DefaultARTMAP2) ComplementCode(input []float64) []float64 {
	// Complement coding: concatenate input with its complement (1-input)
	coded := make([]float64, am.M*2)
	for i := 0; i < am.M; i++ {
		coded[i] = input[i]
		coded[i+am.M] = 1 - input[i]
	}
	return coded
}

func (am *DefaultARTMAP2) addNewCategory(newWeights []float64, targetClass int) {
	am.C++
	am.Weights = append(am.Weights, newWeights)
	am.MapField = append(am.MapField, targetClass)
}

func (am *DefaultARTMAP2) signalsToCommittedNodes(A []float64) []float64 {
	T := make([]float64, am.C)
	for j := 0; j < len(am.Weights); j++ {
		fuzzyIntersectionSum := 0.0
		wSum := 0.0
		for i := 0; i < am.M; i++ {
			fuzzyIntersectionSum += math.Min(A[i], am.Weights[j][i])
			wSum += am.Weights[j][i]
		}
		T[j] = fuzzyIntersectionSum + (1-am.Alpha)*(float64(am.M)-wSum)
	}
	return T
}

// Fit trains the network on a single input-output pair
func (am *DefaultARTMAP2) Fit(a []float64, targetClass int) {
	A := am.ComplementCode(a)

	if len(am.Weights) == 0 {
		am.addNewCategory(A, targetClass)
		return
	}

	signalsToCommittedNodes := am.signalsToCommittedNodes(A)

	// Winner-take-all coding during training
	// Find the best matching existing category or create a new one
	matchFound := false
	bestMatch := -1
	bestMatchValue := -1.0

	// Calculate activations for all existing coding nodes
	for j := 0; j < am.C; j++ {
		// Choice-by-difference activation function
		activation := am.choiceByDifference(A, am.Weights[j])

		if activation > bestMatchValue {
			// Check vigilance criterion
			match := am.matchesCriterion(A, am.Weights[j])
			if match {
				bestMatchValue = activation
				bestMatch = j
				matchFound = true
			}
		}
	}

	// If no match found or prediction is incorrect, create a new category
	if !matchFound || am.MapField[bestMatch] != targetClass {
		// Create a new coding node
		bestMatch = am.C
		am.C++
		am.Weights = append(am.Weights, A)
		am.MapField = append(am.MapField, targetClass)
	} else {
		// Update weights for the matched node (fast learning)
		for i := 0; i < len(A); i++ {
			am.Weights[bestMatch][i] = am.Beta*math.Min(A[i], am.Weights[bestMatch][i]) +
				(1.0-am.Beta)*am.Weights[bestMatch][i]
		}
	}

	// Distributed next-a test (specific to Default ARTMAP 2)
	// Test if the network would correctly classify the a with distributed activation
	prediction := am.Predict(a)
	if prediction != targetClass {
		// If distributed prediction fails, create a new category
		bestMatch = am.C
		am.C++

		newWeights := make([]float64, len(A))
		copy(newWeights, A)
		am.Weights = append(am.Weights, newWeights)
		am.MapField = append(am.MapField, targetClass)
	}
}

// Predict predicts the class of an input using distributed activation
func (am *DefaultARTMAP2) Predict(input []float64) int {
	if am.C == 0 {
		return -1 // No categories learned yet
	}

	// Complement code the input
	codedInput := am.ComplementCode(input)

	// Calculate distributed activations using increased-gradient CAM rule
	activations := am.distributedActivation(codedInput)

	// Count votes for each category
	categoryVotes := make([]float64, am.NumCategories)
	for j := 0; j < am.C; j++ {
		categoryVotes[am.MapField[j]] += activations[j]
	}

	// Find the category with the highest vote
	maxVote := -1.0
	predictedClass := -1
	for k := 0; k < am.NumCategories; k++ {
		if categoryVotes[k] > maxVote {
			maxVote = categoryVotes[k]
			predictedClass = k
		}
	}

	return predictedClass
}

// Helper functions
func (am *DefaultARTMAP2) choiceByDifference(input, weights []float64) float64 {
	// Implement choice-by-difference activation function
	numerator := 0.0
	for i := 0; i < len(input); i++ {
		numerator += math.Min(input[i], weights[i])
	}

	denominator := am.Alpha + vectorSum(weights)
	return numerator / denominator
}

func (am *DefaultARTMAP2) matchesCriterion(input, weights []float64) bool {
	// Implement vigilance criterion check
	matchScore := 0.0
	for i := 0; i < len(input); i++ {
		matchScore += math.Min(input[i], weights[i])
	}

	inputMagnitude := vectorSum(input)
	return (matchScore / inputMagnitude) >= am.RhoBar
}

func (am *DefaultARTMAP2) distributedActivation(input []float64) []float64 {
	// Simplified implementation of increased-gradient CAM rule
	activations := make([]float64, am.C)

	// Calculate initial activations
	for j := 0; j < am.C; j++ {
		activations[j] = am.choiceByDifference(input, am.Weights[j])
	}

	// Normalize activations
	sum := 0.0
	for j := 0; j < am.C; j++ {
		sum += activations[j]
	}

	if sum > 0 {
		for j := 0; j < am.C; j++ {
			activations[j] /= sum
		}
	}

	return activations
}

func vectorSum(vector []float64) float64 {
	sum := 0.0
	for _, v := range vector {
		sum += v
	}
	return sum
}
