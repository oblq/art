package art

import (
	"math"
	"sort"
)

type DefaultARTMAP2 struct {
	// Network parameters
	Rho          float64 // ρ Vigilance parameter
	RhoBar       float64 // ρ - baseline vigilance parameter
	Alpha        float64 // α - signal rule parameter, (0,1), α = 0 + maximizes code compression
	Beta         float64 // β - learning rate, [0,1], β = 1 implements fast learning
	Epsilon      float64 // ε - epsilon, match tracking, (-1,1), ε < 0 (MT–) codes inconsistent cases
	CAMRulePower float64 // Increased Gradient (IG) CAM rule converges to WTA as p→∞

	// Network structure
	M             int // M - dimension of input vectors
	NumCategories int // number of output classes

	// Network weights
	Weights          [][]float64 // w_j - weight vectors for coding nodes
	OutputCategories [][]float64

	// Category mappings
	MapField []int // maps coding nodes to output categories
}

// NewDefaultARTMAP2 creates a new Default ARTMAP 2 network with default parameters
func NewDefaultARTMAP2(inputDim, numCategories int) *DefaultARTMAP2 {
	// Initialize with default parameters as mentioned in the paper
	network := &DefaultARTMAP2{
		RhoBar:           0.0,  // Default value for maximal code compression
		Alpha:            0.01, // Default choice parameter
		Beta:             1.0,  // Fast learning
		Epsilon:          -0.001,
		CAMRulePower:     1.0,
		M:                inputDim,
		NumCategories:    numCategories,
		C:                0, // Start with no coding nodes
		Weights:          make([][]float64, 0),
		OutputCategories: make([][]float64, 0),
		MapField:         []int{},
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
	am.Weights = append(am.Weights, newWeights)
	am.MapField = append(am.MapField, targetClass)
}

type Activation struct {
	j                    int
	val                  float64
	fuzzyIntersection    []float64
	fuzzyIntersectionSum float64
}

// B.7 and B.8:
// Calculate signals to committed coding nodes
// Sort the committed coding nodes with
// T j > αM in descending order.
func (am *DefaultARTMAP2) signalsToCommittedNodes(A []float64) []Activation {
	T := make([]Activation, 0)
	activationThreshold := am.Alpha * float64(am.M)

	for j := 0; j < len(am.Weights); j++ {
		fuzzyIntersection := make([]float64, len(A))
		fuzzyIntersectionSum := 0.0
		wSum := 0.0
		for i := 0; i < am.M; i++ {
			fuzzyIntersection[i] = math.Min(A[i], am.Weights[j][i])
			fuzzyIntersectionSum += fuzzyIntersection[i]
			wSum += am.Weights[j][i]
		}

		activationVal := fuzzyIntersectionSum + (1-am.Alpha)*(float64(am.M)-wSum)
		if activationVal < activationThreshold {
			continue
		}

		T = append(T, Activation{
			j:                    j,
			val:                  activationVal,
			fuzzyIntersection:    fuzzyIntersection,
			fuzzyIntersectionSum: fuzzyIntersectionSum,
		})
	}

	// Sort category val values indices by val values in descending order
	sort.SliceStable(T, func(a, b int) bool {
		// In case of equal val values, sort by category index,
		// because older categories must have the priority.
		if T[a].val == T[b].val {
			return T[a].j < T[b].j
		}
		return T[a].val > T[b].val
	})

	return T
}

func (am *DefaultARTMAP2) checkCategoryMatch(j int, k int) bool {
	return am.MapField[j] == k
}

// B.11. Learning: Update coding weights
func (am *DefaultARTMAP2) updateWeights(A []float64, t Activation) {
	for i := 0; i < 2*am.M; i++ {
		//am.Weights[t.j][i] = am.Beta*math.Min(A[i], am.Weights[t.j][i]) + (1-am.Beta)*am.Weights[t.j][i]
		am.Weights[t.j][i] = am.Beta*t.fuzzyIntersection[i] + (1-am.Beta)*am.Weights[t.j][i]
	}
}

// Fit trains the network on a single input-output pair
func (am *DefaultARTMAP2) Fit(a []float64, k int) {
	A := am.ComplementCode(a)

	if len(am.Weights) == 0 {
		am.addNewCategory(A, k)
		return
	}

	T := am.signalsToCommittedNodes(A)

	// B.9. Search for a coding node J that meets the matching
	// criterion and predicts the correct output class K
	for _, t := range T {
		// B.9.a. Code: For the next sorted coding node that meets the matching criterion
		resonance := t.fuzzyIntersectionSum / float64(am.M)
		if resonance >= am.Rho {
			if am.checkCategoryMatch(t.j, k) {
				am.updateWeights(A, t)
				//match = true
				break
			} else {
				// B.9.d Match tracking: If the active code J fails to
				// predict the correct output class (σK = 0), raise vigilance:
				am.Rho = resonance + am.Epsilon
				continue
			}
		}
	}

	am.addNewCategory(A, k)
	// B.5. Set vigilance ρ to its baseline value
	am.Rho = am.RhoBar

	// ---------------------------------------------------------

	// Winner-take-all coding during training
	// Find the best matching existing category or create a new one
	//matchFound := false
	//bestMatch := -1
	//bestMatchValue := -1.0
	//
	//// Calculate activations for all existing coding nodes
	//for j := 0; j < len(am.Weights); j++ {
	//	// Choice-by-difference val function
	//	activation := am.choiceByDifference(A, am.Weights[j])
	//
	//	if activation > bestMatchValue {
	//		// Check vigilance criterion
	//		match := am.matchesCriterion(A, am.Weights[j])
	//		if match {
	//			bestMatchValue = activation
	//			bestMatch = j
	//			matchFound = true
	//		}
	//	}
	//}
	//
	//// If no match found or prediction is incorrect, create a new category
	//if !matchFound || am.MapField[bestMatch] != k {
	//	// Create a new coding node
	//	bestMatch = len(am.Weights)
	//	am.C++
	//	am.Weights = append(am.Weights, A)
	//	am.MapField = append(am.MapField, k)
	//} else {
	//	// Update weights for the matched node (fast learning)
	//	for i := 0; i < len(A); i++ {
	//		am.Weights[bestMatch][i] = am.Beta*math.Min(A[i], am.Weights[bestMatch][i]) +
	//			(1.0-am.Beta)*am.Weights[bestMatch][i]
	//	}
	//}
	//
	//// Distributed next-a test (specific to Default ARTMAP 2)
	//// Test if the network would correctly classify the a with distributed val
	//prediction := am.Predict(a)
	//if prediction != k {
	//	// If distributed prediction fails, create a new category
	//	bestMatch = len(am.Weights)
	//	am.C++
	//
	//	newWeights := make([]float64, len(A))
	//	copy(newWeights, A)
	//	am.Weights = append(am.Weights, newWeights)
	//	am.MapField = append(am.MapField, k)
	//}
}

// Predict predicts the class of an input using distributed val
func (am *DefaultARTMAP2) Predict(input []float64) int {
	if len(am.Weights) == 0 {
		return -1 // No categories learned yet
	}

	// Complement code the input
	codedInput := am.ComplementCode(input)

	// Calculate distributed activations using increased-gradient CAM rule
	activations := am.distributedActivation(codedInput)

	// Count votes for each category
	categoryVotes := make([]float64, am.NumCategories)
	for j := 0; j < len(am.Weights); j++ {
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

//func (am *DefaultARTMAP2) matchesCriterion(input, weights []float64) bool {
//	// Implement vigilance criterion check
//	matchScore := 0.0
//	for i := 0; i < len(input); i++ {
//		matchScore += math.Min(input[i], weights[i])
//	}
//
//	inputMagnitude := vectorSum(input)
//	return (matchScore / inputMagnitude) >= am.RhoBar
//}

func (am *DefaultARTMAP2) distributedActivation(input []float64) []float64 {
	// Simplified implementation of increased-gradient CAM rule
	activations := make([]float64, len(am.Weights))

	// Calculate initial activations
	for j := 0; j < len(am.Weights); j++ {
		activations[j] = am.choiceByDifference(input, am.Weights[j])
	}

	// Normalize activations
	sum := 0.0
	for j := 0; j < len(am.Weights); j++ {
		sum += activations[j]
	}

	if sum > 0 {
		for j := 0; j < len(am.Weights); j++ {
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
