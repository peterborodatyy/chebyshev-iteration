// Basic example: solve a 3x3 symmetric positive-definite system.
package main

import (
	"fmt"
	"log"

	"github.com/peterborodatyy/chebyshev-iteration/chebyshev"
)

func main() {
	// Solve Ax = b where:
	//   A = [[4 1 0], [1 3 1], [0 1 2]]  (SPD, eigenvalues ~ 1.27, 3.0, 4.73)
	//   b = [5, 5, 3]
	//   Exact solution: x = [1, 1, 1]
	s := chebyshev.Solver{
		A: [][]float64{
			{4, 1, 0},
			{1, 3, 1},
			{0, 1, 2},
		},
		B:         []float64{5, 5, 3},
		LambdaMin: 1.268,
		LambdaMax: 4.732,
		Tolerance: 1e-10,
	}

	result, err := s.Solve()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Converged:  %v\n", result.Converged)
	fmt.Printf("Iterations: %d\n", result.Iterations)
	fmt.Printf("Residual:   %e\n", result.Residual)
	fmt.Printf("Solution:   %v\n", result.X)
}
