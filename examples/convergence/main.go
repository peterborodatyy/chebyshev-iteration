// Convergence example: compare all three parameter orderings.
package main

import (
	"fmt"
	"log"

	"github.com/peterborodatyy/chebyshev-iteration/chebyshev"
)

func main() {
	s := chebyshev.Solver{
		A: [][]float64{
			{4, 1, 0, 0},
			{1, 4, 1, 0},
			{0, 1, 4, 1},
			{0, 0, 1, 4},
		},
		B:         []float64{1, 2, 3, 4},
		LambdaMin: 2.0,
		LambdaMax: 6.0,
		MaxIter:   512,
		Tolerance: 1e-12,
	}

	orderings := []struct {
		name string
		ord  chebyshev.Ordering
	}{
		{"Direct", chebyshev.Direct},
		{"Reverse", chebyshev.Reverse},
		{"Alternating", chebyshev.Alternating},
	}

	fmt.Printf("%-12s  %10s  %12s  %s\n", "Ordering", "Iterations", "Residual", "Converged")
	fmt.Println("----------------------------------------------------")

	for _, o := range orderings {
		result, err := s.SolveWithOrdering(o.ord)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%-12s  %10d  %12.4e  %v\n", o.name, result.Iterations, result.Residual, result.Converged)
	}
}
