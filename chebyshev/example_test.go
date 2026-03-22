package chebyshev_test

import (
	"fmt"

	"github.com/peterborodatyy/chebyshev-iteration/chebyshev"
)

func Example() {
	s := chebyshev.Solver{
		A:         [][]float64{{4, 1}, {1, 3}},
		B:         []float64{1, 2},
		LambdaMin: 2.0,
		LambdaMax: 5.0,
		Tolerance: 1e-10,
	}
	result, err := s.Solve()
	if err != nil {
		panic(err)
	}
	fmt.Printf("converged: %v\n", result.Converged)
	fmt.Printf("x[0]: %.4f\n", result.X[0])
	fmt.Printf("x[1]: %.4f\n", result.X[1])
	// Output:
	// converged: true
	// x[0]: 0.0909
	// x[1]: 0.6364
}

func Example_ordering() {
	s := chebyshev.Solver{
		A:         [][]float64{{2, 0, 0}, {0, 3, 0}, {0, 0, 4}},
		B:         []float64{4, 9, 8},
		LambdaMin: 2,
		LambdaMax: 4,
		Tolerance: 1e-10,
	}
	for _, ord := range []chebyshev.Ordering{chebyshev.Direct, chebyshev.Reverse, chebyshev.Alternating} {
		result, _ := s.SolveWithOrdering(ord)
		fmt.Printf("ordering=%d converged=%v\n", ord, result.Converged)
	}
	// Output:
	// ordering=0 converged=true
	// ordering=1 converged=true
	// ordering=2 converged=true
}
