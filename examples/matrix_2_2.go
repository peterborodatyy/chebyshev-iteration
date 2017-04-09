package main

import (
	"math"
	"chebyshev-iteration/chebyshev_iteration"
	"fmt"
)

func main() {
	n := 3
	eps := 0.1e-2
	gamma1 := 2 - math.Sqrt(2)
	gamma2 := 2 + math.Sqrt(2)
	b := []float64{2, 1, 1}
	a := []float64{1, 0, 1, 0, 2, 0, 1, 0, 3}

	chebyshev_iteration.Init(n, eps,512,a,b,gamma1,gamma2)

	result := chebyshev_iteration.ChebyshevIteration()

	fmt.Println("x := ", result.X, " iterations := ", result.Iterations)
}