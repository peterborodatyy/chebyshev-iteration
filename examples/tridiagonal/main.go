// Tridiagonal example: solve a 10x10 tridiagonal system from a discretized ODE.
//
// This arises from discretizing -u''(x) + 0.1*u(x) = f(x) on [0,1] with
// n interior points and step size h = 1/(n+1).
package main

import (
	"fmt"
	"log"
	"math"

	"github.com/peterborodatyy/chebyshev-iteration/chebyshev"
)

func main() {
	n := 10
	h := 1.0 / float64(n+1)
	h2 := h * h
	d := 0.1 // reaction coefficient

	// Build tridiagonal matrix: A[i][i] = 2/h^2 + d, A[i][i+-1] = -1/h^2
	A := make([][]float64, n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		A[i] = make([]float64, n)
		A[i][i] = 2.0/h2 + d
		if i > 0 {
			A[i][i-1] = -1.0 / h2
		}
		if i < n-1 {
			A[i][i+1] = -1.0 / h2
		}
		// RHS: f(x_i) = sin(pi * x_i)
		xi := float64(i+1) * h
		b[i] = math.Sin(math.Pi * xi)
	}

	// Eigenvalue bounds for tridiagonal matrix with these coefficients
	lmin := d + 4*math.Pow(math.Sin(math.Pi*h/2), 2)/h2
	lmax := d + 4*math.Pow(math.Cos(math.Pi*h/2), 2)/h2

	s := chebyshev.Solver{
		A:         A,
		B:         b,
		LambdaMin: lmin,
		LambdaMax: lmax,
		MaxIter:   128,
		Tolerance: 1e-12,
	}

	// Alternating ordering is more numerically stable for ill-conditioned problems
	result, err := s.SolveWithOrdering(chebyshev.Alternating)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Converged:  %v\n", result.Converged)
	fmt.Printf("Iterations: %d\n", result.Iterations)
	fmt.Printf("Residual:   %e\n", result.Residual)
	fmt.Println("Solution:")
	for i, v := range result.X {
		xi := float64(i+1) * h
		fmt.Printf("  x[%2d] (at %.4f) = %12.8f\n", i, xi, v)
	}
}
