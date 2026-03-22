// Command chebyshev solves a system of linear equations Ax = b using the
// Chebyshev iterative method. Matrix A and vector b are read from stdin.
//
// Usage:
//
//	chebyshev -n 3 -lmin 1.268 -lmax 4.732 < input.txt
//
// Input format (stdin): n*n values for A (row-major), then n values for b,
// all whitespace-separated.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/peterborodatyy/chebyshev-iteration/chebyshev"
)

func main() {
	n := flag.Int("n", 0, "matrix dimension (required)")
	lmin := flag.Float64("lmin", 0, "minimum eigenvalue bound (required, must be > 0)")
	lmax := flag.Float64("lmax", 0, "maximum eigenvalue bound (required, must be > lmin)")
	eps := flag.Float64("eps", 1e-10, "convergence tolerance")
	maxIter := flag.Int("maxiter", 512, "maximum iterations")
	ordering := flag.String("ordering", "direct", "parameter ordering: direct, reverse, alternating")
	flag.Parse()

	if *n <= 0 {
		fmt.Fprintln(os.Stderr, "error: -n must be a positive integer")
		flag.Usage()
		os.Exit(1)
	}
	if *lmin <= 0 || *lmax <= *lmin {
		fmt.Fprintln(os.Stderr, "error: must have 0 < lmin < lmax")
		flag.Usage()
		os.Exit(1)
	}

	// Read A (n*n values) then b (n values) from stdin
	A := make([][]float64, *n)
	for i := 0; i < *n; i++ {
		A[i] = make([]float64, *n)
		for j := 0; j < *n; j++ {
			if _, err := fmt.Scan(&A[i][j]); err != nil {
				log.Fatalf("error reading A[%d][%d]: %v", i, j, err)
			}
		}
	}
	b := make([]float64, *n)
	for i := 0; i < *n; i++ {
		if _, err := fmt.Scan(&b[i]); err != nil {
			log.Fatalf("error reading b[%d]: %v", i, err)
		}
	}

	var ord chebyshev.Ordering
	switch *ordering {
	case "direct":
		ord = chebyshev.Direct
	case "reverse":
		ord = chebyshev.Reverse
	case "alternating":
		ord = chebyshev.Alternating
	default:
		fmt.Fprintf(os.Stderr, "error: unknown ordering %q (use direct, reverse, alternating)\n", *ordering)
		os.Exit(1)
	}

	s := chebyshev.Solver{
		A:         A,
		B:         b,
		LambdaMin: *lmin,
		LambdaMax: *lmax,
		MaxIter:   *maxIter,
		Tolerance: *eps,
	}

	result, err := s.SolveWithOrdering(ord)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Converged:  %v\n", result.Converged)
	fmt.Printf("Iterations: %d\n", result.Iterations)
	fmt.Printf("Residual:   %e\n", result.Residual)
	fmt.Printf("Solution:   %v\n", result.X)
}
