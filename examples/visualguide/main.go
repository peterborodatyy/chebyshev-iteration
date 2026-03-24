// Visual guide: demonstrates Chebyshev iteration properties using the Go library.
//
// This replaces the Python notebook with a pure Go program that exercises
// the chebyshev package and prints convergence tables covering:
//
//  1. Chebyshev vs Richardson iteration
//  2. Parameter ordering comparison (Direct / Reverse / Alternating)
//  3. Effect of eigenvalue bound tightness
//  4. Convergence rate vs condition number
package main

import (
	"fmt"
	"math"
	"strings"

	"github.com/peterborodatyy/chebyshev-iteration/chebyshev"
)

func main() {
	section1()
	section2()
	section3()
	section4()
}

// ---------------------------------------------------------------------------
// Helpers (Richardson solver + tridiagonal builder, not in the library)
// ---------------------------------------------------------------------------

// buildTridiag returns a tridiagonal SPD matrix with diagonal d and off-diagonal 1.
func buildTridiag(n int, diag float64) [][]float64 {
	A := make([][]float64, n)
	for i := range A {
		A[i] = make([]float64, n)
		A[i][i] = diag
		if i > 0 {
			A[i][i-1] = 1.0
		}
		if i < n-1 {
			A[i][i+1] = 1.0
		}
	}
	return A
}

// eigenBoundsTridiag returns exact eigenvalue bounds for a tridiagonal SPD matrix
// with constant diagonal d and off-diagonal 1: eigenvalues = d + 2*cos(k*pi/(n+1)).
func eigenBoundsTridiag(n int, diag float64) (lmin, lmax float64) {
	// min eigenvalue at k=n, max at k=1
	lmin = diag + 2*math.Cos(float64(n)*math.Pi/float64(n+1))
	lmax = diag + 2*math.Cos(math.Pi/float64(n+1))
	return
}

// richardsonSolve solves Ax=b with fixed Richardson iteration (optimal constant tau).
func richardsonSolve(A [][]float64, b []float64, lmin, lmax float64, maxIter int, tol float64) (x []float64, residuals []float64) {
	n := len(b)
	x = make([]float64, n)
	tauFixed := 2.0 / (lmin + lmax)

	for iter := 0; iter < maxIter; iter++ {
		r := residualVec(A, b, x)
		rn := vecNorm(r)
		residuals = append(residuals, rn)
		if rn <= tol || math.IsNaN(rn) || math.IsInf(rn, 0) {
			break
		}
		for i := range x {
			x[i] += tauFixed * r[i]
		}
	}
	return
}

// chebyshevSolveTrace solves Ax=b with Chebyshev iteration and records per-iteration residuals.
func chebyshevSolveTrace(A [][]float64, b []float64, lmin, lmax float64, maxIter int, tol float64, ord chebyshev.Ordering) (x []float64, residuals []float64) {
	n := len(b)
	x = make([]float64, n)
	seq := orderingSeq(ord, maxIter)

	for _, k := range seq {
		r := residualVec(A, b, x)
		rn := vecNorm(r)
		residuals = append(residuals, rn)
		if rn <= tol || math.IsNaN(rn) || math.IsInf(rn, 0) {
			break
		}
		tk := tau(k, maxIter, lmin, lmax)
		for i := range x {
			x[i] += tk * r[i]
		}
	}
	return
}

func tau(k, n int, lmin, lmax float64) float64 {
	cosArg := math.Pi * (2*float64(k) - 1) / (2 * float64(n))
	return 2.0 / (lmin + lmax + (lmax-lmin)*math.Cos(cosArg))
}

func orderingSeq(ord chebyshev.Ordering, n int) []int {
	seq := make([]int, n)
	switch ord {
	case chebyshev.Direct:
		for i := range seq {
			seq[i] = i + 1
		}
	case chebyshev.Reverse:
		for i := range seq {
			seq[i] = n - i
		}
	case chebyshev.Alternating:
		lo, hi := 1, n
		for i := range seq {
			if i%2 == 0 {
				seq[i] = lo
				lo++
			} else {
				seq[i] = hi
				hi--
			}
		}
	}
	return seq
}

func residualVec(A [][]float64, b, x []float64) []float64 {
	n := len(b)
	r := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += A[i][j] * x[j]
		}
		r[i] = b[i] - sum
	}
	return r
}

func vecNorm(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

func header(title string) {
	fmt.Println()
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println(title)
	fmt.Println(strings.Repeat("=", 70))
}

// ---------------------------------------------------------------------------
// 1. Chebyshev vs Richardson Iteration
// ---------------------------------------------------------------------------

func section1() {
	header("1. Chebyshev vs Richardson Iteration")
	fmt.Println("20x20 tridiagonal SPD system (diag=4, off-diag=1)")

	n := 20
	A := buildTridiag(n, 4.0)
	b := make([]float64, n)
	for i := range b {
		b[i] = 1.0
	}
	lmin, lmax := eigenBoundsTridiag(n, 4.0)
	fmt.Printf("Eigenvalue range: [%.4f, %.4f], condition number: %.1f\n\n", lmin, lmax, lmax/lmin)

	_, resCheb := chebyshevSolveTrace(A, b, lmin, lmax, 200, 1e-14, chebyshev.Direct)
	_, resRich := richardsonSolve(A, b, lmin, lmax, 200, 1e-14)

	// Print residual comparison at selected iterations
	fmt.Printf("%-10s  %14s  %14s\n", "Iteration", "Chebyshev", "Richardson")
	fmt.Println(strings.Repeat("-", 42))
	milestones := []int{1, 5, 10, 20, 50, 100, 150, 200}
	for _, m := range milestones {
		cheb := "converged"
		rich := "converged"
		if m <= len(resCheb) {
			cheb = fmt.Sprintf("%.4e", resCheb[m-1])
		}
		if m <= len(resRich) {
			rich = fmt.Sprintf("%.4e", resRich[m-1])
		}
		fmt.Printf("%-10d  %14s  %14s\n", m, cheb, rich)
	}
	fmt.Printf("\nChebyshev converged in %d iterations\n", len(resCheb))
	fmt.Printf("Richardson converged in %d iterations\n", len(resRich))

	// Also verify using the library's Solver directly
	s := chebyshev.Solver{A: A, B: b, LambdaMin: lmin, LambdaMax: lmax, MaxIter: 200, Tolerance: 1e-14}
	result, _ := s.Solve()
	fmt.Printf("\n(Library Solver: converged=%v, iterations=%d, residual=%.4e)\n", result.Converged, result.Iterations, result.Residual)
}

// ---------------------------------------------------------------------------
// 2. Parameter Ordering Comparison
// ---------------------------------------------------------------------------

func section2() {
	header("2. Parameter Ordering Comparison")
	fmt.Println("30x30 tridiagonal SPD system, comparing Direct / Reverse / Alternating")

	n := 30
	A := buildTridiag(n, 4.0)
	// Deterministic RHS (matching notebook's seed pattern)
	b := make([]float64, n)
	for i := range b {
		b[i] = math.Sin(float64(i+1) * 0.7)
	}
	lmin, lmax := eigenBoundsTridiag(n, 4.0)

	orderings := []struct {
		name string
		ord  chebyshev.Ordering
	}{
		{"Direct", chebyshev.Direct},
		{"Reverse", chebyshev.Reverse},
		{"Alternating", chebyshev.Alternating},
	}

	fmt.Printf("\n%-12s  %10s  %14s\n", "Ordering", "Iterations", "Final Residual")
	fmt.Println(strings.Repeat("-", 40))

	for _, o := range orderings {
		s := chebyshev.Solver{A: A, B: b, LambdaMin: lmin, LambdaMax: lmax, MaxIter: 512, Tolerance: 1e-14}
		result, _ := s.SolveWithOrdering(o.ord)
		fmt.Printf("%-12s  %10d  %14.4e\n", o.name, result.Iterations, result.Residual)
	}

	// Show per-iteration detail
	fmt.Println("\nPer-iteration residual comparison:")
	fmt.Printf("%-6s", "Iter")
	for _, o := range orderings {
		fmt.Printf("  %14s", o.name)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 52))

	traces := make([][]float64, len(orderings))
	for i, o := range orderings {
		_, traces[i] = chebyshevSolveTrace(A, b, lmin, lmax, 512, 1e-14, o.ord)
	}
	for _, m := range []int{1, 10, 25, 50, 100, 200, 300, 400, 500} {
		fmt.Printf("%-6d", m)
		for _, t := range traces {
			if m <= len(t) {
				fmt.Printf("  %14.4e", t[m-1])
			} else {
				fmt.Printf("  %14s", "converged")
			}
		}
		fmt.Println()
	}
}

// ---------------------------------------------------------------------------
// 3. Effect of Eigenvalue Bound Tightness
// ---------------------------------------------------------------------------

func section3() {
	header("3. Effect of Eigenvalue Bound Tightness")
	fmt.Println("20x20 tridiagonal SPD system, varying eigenvalue bound accuracy")

	n := 20
	A := buildTridiag(n, 4.0)
	b := make([]float64, n)
	for i := range b {
		b[i] = 1.0
	}
	lminExact, lmaxExact := eigenBoundsTridiag(n, 4.0)

	configs := []struct {
		label string
		lmin  float64
		lmax  float64
	}{
		{"Exact bounds", lminExact, lmaxExact},
		{"2x looser", lminExact / 2, lmaxExact * 2},
		{"5x looser", lminExact / 5, lmaxExact * 5},
	}

	fmt.Printf("\n%-16s  %8s  %10s  %14s\n", "Bounds", "kappa", "Iterations", "Final Residual")
	fmt.Println(strings.Repeat("-", 54))

	for _, c := range configs {
		kappa := c.lmax / c.lmin
		s := chebyshev.Solver{A: A, B: b, LambdaMin: c.lmin, LambdaMax: c.lmax, MaxIter: 300, Tolerance: 1e-14}
		result, _ := s.SolveWithOrdering(chebyshev.Alternating)
		fmt.Printf("%-16s  %8.0f  %10d  %14.4e\n", c.label, kappa, result.Iterations, result.Residual)
	}

	// Residual trace
	fmt.Println("\nPer-iteration residual comparison:")
	fmt.Printf("%-6s", "Iter")
	for _, c := range configs {
		fmt.Printf("  %14s", c.label)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 54))

	traces := make([][]float64, len(configs))
	for i, c := range configs {
		_, traces[i] = chebyshevSolveTrace(A, b, c.lmin, c.lmax, 300, 1e-14, chebyshev.Alternating)
	}
	for _, m := range []int{1, 10, 25, 50, 100, 150, 200, 250, 300} {
		fmt.Printf("%-6d", m)
		for _, t := range traces {
			if m <= len(t) {
				fmt.Printf("  %14.4e", t[m-1])
			} else {
				fmt.Printf("  %14s", "converged")
			}
		}
		fmt.Println()
	}
}

// ---------------------------------------------------------------------------
// 4. Convergence Rate vs Condition Number
// ---------------------------------------------------------------------------

func section4() {
	header("4. Convergence Rate vs Condition Number")
	fmt.Println("Theoretical convergence factor rho = (1-sqrt(xi))/(1+sqrt(xi)), xi=lmin/lmax")
	fmt.Println("Iterations needed to reduce error by 1e-10")

	fmt.Printf("%-12s  %12s  %12s\n", "kappa", "rho", "Iterations")
	fmt.Println(strings.Repeat("-", 40))

	kappas := []float64{2, 5, 10, 20, 50, 100, 200, 500, 1000}
	for _, k := range kappas {
		xi := 1.0 / k
		rho := (1 - math.Sqrt(xi)) / (1 + math.Sqrt(xi))
		iters := math.Log(1e-10) / math.Log(rho)
		fmt.Printf("%-12.0f  %12.6f  %12.0f\n", k, rho, iters)
	}

	// Verify a few with actual solves
	fmt.Println("\nVerification with actual solves (20x20, alternating ordering):")
	fmt.Printf("%-12s  %12s  %12s  %14s\n", "kappa (approx)", "Predicted", "Actual", "Residual")
	fmt.Println(strings.Repeat("-", 56))

	// Build matrices with different condition numbers by varying the diagonal
	for _, diag := range []float64{4.0, 6.0, 12.0, 22.0} {
		n := 20
		A := buildTridiag(n, diag)
		b := make([]float64, n)
		for i := range b {
			b[i] = 1.0
		}
		lmin, lmax := eigenBoundsTridiag(n, diag)
		kappa := lmax / lmin

		xi := 1.0 / kappa
		rho := (1 - math.Sqrt(xi)) / (1 + math.Sqrt(xi))
		predicted := math.Log(1e-10) / math.Log(rho)

		s := chebyshev.Solver{A: A, B: b, LambdaMin: lmin, LambdaMax: lmax, MaxIter: 2000, Tolerance: 1e-10}
		result, _ := s.SolveWithOrdering(chebyshev.Alternating)

		fmt.Printf("%-12.1f  %12.0f  %12d  %14.4e\n", kappa, predicted, result.Iterations, result.Residual)
	}
}
