package chebyshev

import (
	"errors"
	"math"
	"strconv"
)

// Solver configures the Chebyshev iterative method for solving Ax = b.
// The matrix A must be symmetric positive-definite (SPD).
// LambdaMin and LambdaMax must bound A's eigenvalues: 0 < LambdaMin <= lambda_i <= LambdaMax.
type Solver struct {
	A         [][]float64 // coefficient matrix (n x n), must be SPD
	B         []float64   // right-hand side vector (length n)
	X0        []float64   // initial guess (optional; zero vector if nil)
	LambdaMin float64     // lower eigenvalue bound (must be > 0)
	LambdaMax float64     // upper eigenvalue bound (must be > LambdaMin)
	MaxIter   int         // maximum iterations (default 512)
	Tolerance float64     // convergence tolerance on ||b - Ax||_2 (default 1e-10)
}

// Result holds the output of a Chebyshev iteration solve.
type Result struct {
	X          []float64 // solution vector
	Iterations int       // number of iterations performed
	Converged  bool      // true if ||b - Ax||_2 <= Tolerance
	Residual   float64   // final residual norm ||b - Ax||_2
}

// Ordering controls the sequence in which Chebyshev parameters are applied.
type Ordering int

const (
	// Direct applies parameters in order k = 1, 2, ..., n.
	Direct Ordering = iota
	// Reverse applies parameters in order k = n, n-1, ..., 1.
	Reverse
	// Alternating applies parameters in order k = 1, n, 2, n-1, ...
	Alternating
)

const (
	defaultMaxIter   = 512
	defaultTolerance = 1e-10
)

// Solve runs the Chebyshev iteration with Direct ordering.
func (s *Solver) Solve() (Result, error) {
	return s.SolveWithOrdering(Direct)
}

// SolveWithOrdering runs the Chebyshev iteration with the given parameter ordering.
func (s *Solver) SolveWithOrdering(ord Ordering) (Result, error) {
	if err := s.validate(); err != nil {
		return Result{}, err
	}

	n := len(s.B)
	maxIter := s.MaxIter
	if maxIter <= 0 {
		maxIter = defaultMaxIter
	}
	tol := s.Tolerance
	if tol <= 0 {
		tol = defaultTolerance
	}

	// Initialize x
	x := make([]float64, n)
	if s.X0 != nil {
		copy(x, s.X0)
	}

	seq := orderingSequence(ord, maxIter)

	var residual float64
	for iter, k := range seq {
		// r = b - A*x
		r := vecsub(s.B, matvec(s.A, x))
		residual = vecnorm(r)

		if residual <= tol {
			return Result{X: x, Iterations: iter, Converged: true, Residual: residual}, nil
		}
		if math.IsNaN(residual) || math.IsInf(residual, 0) {
			return Result{X: x, Iterations: iter, Converged: false, Residual: residual}, nil
		}

		// x = x + tau_k * r
		tk := tau(k, maxIter, s.LambdaMin, s.LambdaMax)
		x = vecadd(x, vecscale(tk, r))
	}

	// Final residual check
	r := vecsub(s.B, matvec(s.A, x))
	residual = vecnorm(r)
	converged := residual <= tol

	return Result{X: x, Iterations: maxIter, Converged: converged, Residual: residual}, nil
}

// tau computes the Chebyshev iteration parameter for step k out of n total.
// tau_k = 2 / (lmin + lmax - (lmax - lmin) * cos(pi*(2k-1)/(2n)))
func tau(k, n int, lambdaMin, lambdaMax float64) float64 {
	cosArg := math.Pi * (2*float64(k) - 1) / (2 * float64(n))
	return 2.0 / (lambdaMin + lambdaMax - (lambdaMax-lambdaMin)*math.Cos(cosArg))
}

// orderingSequence returns the parameter indices for the given ordering.
func orderingSequence(ord Ordering, n int) []int {
	seq := make([]int, n)
	switch ord {
	case Direct:
		for i := 0; i < n; i++ {
			seq[i] = i + 1
		}
	case Reverse:
		for i := 0; i < n; i++ {
			seq[i] = n - i
		}
	case Alternating:
		lo, hi := 1, n
		for i := 0; i < n; i++ {
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

func (s *Solver) validate() error {
	if len(s.A) == 0 {
		return errors.New("chebyshev: matrix A must not be empty")
	}
	n := len(s.A)
	for i, row := range s.A {
		if len(row) != n {
			return errors.New("chebyshev: matrix A must be square (row " + strconv.Itoa(i) + " has length " + strconv.Itoa(len(row)) + ", expected " + strconv.Itoa(n) + ")")
		}
	}
	if len(s.B) == 0 {
		return errors.New("chebyshev: vector B must not be empty")
	}
	if len(s.B) != n {
		return errors.New("chebyshev: dimension mismatch: A is " + strconv.Itoa(n) + "x" + strconv.Itoa(n) + " but B has length " + strconv.Itoa(len(s.B)))
	}
	if s.X0 != nil && len(s.X0) != n {
		return errors.New("chebyshev: X0 has length " + strconv.Itoa(len(s.X0)) + ", expected " + strconv.Itoa(n))
	}
	if s.LambdaMin <= 0 {
		return errors.New("chebyshev: LambdaMin must be positive")
	}
	if s.LambdaMax <= s.LambdaMin {
		return errors.New("chebyshev: LambdaMax must be greater than LambdaMin")
	}
	return nil
}
