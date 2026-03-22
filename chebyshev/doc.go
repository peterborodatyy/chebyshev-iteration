// Package chebyshev implements the Chebyshev iterative method for solving
// systems of linear equations Ax = b, where A is a symmetric positive-definite
// (SPD) matrix.
//
// The Chebyshev iteration uses optimal relaxation parameters derived from the
// roots of Chebyshev polynomials to accelerate convergence. It requires
// estimates of the smallest and largest eigenvalues of A.
//
// The iteration formula is:
//
//	x_{k+1} = x_k + tau_k * (b - A * x_k)
//
// where tau_k = 2 / (lmin + lmax - (lmax - lmin) * cos(pi*(2k-1)/(2n)))
// and n is the total number of iterations (the Chebyshev polynomial degree).
//
// # Usage
//
//	s := chebyshev.Solver{
//	    A:         [][]float64{{4, 1}, {1, 3}},
//	    B:         []float64{1, 2},
//	    LambdaMin: 2.0,
//	    LambdaMax: 5.0,
//	}
//	result, err := s.Solve()
//
// # Convergence
//
// The method converges when the residual norm ||b - Ax||_2 falls below the
// specified Tolerance. Convergence rate depends on the condition number
// LambdaMax/LambdaMin — tighter eigenvalue bounds yield faster convergence.
//
// Three parameter orderings are available (Direct, Reverse, Alternating)
// which apply the Chebyshev parameters in different sequences. In exact
// arithmetic they converge to the same solution; in floating-point arithmetic
// the ordering can affect numerical stability.
package chebyshev
