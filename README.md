# Chebyshev Iteration

[![Tests](https://github.com/peterborodatyy/chebyshev-iteration/actions/workflows/test.yml/badge.svg)](https://github.com/peterborodatyy/chebyshev-iteration/actions/workflows/test.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/peterborodatyy/chebyshev-iteration/chebyshev.svg)](https://pkg.go.dev/github.com/peterborodatyy/chebyshev-iteration/chebyshev)
[![Go Report Card](https://goreportcard.com/badge/github.com/peterborodatyy/chebyshev-iteration)](https://goreportcard.com/report/github.com/peterborodatyy/chebyshev-iteration)
[![Share on X](https://img.shields.io/badge/share-on%20X-black?logo=x)](https://twitter.com/intent/tweet?text=Chebyshev%20Iteration%20%E2%80%93%20a%20zero-dependency%20Go%20library%20for%20solving%20linear%20systems%20Ax%3Db&url=https%3A%2F%2Fgithub.com%2Fpeterborodatyy%2Fchebyshev-iteration)
[![Share on Reddit](https://img.shields.io/badge/share-on%20Reddit-orange?logo=reddit&logoColor=white)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fpeterborodatyy%2Fchebyshev-iteration&title=Chebyshev%20Iteration%20%E2%80%93%20zero-dependency%20Go%20library%20for%20solving%20linear%20systems)

A zero-dependency Go library implementing the [Chebyshev semi-iterative method](https://en.wikipedia.org/wiki/Chebyshev_iteration) for solving systems of linear equations **Ax = b**, where **A** is a symmetric positive-definite (SPD) matrix.

The method uses optimal relaxation parameters derived from the roots of [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) of the first kind to minimize the error polynomial over the eigenvalue spectrum of **A**. Unlike Krylov subspace methods (CG, GMRES), the Chebyshev iteration requires no inner products -- each step is a single matrix-vector multiply plus vector updates -- making it naturally suited for parallel and distributed computation. The trade-off is that it requires estimates of the extreme eigenvalues of **A**.

## Installation

```bash
go get github.com/peterborodatyy/chebyshev-iteration/chebyshev
```

Requires Go 1.21+. Zero external dependencies.

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    "github.com/peterborodatyy/chebyshev-iteration/chebyshev"
)

func main() {
    s := chebyshev.Solver{
        A: [][]float64{
            {4, 1, 0},
            {1, 3, 1},
            {0, 1, 2},
        },
        B:         []float64{5, 5, 3},
        LambdaMin: 1.268,  // lower eigenvalue bound
        LambdaMax: 4.732,  // upper eigenvalue bound
        Tolerance: 1e-10,
    }

    result, err := s.Solve()
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Converged: %v in %d iterations\n", result.Converged, result.Iterations)
    fmt.Printf("Solution:  %v\n", result.X)
    // Output: x ~ [1, 1, 1]
}
```

## API

### Solver

```go
type Solver struct {
    A         [][]float64  // Coefficient matrix (n x n), must be SPD
    B         []float64    // Right-hand side vector (length n)
    X0        []float64    // Initial guess (optional; zero vector if nil)
    LambdaMin float64      // Lower eigenvalue bound (must be > 0)
    LambdaMax float64      // Upper eigenvalue bound (must be > LambdaMin)
    MaxIter   int          // Maximum iterations (default: 512)
    Tolerance float64      // Convergence tolerance on ||b - Ax||_2 (default: 1e-10)
}
```

**Methods:**

- `Solve() (Result, error)` -- solve using Direct parameter ordering
- `SolveWithOrdering(ord Ordering) (Result, error)` -- solve with a specific ordering

`Solve()` does not mutate the `Solver`, so it is safe to call concurrently from multiple goroutines.

### Result

```go
type Result struct {
    X          []float64  // Solution vector (best found, even if not converged)
    Iterations int        // Number of iterations performed
    Converged  bool       // Whether ||b - Ax||_2 <= Tolerance
    Residual   float64    // Final residual norm ||b - Ax||_2
}
```

### Ordering

Three parameter orderings control the sequence in which Chebyshev relaxation parameters are applied:

| Ordering | Sequence | Notes |
|---|---|---|
| `Direct` | k = 1, 2, ..., n | Standard choice |
| `Reverse` | k = n, n-1, ..., 1 | |
| `Alternating` | k = 1, n, 2, n-1, ... | Often most numerically stable |

In exact arithmetic, all orderings converge to the same solution. In floating-point arithmetic, the ordering affects numerical stability -- `Alternating` tends to be the most robust for ill-conditioned systems.

## Examples

### Basic (3x3 SPD system)

```bash
go run ./examples/basic/
```

Solves a simple 3x3 system with known exact solution x = [1, 1, 1]. See [`examples/basic/main.go`](examples/basic/main.go).

### Tridiagonal (discretized ODE)

```bash
go run ./examples/tridiagonal/
```

Solves a 10x10 tridiagonal system arising from discretizing -u''(x) + 0.1*u(x) = sin(pi*x) on [0,1]. Demonstrates eigenvalue bound computation for structured matrices. See [`examples/tridiagonal/main.go`](examples/tridiagonal/main.go).

### Convergence comparison

```bash
go run ./examples/convergence/
```

Compares all three parameter orderings on a 4x4 tridiagonal system, showing how ordering affects iteration count. See [`examples/convergence/main.go`](examples/convergence/main.go).

### CLI tool

```bash
go build -o chebyshev ./cmd/chebyshev/
echo "4 1 0 1 3 1 0 1 2 5 5 3" | ./chebyshev -n 3 -lmin 1.268 -lmax 4.732
```

Reads matrix A (row-major) and vector b from stdin. Flags: `-n`, `-lmin`, `-lmax`, `-eps`, `-maxiter`, `-ordering`.

### Jupyter notebook

An interactive [Python notebook](notebooks/chebyshev_iteration.ipynb) with visualizations of the method:
- Chebyshev vs Richardson convergence comparison
- Parameter ordering effects on numerical stability
- Eigenvalue bound tightness impact
- Chebyshev polynomial roots on the eigenvalue spectrum
- Error polynomial minimax optimality
- Convergence rate vs condition number

## Mathematical Background

### The iteration

The Chebyshev semi-iterative method solves **Ax = b** via the [Richardson iteration](https://en.wikipedia.org/wiki/Modified_Richardson_iteration) with optimally chosen relaxation parameters:

```
x_{k+1} = x_k + tau_k * (b - A * x_k)
```

The key insight is that the relaxation parameters tau_k are chosen as the reciprocals of the roots of the degree-n [Chebyshev polynomial of the first kind](https://en.wikipedia.org/wiki/Chebyshev_polynomials), shifted from [-1, 1] to the eigenvalue interval [lambda_min, lambda_max]:

```
d_k = (lambda_min + lambda_max)/2 + (lambda_max - lambda_min)/2 * cos(pi * (2k - 1) / (2n))

tau_k = 1/d_k = 2 / (lambda_min + lambda_max + (lambda_max - lambda_min) * cos(pi * (2k - 1) / (2n)))
```

Here **n** is the polynomial degree (equal to `MaxIter`) and **k** ranges from 1 to n. This choice of parameters minimizes the maximum value of the error polynomial over [lambda_min, lambda_max], leveraging the [minimax property](https://en.wikipedia.org/wiki/Chebyshev_polynomials#Properties) of Chebyshev polynomials (see Golub & Van Loan, Section 10.1.5).

### Convergence

The convergence factor after n iterations is bounded by:

```
||e_n|| / ||e_0|| <= 2 * rho^n / (1 + rho^(2n))
```

where `rho = (1 - sqrt(xi)) / (1 + sqrt(xi))` and `xi = lambda_min / lambda_max` (the reciprocal of the [spectral condition number](https://en.wikipedia.org/wiki/Condition_number#Matrices)). This bound is optimal among all polynomial iteration methods of the same degree (Varga, *Matrix Iterative Analysis*, Chapter 5).

Tighter eigenvalue bounds (closer to the true extreme eigenvalues) yield faster convergence. Loose bounds still guarantee convergence but require more iterations.

### Eigenvalue bounds

The method requires estimates satisfying `0 < lambda_min <= lambda_i <= lambda_max` for all eigenvalues lambda_i of A. Common approaches:

- **[Gershgorin circles](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem):** For each row i, lambda lies in [a_ii - R_i, a_ii + R_i] where R_i = sum of |a_ij| for j != i. Take min of lower bounds and max of upper bounds. Easy to compute but often loose.
- **Known structure:** For tridiagonal matrices from finite differences, eigenvalue formulas exist in closed form. For example, the standard 1D Laplacian discretization on n points with step h has eigenvalues `(4/h^2) * sin^2(k*pi*h/2)` for k = 1, ..., n.
- **Power iteration / [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm):** Compute a few extreme eigenvalues numerically. A few Lanczos steps can produce tight bounds cheaply.
- **A priori bounds:** For specific PDE discretizations, eigenvalue bounds can be derived from the continuous operator's spectrum.

### Numerical stability

The Chebyshev iteration can be numerically unstable when the polynomial degree (MaxIter) is large relative to the condition number. This is because the tau parameters near the edges of the Chebyshev spectrum can be very large or very small, amplifying rounding errors (see Gutknecht & Rollin, Section 3.2).

The `Alternating` ordering mitigates this by interleaving parameters from both ends of the spectrum. For well-conditioned systems (condition number < 10), any ordering works. For moderate condition numbers (10-100), prefer `Alternating`. For poorly conditioned systems, consider using a preconditioner or a different method (e.g., conjugate gradient).

### Comparison with other methods

| Method | Inner products | Eigenvalue bounds | SPD only | Parallelizability |
|--------|---------------|-------------------|----------|-------------------|
| **Chebyshev iteration** | No | Required | Yes | Excellent |
| [Conjugate Gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method) | Yes (2 per step) | Not required | Yes | Good (but global reductions) |
| [GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method) | Yes | Not required | No | Good (but global reductions) |
| [Jacobi / Gauss-Seidel](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) | No | Not required | No | Limited (GS is sequential) |

The absence of inner products (global reductions) makes the Chebyshev iteration particularly attractive for massively parallel architectures where synchronization is expensive.

## When to Use

The Chebyshev iteration is a good choice when:

- **A is SPD** and you have reasonable eigenvalue bounds
- **No inner products needed** -- unlike conjugate gradient, each iteration is a simple matrix-vector product plus vector updates, making it excellent for parallel and distributed computation
- **The condition number is moderate** (< 100) -- for very ill-conditioned systems, conjugate gradient or preconditioned methods are more robust
- **Simplicity matters** -- the algorithm is straightforward to implement and understand
- **Communication cost dominates** -- on distributed systems where global reductions (dot products) are expensive

### Limitations

- Requires eigenvalue bounds (not always easy to estimate)
- Only works for SPD matrices (for nonsymmetric systems, see [GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method))
- Numerically unstable for large polynomial degrees with ill-conditioned systems
- No adaptive step size -- convergence rate is fixed by the eigenvalue bounds
- Convergence rate is slightly worse than CG for the same condition number (CG adapts to the actual spectrum, Chebyshev assumes a uniform distribution)

## References

### Primary sources

1. **Golub, G.H. & Van Loan, C.F.** *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013. Section 10.1.5 -- derivation of the Chebyshev semi-iterative method and its optimality properties.
2. **Varga, R.S.** *Matrix Iterative Analysis*, 2nd ed., Springer, 2000. Chapter 5 -- comprehensive treatment of Chebyshev polynomial acceleration and convergence theory.
3. **Barrett, R. et al.** [*Templates for the Solution of Linear Systems*](http://www.netlib.org/templates/templates.pdf), SIAM, 1994. Section 2.3.9, Figure 2.11 -- practical algorithm template for the Chebyshev iteration.

### Additional resources

4. [Chebyshev iteration -- Wikipedia](https://en.wikipedia.org/wiki/Chebyshev_iteration)
5. [Chebyshev iteration method -- Encyclopedia of Mathematics](https://encyclopediaofmath.org/wiki/Chebyshev_iteration_method)
6. **Gutknecht, M.H. & Rollin, S.** [*The Chebyshev iteration revisited*](http://www.sam.math.ethz.ch/~mhg/pub/Cheby-02ParComp.pdf), Parallel Computing 28, 2002. Analysis of numerical stability and parameter orderings.
7. **Saad, Y.** *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003. Chapter 12 -- polynomial acceleration methods including Chebyshev iteration in the context of modern iterative solvers.

## License

[Apache License 2.0](LICENSE)
