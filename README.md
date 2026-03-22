# Chebyshev Iteration

[![Tests](https://github.com/peterborodatyy/chebyshev-iteration/actions/workflows/test.yml/badge.svg)](https://github.com/peterborodatyy/chebyshev-iteration/actions/workflows/test.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/peterborodatyy/chebyshev-iteration/chebyshev.svg)](https://pkg.go.dev/github.com/peterborodatyy/chebyshev-iteration/chebyshev)
[![Go Report Card](https://goreportcard.com/badge/github.com/peterborodatyy/chebyshev-iteration)](https://goreportcard.com/report/github.com/peterborodatyy/chebyshev-iteration)

A Go library implementing the [Chebyshev iterative method](https://en.wikipedia.org/wiki/Chebyshev_iteration) for solving systems of linear equations **Ax = b**, where **A** is a symmetric positive-definite (SPD) matrix. The method uses optimal relaxation parameters derived from Chebyshev polynomial roots to accelerate convergence, requiring only estimates of the extreme eigenvalues of **A**.

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

## Mathematical Background

### The iteration

The Chebyshev semi-iterative method solves **Ax = b** via the recurrence:

```
x_{k+1} = x_k + tau_k * (b - A * x_k)
```

where the relaxation parameters tau_k are chosen as:

```
tau_k = 2 / (lambda_min + lambda_max - (lambda_max - lambda_min) * cos(pi * (2k - 1) / (2n)))
```

Here **n** is the polynomial degree (equal to `MaxIter`) and **k** ranges from 1 to n. These parameters are the reciprocals of the roots of the degree-n Chebyshev polynomial shifted to the interval [lambda_min, lambda_max].

### Convergence

The convergence factor after n iterations is bounded by:

```
||e_n|| / ||e_0|| <= 2 * rho^n / (1 + rho^(2n))
```

where `rho = (1 - sqrt(xi)) / (1 + sqrt(xi))` and `xi = lambda_min / lambda_max` (the reciprocal of the spectral condition number).

Tighter eigenvalue bounds (closer to the true extreme eigenvalues) yield faster convergence. Loose bounds still guarantee convergence but require more iterations.

### Eigenvalue bounds

The method requires estimates satisfying `0 < lambda_min <= lambda_i <= lambda_max` for all eigenvalues lambda_i of A. Common approaches:

- **Gershgorin circles:** For each row i, lambda lies in [a_ii - R_i, a_ii + R_i] where R_i = sum of |a_ij| for j != i. Take min of lower bounds and max of upper bounds.
- **Known structure:** For tridiagonal matrices from finite differences, eigenvalue formulas exist in closed form.
- **Power iteration / Lanczos:** Compute a few extreme eigenvalues numerically.

### Numerical stability

The Chebyshev iteration can be numerically unstable when the polynomial degree (MaxIter) is large relative to the condition number. The `Alternating` ordering mitigates this by interleaving parameters from both ends of the spectrum. For well-conditioned systems (condition number < 10), any ordering works. For moderate condition numbers (10-100), prefer `Alternating`. For poorly conditioned systems, consider using a preconditioner or a different method (e.g., conjugate gradient).

## When to Use

The Chebyshev iteration is a good choice when:

- **A is SPD** and you have reasonable eigenvalue bounds
- **No inner products needed** -- unlike conjugate gradient, each iteration is a simple matrix-vector product plus vector operations, making it easy to parallelize
- **The condition number is moderate** (< 100) -- for very ill-conditioned systems, conjugate gradient or preconditioned methods are more robust
- **Simplicity matters** -- the algorithm is straightforward to implement and understand

### Limitations

- Requires eigenvalue bounds (not always easy to estimate)
- Only works for SPD matrices
- Numerically unstable for large polynomial degrees with ill-conditioned systems
- No adaptive step size -- convergence rate is fixed by the eigenvalue bounds
- For general sparse systems, Krylov methods (CG, GMRES) are usually more practical

## References

1. [Chebyshev iteration - Wikipedia](https://en.wikipedia.org/wiki/Chebyshev_iteration)
2. [Chebyshev iteration method - Encyclopedia of Mathematics](https://encyclopediaofmath.org/wiki/Chebyshev_iteration_method)
3. [Parallel Chebyshev iteration (ETH Zurich)](http://www.sam.math.ethz.ch/~mhg/pub/Cheby-02ParComp.pdf)
4. [Templates for the Solution of Linear Systems (Netlib)](http://www.netlib.org/templates/templates.pdf)

## License

[Apache License 2.0](LICENSE)
