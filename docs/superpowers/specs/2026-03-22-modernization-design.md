# Chebyshev Iteration Modernization Design

## Goal

Modernize the chebyshev-iteration repository: rewrite the core library to be idiomatic Go, add comprehensive tests, multiple examples, and a professional README.

## Current Problems

1. **No Go modules** — uses GOPATH-style imports
2. **Global mutable state** — `Init()` sets package-level vars, not safe for concurrent use
3. **Race condition** — three goroutines share mutable `x_k` without synchronization
4. **Unmaintained dependency** — `github.com/mitsuse/matrix-go`
5. **No tests**
6. **Minimal README** — no math background, no API docs, broken import paths
7. **Examples are standalone main packages** — not runnable godoc examples

## Design

### Module Path

```
module github.com/peterborodatyy/chebyshev-iteration
```

### Zero External Dependencies

All needed matrix operations (mat-vec multiply, vector add/scale, norm) are simple enough to implement in ~50 lines. Removes the unmaintained `matrix-go` dependency.

### Core API

```go
package chebyshev

type Solver struct {
    A          [][]float64  // coefficient matrix (n x n), must be SPD
    B          []float64    // right-hand side vector
    X0         []float64    // initial guess (optional, defaults to zero vector)
    LambdaMin  float64      // smallest eigenvalue estimate (must be > 0)
    LambdaMax  float64      // largest eigenvalue estimate (must be > LambdaMin)
    MaxIter    int          // max iterations (default 512)
    Tolerance  float64      // convergence tolerance on ||b - Ax||_2 (default 1e-10)
}

type Result struct {
    X          []float64    // solution vector (best found, even if not converged)
    Iterations int          // iterations performed
    Converged  bool         // whether tolerance was met
    Residual   float64      // final ||b - Ax||_2
}

type Ordering int
const (
    Direct      Ordering = iota  // k = 1, 2, ..., n
    Reverse                      // k = n, n-1, ..., 1
    Alternating                  // k = 1, n, 2, n-1, ...
)

func (s *Solver) Solve() (Result, error)
func (s *Solver) SolveWithOrdering(ord Ordering) (Result, error)
```

**Semantics:**
- `Solve()` uses `Direct` ordering (the standard choice).
- `SolveWithOrdering()` allows experimenting with parameter orderings.
- **Initial guess:** If `X0` is nil, the zero vector is used. If provided, must have length n.
- **Convergence:** Checked via absolute L2 residual norm: `||b - A*x_k||_2 <= Tolerance`.
- **Non-convergence:** Returns the best solution found with `Converged: false` and `error == nil`. Errors are reserved for invalid inputs only.
- **NaN/Inf:** If the residual becomes NaN or Inf during iteration, the solver returns early with `Converged: false`.
- **Defaults:** MaxIter defaults to 512, Tolerance defaults to 1e-10 (when zero-valued).
- **Validation errors:** nil/empty matrices, dimension mismatches, LambdaMin >= LambdaMax, LambdaMin <= 0.
- **SPD requirement:** The Chebyshev iteration converges only for symmetric positive-definite matrices. The solver does not validate SPD-ness (O(n^3) cost) but documents this precondition.
- **Thread safety:** `Solve()` does not mutate the Solver. Multiple goroutines may call `Solve()` concurrently on the same Solver.

### Math Helpers (internal)

```go
func matvec(A [][]float64, x []float64) []float64
func vecsub(a, b []float64) []float64
func vecadd(a, b []float64) []float64
func vecscale(alpha float64, x []float64) []float64
func vecnorm(x []float64) float64
func tau(k, n int, lambdaMin, lambdaMax float64) float64
```

Note: `tau`'s `n` parameter is the Chebyshev polynomial degree (number of parameters in the cycle), which equals `MaxIter`. The `tau` function lives in `solver.go` alongside the iteration logic since it is Chebyshev-specific, not generic linear algebra.

### Iteration Formula

```
x_{k+1} = x_k + tau_k * (b - A * x_k)
```

where:

```
tau_k = 2 / (lambda_min + lambda_max - (lambda_max - lambda_min) * cos(pi * (2k - 1) / (2n)))
```

Here `n` is `MaxIter` (the Chebyshev polynomial degree / cycle length). This conflates the iteration cap with the polynomial degree, matching the original implementation. For this library's scope, this is acceptable.

### Package Structure

```
chebyshev-iteration/
  chebyshev/
    solver.go          # Solver struct, Solve, SolveWithOrdering, validate
    ordering.go        # Ordering type, iteration sequence generators
    math.go            # matvec, vecadd, vecsub, vecscale, vecnorm, tau
    math_test.go       # tests for math helpers
    solver_test.go     # unit tests + benchmarks for Solver
    example_test.go    # godoc examples
    doc.go             # package-level documentation
  examples/
    basic/main.go          # simple 3x3 SPD system
    tridiagonal/main.go    # 10x10 tridiagonal (discretized ODE)
    convergence/main.go    # compare orderings
  cmd/
    chebyshev/main.go      # CLI tool (flag-based: -n, -eps, -maxiter, -lmin, -lmax, reads A and b from stdin)
  README.md
  go.mod
  .gitignore
```

### Tests

| Category | What |
|----------|------|
| Validation | dimension mismatch, invalid eigenvalue bounds, nil inputs |
| 1x1 degenerate | single-element system |
| Identity matrix | Ax=b where A=I, solution is x=b |
| Diagonal matrix | known exact solution |
| 3x3 SPD | hand-verified solution |
| Ordering equivalence | all three orderings converge to same answer |
| Non-convergence | wrong eigenvalue bounds -> Converged=false |
| Math helpers | matvec, norm, tau against known values |
| Benchmarks | 10x10, 100x100 |
| Godoc examples | Example, Example_ordering |

### README Sections

1. Title + badges (Go Reference)
2. One-paragraph description with math context
3. Installation (`go get`)
4. Quick start (minimal working example)
5. API overview (Solver fields, Result fields, Ordering)
6. Examples (brief description of each, link to code)
7. Mathematical background (iteration formula, tau, convergence)
8. When to use / limitations
9. References
10. License

### Files to Remove

- `chebyshev_iteration/` (old package, replaced by `chebyshev/`)
- `cli/` (replaced by `cmd/chebyshev/`)
- `examples/matrix_2_2.go`, `examples/matrix_10_10.go` (replaced by new examples)
- `examples/cli-example.png` (outdated)

### Migration

This is a complete rewrite — old code is removed, new code is written from scratch. The repo has only 2 commits and no known downstream users, so breaking changes are fine.
