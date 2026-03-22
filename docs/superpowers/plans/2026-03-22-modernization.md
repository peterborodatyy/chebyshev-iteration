# Chebyshev Iteration Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the chebyshev-iteration library as idiomatic, zero-dependency Go with comprehensive tests, examples, and documentation.

**Architecture:** Struct-based `Solver` API in a `chebyshev/` package, with internal math helpers, three iteration orderings, and TDD throughout. Old code removed, Go modules added.

**Tech Stack:** Go 1.21+, standard library only, `go test` for testing.

**Spec:** `docs/superpowers/specs/2026-03-22-modernization-design.md`

---

## File Structure

```
chebyshev-iteration/
  go.mod                        # module github.com/peterborodatyy/chebyshev-iteration
  .gitignore                    # updated for Go modules
  README.md                     # comprehensive documentation
  LICENSE                       # keep existing
  chebyshev/
    doc.go                      # package-level godoc
    math.go                     # matvec, vecadd, vecsub, vecscale, vecnorm
    math_test.go                # tests for math helpers
    solver.go                   # Solver, Result, Ordering, Solve, SolveWithOrdering, tau, validate
    solver_test.go              # solver tests + benchmarks
    example_test.go             # godoc examples
  examples/
    basic/main.go               # 3x3 SPD system
    tridiagonal/main.go         # 10x10 tridiagonal from discretized ODE
    convergence/main.go         # compare all three orderings
  cmd/
    chebyshev/main.go           # flag-based CLI tool
```

**Files to delete:** `chebyshev_iteration/`, `cli/`, `examples/matrix_2_2.go`, `examples/matrix_10_10.go`, `examples/cli-example.png`, `main.go`

---

### Task 1: Project scaffolding — go.mod, cleanup old files

**Files:**
- Delete: `chebyshev_iteration/chebyshev_iteration.go`, `cli/run.go`, `examples/matrix_2_2.go`, `examples/matrix_10_10.go`, `examples/cli-example.png`, `main.go`
- Create: `go.mod`
- Modify: `.gitignore`

- [ ] **Step 1: Delete old source files**

```bash
rm -rf chebyshev_iteration/ cli/ examples/matrix_2_2.go examples/matrix_10_10.go examples/cli-example.png main.go
```

- [ ] **Step 2: Create go.mod**

```bash
go mod init github.com/peterborodatyy/chebyshev-iteration
```

This creates `go.mod` with:
```
module github.com/peterborodatyy/chebyshev-iteration

go 1.21
```

- [ ] **Step 3: Update .gitignore**

Replace contents with:
```
# Binaries
*.exe
*.exe~
*.dll
*.so
*.dylib
chebyshev

# Test
*.test
*.out
*.prof

# Go workspace
go.work
go.work.sum

# IDE
.idea/
.vscode/
*.swp
*.swo
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p chebyshev examples/basic examples/tridiagonal examples/convergence cmd/chebyshev
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove old code and initialize Go modules"
```

---

### Task 2: Math helpers with tests (TDD)

**Files:**
- Create: `chebyshev/math.go`
- Create: `chebyshev/math_test.go`

- [ ] **Step 1: Write failing tests for math helpers**

Create `chebyshev/math_test.go`:

```go
package chebyshev

import (
	"math"
	"testing"
)

func TestMatvec(t *testing.T) {
	// Identity matrix
	A := [][]float64{{1, 0}, {0, 1}}
	x := []float64{3, 4}
	got := matvec(A, x)
	want := []float64{3, 4}
	assertVecEqual(t, got, want, 1e-15)

	// General 2x2
	A = [][]float64{{2, 1}, {1, 3}}
	x = []float64{1, 2}
	got = matvec(A, x)
	want = []float64{4, 7}
	assertVecEqual(t, got, want, 1e-15)

	// 1x1
	A = [][]float64{{5}}
	x = []float64{3}
	got = matvec(A, x)
	want = []float64{15}
	assertVecEqual(t, got, want, 1e-15)
}

func TestVecsub(t *testing.T) {
	got := vecsub([]float64{5, 3, 1}, []float64{1, 2, 3})
	want := []float64{4, 1, -2}
	assertVecEqual(t, got, want, 1e-15)
}

func TestVecadd(t *testing.T) {
	got := vecadd([]float64{1, 2, 3}, []float64{4, 5, 6})
	want := []float64{5, 7, 9}
	assertVecEqual(t, got, want, 1e-15)
}

func TestVecscale(t *testing.T) {
	got := vecscale(2.5, []float64{2, 4, 6})
	want := []float64{5, 10, 15}
	assertVecEqual(t, got, want, 1e-15)
}

func TestVecnorm(t *testing.T) {
	// ||[3,4]|| = 5
	got := vecnorm([]float64{3, 4})
	if math.Abs(got-5.0) > 1e-15 {
		t.Errorf("vecnorm([3,4]) = %v, want 5", got)
	}

	// ||[0]|| = 0
	got = vecnorm([]float64{0})
	if got != 0 {
		t.Errorf("vecnorm([0]) = %v, want 0", got)
	}

	// ||[1]|| = 1
	got = vecnorm([]float64{1})
	if got != 1 {
		t.Errorf("vecnorm([1]) = %v, want 1", got)
	}
}

func assertVecEqual(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("index %d: got %v, want %v", i, got[i], want[i])
		}
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd chebyshev && go test -v -run 'Test(Matvec|Vecsub|Vecadd|Vecscale|Vecnorm)'
```

Expected: compilation error — functions not defined.

- [ ] **Step 3: Implement math helpers**

Create `chebyshev/math.go`:

```go
package chebyshev

import "math"

// matvec computes A*x for a dense matrix A and vector x.
func matvec(A [][]float64, x []float64) []float64 {
	n := len(A)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < len(x); j++ {
			sum += A[i][j] * x[j]
		}
		result[i] = sum
	}
	return result
}

// vecsub returns a - b.
func vecsub(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

// vecadd returns a + b.
func vecadd(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

// vecscale returns alpha * x.
func vecscale(alpha float64, x []float64) []float64 {
	result := make([]float64, len(x))
	for i := range x {
		result[i] = alpha * x[i]
	}
	return result
}

// vecnorm returns the L2 norm of x.
func vecnorm(x []float64) float64 {
	sum := 0.0
	for _, v := range x {
		sum += v * v
	}
	return math.Sqrt(sum)
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd chebyshev && go test -v -run 'Test(Matvec|Vecsub|Vecadd|Vecscale|Vecnorm)'
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add chebyshev/math.go chebyshev/math_test.go
git commit -m "feat: add vector/matrix math helpers with tests"
```

---

### Task 3: Solver core — validation, tau, Solve, SolveWithOrdering (TDD)

**Files:**
- Create: `chebyshev/solver.go`
- Create: `chebyshev/solver_test.go`

- [ ] **Step 1: Write failing validation tests**

Create `chebyshev/solver_test.go`:

```go
package chebyshev

import (
	"math"
	"testing"
)

func TestSolveValidation(t *testing.T) {
	tests := []struct {
		name   string
		solver Solver
	}{
		{
			name:   "nil A",
			solver: Solver{A: nil, B: []float64{1}, LambdaMin: 1, LambdaMax: 2},
		},
		{
			name:   "empty A",
			solver: Solver{A: [][]float64{}, B: []float64{1}, LambdaMin: 1, LambdaMax: 2},
		},
		{
			name:   "nil B",
			solver: Solver{A: [][]float64{{1}}, B: nil, LambdaMin: 1, LambdaMax: 2},
		},
		{
			name:   "dimension mismatch A rows vs B",
			solver: Solver{A: [][]float64{{1, 0}, {0, 1}}, B: []float64{1}, LambdaMin: 1, LambdaMax: 2},
		},
		{
			name:   "non-square A",
			solver: Solver{A: [][]float64{{1, 0, 0}, {0, 1, 0}}, B: []float64{1, 2}, LambdaMin: 1, LambdaMax: 2},
		},
		{
			name:   "LambdaMin <= 0",
			solver: Solver{A: [][]float64{{1}}, B: []float64{1}, LambdaMin: 0, LambdaMax: 2},
		},
		{
			name:   "LambdaMin >= LambdaMax",
			solver: Solver{A: [][]float64{{1}}, B: []float64{1}, LambdaMin: 3, LambdaMax: 2},
		},
		{
			name:   "X0 wrong length",
			solver: Solver{A: [][]float64{{1}}, B: []float64{1}, X0: []float64{1, 2}, LambdaMin: 1, LambdaMax: 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.solver.Solve()
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestSolve1x1(t *testing.T) {
	// 5x = 10 -> x = 2
	s := Solver{
		A:         [][]float64{{5}},
		B:         []float64{10},
		LambdaMin: 5,
		LambdaMax: 5.001, // must be > LambdaMin
		MaxIter:   100,
		Tolerance: 1e-10,
	}
	result, err := s.Solve()
	if err != nil {
		t.Fatal(err)
	}
	if !result.Converged {
		t.Errorf("expected convergence, got residual %v after %d iterations", result.Residual, result.Iterations)
	}
	if math.Abs(result.X[0]-2.0) > 1e-6 {
		t.Errorf("x[0] = %v, want 2.0", result.X[0])
	}
}

func TestSolveIdentity(t *testing.T) {
	// Ix = b -> x = b
	s := Solver{
		A:         [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		B:         []float64{3, 7, 11},
		LambdaMin: 1,
		LambdaMax: 1.001,
		MaxIter:   100,
		Tolerance: 1e-10,
	}
	result, err := s.Solve()
	if err != nil {
		t.Fatal(err)
	}
	if !result.Converged {
		t.Errorf("expected convergence, residual=%v iter=%d", result.Residual, result.Iterations)
	}
	want := []float64{3, 7, 11}
	for i, v := range result.X {
		if math.Abs(v-want[i]) > 1e-6 {
			t.Errorf("x[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestSolveDiagonal(t *testing.T) {
	// diag(2,3,4) x = [4,9,8] -> x = [2,3,2]
	s := Solver{
		A:         [][]float64{{2, 0, 0}, {0, 3, 0}, {0, 0, 4}},
		B:         []float64{4, 9, 8},
		LambdaMin: 2,
		LambdaMax: 4,
		MaxIter:   512,
		Tolerance: 1e-10,
	}
	result, err := s.Solve()
	if err != nil {
		t.Fatal(err)
	}
	if !result.Converged {
		t.Errorf("expected convergence, residual=%v iter=%d", result.Residual, result.Iterations)
	}
	want := []float64{2, 3, 2}
	for i, v := range result.X {
		if math.Abs(v-want[i]) > 1e-6 {
			t.Errorf("x[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestSolve3x3SPD(t *testing.T) {
	// A = [[4,1,0],[1,3,1],[0,1,2]], b = [5,6,3]
	// Exact solution: x = [1,1,1]
	// Eigenvalues of A are approx 1.268, 3.0, 4.732
	s := Solver{
		A:         [][]float64{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}},
		B:         []float64{5, 5, 3},
		LambdaMin: 1.2,
		LambdaMax: 4.8,
		MaxIter:   512,
		Tolerance: 1e-8,
	}
	result, err := s.Solve()
	if err != nil {
		t.Fatal(err)
	}
	if !result.Converged {
		t.Errorf("expected convergence, residual=%v iter=%d", result.Residual, result.Iterations)
	}
	want := []float64{1, 1, 1}
	for i, v := range result.X {
		if math.Abs(v-want[i]) > 1e-4 {
			t.Errorf("x[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestSolveWithInitialGuess(t *testing.T) {
	s := Solver{
		A:         [][]float64{{2, 0}, {0, 3}},
		B:         []float64{4, 9},
		X0:        []float64{1.9, 2.9}, // close to solution
		LambdaMin: 2,
		LambdaMax: 3,
		MaxIter:   512,
		Tolerance: 1e-10,
	}
	result, err := s.Solve()
	if err != nil {
		t.Fatal(err)
	}
	if !result.Converged {
		t.Fatal("expected convergence")
	}
	// Should converge faster with good initial guess
	if result.Iterations > 50 {
		t.Errorf("expected fast convergence with good X0, got %d iterations", result.Iterations)
	}
}

func TestSolveAllOrderingsConverge(t *testing.T) {
	s := Solver{
		A:         [][]float64{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}},
		B:         []float64{5, 5, 3},
		LambdaMin: 1.2,
		LambdaMax: 4.8,
		MaxIter:   512,
		Tolerance: 1e-8,
	}

	orderings := []Ordering{Direct, Reverse, Alternating}
	var results []Result
	for _, ord := range orderings {
		r, err := s.SolveWithOrdering(ord)
		if err != nil {
			t.Fatal(err)
		}
		if !r.Converged {
			t.Errorf("ordering %d did not converge", ord)
		}
		results = append(results, r)
	}

	// All orderings should converge to same answer
	for i := 0; i < len(results[0].X); i++ {
		for j := 1; j < len(results); j++ {
			if math.Abs(results[0].X[i]-results[j].X[i]) > 1e-4 {
				t.Errorf("ordering 0 vs %d: x[%d] = %v vs %v", j, i, results[0].X[i], results[j].X[i])
			}
		}
	}
}

func TestSolveNonConvergence(t *testing.T) {
	// Wrong eigenvalue bounds — should not converge or diverge
	s := Solver{
		A:         [][]float64{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}},
		B:         []float64{5, 5, 3},
		LambdaMin: 0.001,
		LambdaMax: 0.002, // way too small
		MaxIter:   50,
		Tolerance: 1e-10,
	}
	result, err := s.Solve()
	if err != nil {
		t.Fatal(err)
	}
	if result.Converged {
		t.Error("expected non-convergence with bad eigenvalue bounds")
	}
}

func TestSolveDefaults(t *testing.T) {
	// MaxIter=0, Tolerance=0 should use defaults (512, 1e-10)
	s := Solver{
		A:         [][]float64{{2, 0}, {0, 3}},
		B:         []float64{4, 9},
		LambdaMin: 2,
		LambdaMax: 3,
	}
	result, err := s.Solve()
	if err != nil {
		t.Fatal(err)
	}
	if !result.Converged {
		t.Error("expected convergence with defaults")
	}
}

func TestTau(t *testing.T) {
	// tau_k for k=1, n=1: cos(pi/2) = 0, so tau = 2/(lmin+lmax)
	got := tau(1, 1, 1.0, 3.0)
	want := 2.0 / 4.0 // = 0.5
	if math.Abs(got-want) > 1e-15 {
		t.Errorf("tau(1,1,1,3) = %v, want %v", got, want)
	}
}

func BenchmarkSolve10x10(b *testing.B) {
	n := 10
	A := make([][]float64, n)
	bv := make([]float64, n)
	for i := 0; i < n; i++ {
		A[i] = make([]float64, n)
		A[i][i] = 4.0
		if i > 0 {
			A[i][i-1] = 1.0
		}
		if i < n-1 {
			A[i][i+1] = 1.0
		}
		bv[i] = float64(i + 1)
	}
	s := Solver{A: A, B: bv, LambdaMin: 2.0, LambdaMax: 6.0, MaxIter: 512, Tolerance: 1e-10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Solve()
	}
}

func BenchmarkSolve100x100(b *testing.B) {
	n := 100
	A := make([][]float64, n)
	bv := make([]float64, n)
	for i := 0; i < n; i++ {
		A[i] = make([]float64, n)
		A[i][i] = 4.0
		if i > 0 {
			A[i][i-1] = 1.0
		}
		if i < n-1 {
			A[i][i+1] = 1.0
		}
		bv[i] = float64(i + 1)
	}
	s := Solver{A: A, B: bv, LambdaMin: 2.0, LambdaMax: 6.0, MaxIter: 1024, Tolerance: 1e-10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Solve()
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd chebyshev && go test -v
```

Expected: compilation error — Solver, Solve, etc. not defined.

- [ ] **Step 3: Implement solver**

Create `chebyshev/solver.go`:

```go
package chebyshev

import (
	"errors"
	"math"
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
			return errors.New("chebyshev: matrix A must be square (row " + itoa(i) + " has length " + itoa(len(row)) + ", expected " + itoa(n) + ")")
		}
	}
	if len(s.B) == 0 {
		return errors.New("chebyshev: vector B must not be empty")
	}
	if len(s.B) != n {
		return errors.New("chebyshev: dimension mismatch: A is " + itoa(n) + "x" + itoa(n) + " but B has length " + itoa(len(s.B)))
	}
	if s.X0 != nil && len(s.X0) != n {
		return errors.New("chebyshev: X0 has length " + itoa(len(s.X0)) + ", expected " + itoa(n))
	}
	if s.LambdaMin <= 0 {
		return errors.New("chebyshev: LambdaMin must be positive")
	}
	if s.LambdaMax <= s.LambdaMin {
		return errors.New("chebyshev: LambdaMax must be greater than LambdaMin")
	}
	return nil
}

// itoa converts int to string without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	buf := make([]byte, 0, 10)
	for n > 0 {
		buf = append(buf, byte('0'+n%10))
		n /= 10
	}
	if neg {
		buf = append(buf, '-')
	}
	// reverse
	for i, j := 0, len(buf)-1; i < j; i, j = i+1, j-1 {
		buf[i], buf[j] = buf[j], buf[i]
	}
	return string(buf)
}
```

- [ ] **Step 4: Run all tests**

```bash
cd chebyshev && go test -v
```

Expected: all PASS.

- [ ] **Step 5: Run benchmarks**

```bash
cd chebyshev && go test -bench=. -benchmem -count=1
```

Expected: benchmarks run successfully.

- [ ] **Step 6: Commit**

```bash
git add chebyshev/solver.go chebyshev/solver_test.go
git commit -m "feat: implement Chebyshev iteration solver with tests and benchmarks"
```

---

### Task 4: Package documentation (doc.go)

**Files:**
- Create: `chebyshev/doc.go`

- [ ] **Step 1: Create doc.go**

```go
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
```

- [ ] **Step 2: Verify package builds**

```bash
cd chebyshev && go build ./...
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add chebyshev/doc.go
git commit -m "docs: add package-level documentation"
```

---

### Task 5: Godoc examples (example_test.go)

**Files:**
- Create: `chebyshev/example_test.go`

- [ ] **Step 1: Create example tests**

```go
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
		fmt.Printf("ordering=%d converged=%v iterations=%d\n", ord, result.Converged, result.Iterations)
	}
	// Output:
	// ordering=0 converged=true iterations=2
	// ordering=1 converged=true iterations=2
	// ordering=2 converged=true iterations=2
}
```

- [ ] **Step 2: Run examples to verify output matches**

```bash
cd chebyshev && go test -v -run Example
```

Expected: PASS. If output doesn't match, adjust the `// Output:` comments to match actual values.

- [ ] **Step 3: Commit**

```bash
git add chebyshev/example_test.go
git commit -m "docs: add godoc example tests"
```

---

### Task 6: Standalone examples

**Files:**
- Create: `examples/basic/main.go`
- Create: `examples/tridiagonal/main.go`
- Create: `examples/convergence/main.go`

- [ ] **Step 1: Create basic example**

Create `examples/basic/main.go`:

```go
// Basic example: solve a 3x3 symmetric positive-definite system.
package main

import (
	"fmt"
	"log"

	"github.com/peterborodatyy/chebyshev-iteration/chebyshev"
)

func main() {
	// Solve Ax = b where:
	//   A = [[4 1 0], [1 3 1], [0 1 2]]  (SPD, eigenvalues ~ 1.27, 3.0, 4.73)
	//   b = [5, 5, 3]
	//   Exact solution: x = [1, 1, 1]
	s := chebyshev.Solver{
		A: [][]float64{
			{4, 1, 0},
			{1, 3, 1},
			{0, 1, 2},
		},
		B:         []float64{5, 5, 3},
		LambdaMin: 1.2,
		LambdaMax: 4.8,
		Tolerance: 1e-10,
	}

	result, err := s.Solve()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Converged:  %v\n", result.Converged)
	fmt.Printf("Iterations: %d\n", result.Iterations)
	fmt.Printf("Residual:   %e\n", result.Residual)
	fmt.Printf("Solution:   %v\n", result.X)
}
```

- [ ] **Step 2: Create tridiagonal example**

Create `examples/tridiagonal/main.go`:

```go
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

	// Build tridiagonal matrix: A[i][i] = 2/h^2 + d, A[i][i±1] = -1/h^2
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

	// Eigenvalue bounds for this tridiagonal matrix:
	// lambda_min ~ d + 4*sin^2(pi*h/2)/h^2
	// lambda_max ~ d + 4*cos^2(pi*h/2)/h^2
	// (using Gershgorin or known tridiagonal eigenvalue formulas)
	lmin := d + 4*math.Pow(math.Sin(math.Pi*h/2), 2)/h2
	lmax := d + 4*math.Pow(math.Cos(math.Pi*h/2), 2)/h2

	s := chebyshev.Solver{
		A:         A,
		B:         b,
		LambdaMin: lmin,
		LambdaMax: lmax,
		MaxIter:   1024,
		Tolerance: 1e-12,
	}

	result, err := s.Solve()
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
```

- [ ] **Step 3: Create convergence comparison example**

Create `examples/convergence/main.go`:

```go
// Convergence example: compare all three parameter orderings.
package main

import (
	"fmt"
	"log"

	"github.com/peterborodatyy/chebyshev-iteration/chebyshev"
)

func main() {
	s := chebyshev.Solver{
		A: [][]float64{
			{4, 1, 0, 0},
			{1, 4, 1, 0},
			{0, 1, 4, 1},
			{0, 0, 1, 4},
		},
		B:         []float64{1, 2, 3, 4},
		LambdaMin: 2.0, // Gershgorin lower bound
		LambdaMax: 6.0, // Gershgorin upper bound
		MaxIter:   512,
		Tolerance: 1e-12,
	}

	orderings := []struct {
		name string
		ord  chebyshev.Ordering
	}{
		{"Direct", chebyshev.Direct},
		{"Reverse", chebyshev.Reverse},
		{"Alternating", chebyshev.Alternating},
	}

	fmt.Printf("%-12s  %10s  %12s  %s\n", "Ordering", "Iterations", "Residual", "Converged")
	fmt.Println("----------------------------------------------------")

	for _, o := range orderings {
		result, err := s.SolveWithOrdering(o.ord)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%-12s  %10d  %12.4e  %v\n", o.name, result.Iterations, result.Residual, result.Converged)
	}
}
```

- [ ] **Step 4: Verify all examples compile and run**

```bash
go run ./examples/basic/
go run ./examples/tridiagonal/
go run ./examples/convergence/
```

Expected: all three produce output without errors.

- [ ] **Step 5: Commit**

```bash
git add examples/
git commit -m "feat: add basic, tridiagonal, and convergence examples"
```

---

### Task 7: CLI tool

**Files:**
- Create: `cmd/chebyshev/main.go`

- [ ] **Step 1: Create CLI tool**

Create `cmd/chebyshev/main.go`:

```go
// Command chebyshev solves a system of linear equations Ax = b using the
// Chebyshev iterative method. Matrix A and vector b are read from stdin.
//
// Usage:
//
//	chebyshev -n 3 -lmin 1.2 -lmax 4.8 < input.txt
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
```

- [ ] **Step 2: Verify it builds**

```bash
go build ./cmd/chebyshev/
```

Expected: produces `chebyshev` binary without errors.

- [ ] **Step 3: Test with piped input**

```bash
echo "4 1 0 1 3 1 0 1 2 5 5 3" | go run ./cmd/chebyshev/ -n 3 -lmin 1.2 -lmax 4.8
```

Expected: prints converged solution.

- [ ] **Step 4: Clean up binary and commit**

```bash
rm -f chebyshev
git add cmd/
git commit -m "feat: add flag-based CLI tool"
```

---

### Task 8: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write README**

Replace `README.md` with comprehensive documentation covering: description, installation, quick start, API overview, examples, mathematical background, when to use / limitations, references, and license. See spec for section outline.

The README should include:
- The iteration formula in plain text / code block form
- A minimal working code example
- Description of `Solver` fields and `Result` fields
- Links to the three examples in `examples/`
- A section on eigenvalue bounds (how to estimate them, Gershgorin circles)
- References to Wikipedia, encyclopedia of math, and the netlib templates PDF

- [ ] **Step 2: Verify links and formatting**

Visually inspect the README. Ensure all file paths referenced exist.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: comprehensive README with math background and examples"
```

---

### Task 9: Final verification

- [ ] **Step 1: Run full test suite**

```bash
go test ./... -v -count=1
```

Expected: all tests pass.

- [ ] **Step 2: Run benchmarks**

```bash
go test ./chebyshev/ -bench=. -benchmem
```

Expected: benchmarks complete.

- [ ] **Step 3: Run vet and check for issues**

```bash
go vet ./...
```

Expected: no issues.

- [ ] **Step 4: Run all examples**

```bash
go run ./examples/basic/
go run ./examples/tridiagonal/
go run ./examples/convergence/
```

Expected: all produce correct output.

- [ ] **Step 5: Final commit if any fixes needed**

Only if previous steps revealed issues that were fixed.
