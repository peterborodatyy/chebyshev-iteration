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
		LambdaMax: 5.001,
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
	// A = [[4,1,0],[1,3,1],[0,1,2]], b = [5,5,3]
	// Exact solution: x = [1,1,1]
	// Eigenvalues of A are approx 1.268, 3.0, 4.732
	s := Solver{
		A:         [][]float64{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}},
		B:         []float64{5, 5, 3},
		LambdaMin: 1.268,
		LambdaMax: 4.732,
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
		X0:        []float64{1.9, 2.9},
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
	if result.Iterations > 50 {
		t.Errorf("expected fast convergence with good X0, got %d iterations", result.Iterations)
	}
}

func TestSolveAllOrderingsConverge(t *testing.T) {
	s := Solver{
		A:         [][]float64{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}},
		B:         []float64{5, 5, 3},
		LambdaMin: 1.268,
		LambdaMax: 4.732,
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
	s := Solver{
		A:         [][]float64{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}},
		B:         []float64{5, 5, 3},
		LambdaMin: 0.001,
		LambdaMax: 0.002,
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

func TestSolveDivergenceNaNInf(t *testing.T) {
	// Eigenvalue bounds far too small cause tau to be huge, leading to divergence
	s := Solver{
		A:         [][]float64{{100, 0}, {0, 200}},
		B:         []float64{100, 200},
		LambdaMin: 0.0001,
		LambdaMax: 0.0002,
		MaxIter:   100,
		Tolerance: 1e-10,
	}
	result, err := s.Solve()
	if err != nil {
		t.Fatal(err)
	}
	if result.Converged {
		t.Error("expected non-convergence")
	}
}

func TestSolveDefaults(t *testing.T) {
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
	want := 0.5
	if math.Abs(got-want) > 1e-15 {
		t.Errorf("tau(1,1,1,3) = %v, want %v", got, want)
	}
}

func TestOrderingSequence(t *testing.T) {
	direct := orderingSequence(Direct, 5)
	if len(direct) != 5 || direct[0] != 1 || direct[4] != 5 {
		t.Errorf("direct = %v, want [1,2,3,4,5]", direct)
	}

	rev := orderingSequence(Reverse, 5)
	if len(rev) != 5 || rev[0] != 5 || rev[4] != 1 {
		t.Errorf("reverse = %v, want [5,4,3,2,1]", rev)
	}

	alt := orderingSequence(Alternating, 6)
	want := []int{1, 6, 2, 5, 3, 4}
	for i, v := range alt {
		if v != want[i] {
			t.Errorf("alternating[%d] = %d, want %d", i, v, want[i])
		}
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
