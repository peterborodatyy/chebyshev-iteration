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
	got := vecnorm([]float64{3, 4})
	if math.Abs(got-5.0) > 1e-15 {
		t.Errorf("vecnorm([3,4]) = %v, want 5", got)
	}

	got = vecnorm([]float64{0})
	if got != 0 {
		t.Errorf("vecnorm([0]) = %v, want 0", got)
	}

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
