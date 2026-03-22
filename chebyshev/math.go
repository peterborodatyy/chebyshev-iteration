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
