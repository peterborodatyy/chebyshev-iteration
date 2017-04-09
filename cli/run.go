package cli

import (
	"fmt"
	"chebyshev-iteration/chebyshev_iteration"
)

var n int
var eps float64
var iterationsCount int
var lmin float64
var lmax float64

func Run() {

	fmt.Print("Input matrix dimension: ")

	fmt.Scanf("%d", &n)

	fmt.Print("Input epsilon: ")

	fmt.Scanf("%e", &eps)

	fmt.Print("Input iterations count: ")

	fmt.Scanf("%d", &iterationsCount)

	fmt.Println("Input matrix B (nx1):")

	b, _ := readMatrixRow(n)

	fmt.Println("B: ", b)

	a := make([]float64, 0)

	fmt.Println("Input matrix A (nxn):")

	for i := 0; i < n; i++ {
		row, _ := readMatrixRow(n)

		a = append(a, row...)
	}

	fmt.Println("Matrix A:", a)

	fmt.Print("Input lMin: ")

	fmt.Scanf("%e", &lmin)

	fmt.Print("Input lMax: ")

	fmt.Scanf("%e", &lmax)

	chebyshev_iteration.Init(n, eps, iterationsCount, a, b, lmin, lmax)

	result := chebyshev_iteration.ChebyshevIteration()

	fmt.Println("x := ", result.X, " iterations := ", result.Iterations)
}

func readMatrixRow(n int) ([]float64, error) {
	in := make([]float64, n)

	for i := range in {
		_, err := fmt.Scan(&in[i])

		if err != nil {
			return in[:i], err
		}
	}

	return in, nil
}