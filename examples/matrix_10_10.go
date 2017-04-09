package main

import (
	"math"
	"chebyshev-iteration/chebyshev_iteration"
	"fmt"
)

func main() {
	n := 10
	eps := 0.1e-15
	h := 1/float64(n)
	gamma1 := 0.1
	gamma2 := 0.1+(4*math.Cos(math.Pi*h))/math.Pow(h, 2)
	d := 0.1 - (4*math.Pow(math.Sin(math.Pi*h/2), 2))/math.Pow(h, 2)
	h_2 := math.Pow(h, 2)

	b := []float64{d*1*h, d*2*h, d*3*h, d*4*h, d*5*h, d*6*h, d*7*h, d*8*h, d*9*h, d*(h-1) + d/h_2}
	a := []float64{
		2/h_2+d, 1/h_2, 0, 0, 0, 0, 0, 0, 0, 0,
		1/h_2, 2/h_2+d, 1/h_2, 0, 0, 0, 0, 0, 0, 0,
		0, 1/h_2, 2/h_2+d, 1/h_2, 0, 0, 0, 0, 0, 0,
		0, 0, 1/h_2, 2/h_2+d, 1/h_2, 0, 0, 0, 0, 0,
		0, 0, 0, 1/h_2, 2/h_2+d, 1/h_2, 0, 0, 0, 0,
		0, 0, 0, 0, 1/h_2, 2/h_2+d, 1/h_2, 0, 0, 0,
		0, 0, 0, 0, 0, 1/h_2, 2/h_2+d, 1/h_2, 0, 0,
		0, 0, 0, 0, 0, 0, 1/h_2, 2/h_2+d, 1/h_2, 0,
		0, 0, 0, 0, 0, 0, 0, 1/h_2, 2/h_2+d, 1/h_2,
		0, 0, 0, 0, 0, 0, 0, 0, 1/h_2, 2/h_2+d,
	}

	chebyshev_iteration.Init(n, eps,5120,a,b,gamma1,gamma2)

	result := chebyshev_iteration.ChebyshevIteration()

	fmt.Println("x := ", result.X, " iterations := ", result.Iterations)
}