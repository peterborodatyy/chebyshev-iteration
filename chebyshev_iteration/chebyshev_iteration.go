package chebyshev_iteration

import (
	"github.com/mitsuse/matrix-go"
	"math"
	"github.com/mitsuse/matrix-go/dense"
	"sync"
)

var n int // Matrix dimension

var h float64 // Step size

var eps float64 // Precision

var iterationCount int = 512 // Number of iterations

var x_k matrix.Matrix // Matrix х

var gamma1 float64 // m
var gamma2 float64 // M

var A *dense.Matrix
var B *dense.Matrix

var results chan Result

var wg sync.WaitGroup

type Result struct {
	Iterations int
	X []float64
}

func Init(dimension int, epsilon float64, iterations int, a []float64, b []float64, lmin float64, lmax float64) {
	n = dimension
	eps = epsilon
	iterationCount = iterations

	h = 1 / float64(n) // Setting step size.

	// Defining gamma1, gama2
	gamma1 = lmin
	gamma2 = lmax

	x := dense.Zeros(n, 1)

	B = dense.New(n, 1)(b...)
	A = dense.New(n, n)(a...)

	x_k = x.Base()
}

var (
	response_lock sync.Mutex
	response Result
)

// Chebyshev iteration method
func ChebyshevIteration() Result {

	wg.Add(3)

	results = make(chan Result)

	response = Result{}

	go direct(iterationCount, iterate)
	go reverse(iterationCount, iterate)
	go alternating(iterationCount, iterate)

	go func() {
		for i := range results {
			if len(response.X) == 0 {
				response_lock.Lock()

				response = i

				response_lock.Unlock()
			}
		}
	}()

	wg.Wait()

	return response
}

func iterate(k int) bool {

	x_k = iteration(x_k, k, B, A)

	response := []float64{}

	// Getting norm
	norm := norm(dense.Zeros(int(n), int(n)).Add(A).Scalar(-1).Multiply(x_k).Add(B))

	if norm <= eps { // Precision less then epsilon

		c := x_k.All()

		for c.HasNext() {
			element, _, _ := c.Get()

			response = append(response, element)

		}

		results <- Result{
			Iterations:k,
			X: response,
		}

		return false

	} else if math.IsInf(norm, 1) { // Precision == infinity

		results <- Result{
			Iterations:k,
			X: []float64{},
		}

		return false
	}

	return true
}

// Iteration with Chebyshev params
func iteration(x matrix.Matrix, k int, b *dense.Matrix, A *dense.Matrix) matrix.Matrix {
	return dense.Zeros(int(n), int(n)).Add(A).Scalar(-1).Multiply(dense.Convert(x)).Add(b).Scalar(tau(k)).Add(dense.Convert(x))
}

// Getting tau.
func tau(k int) float64 {
	return 2 / (gamma1 + gamma2 -(gamma2 - gamma1)*(-math.Cos(math.Pi*(2*float64(k)-1)/(2*float64(iterationCount)))))
}

// Getting norm of х
func norm(x matrix.Matrix) float64 {
	return math.Sqrt(x.Transpose().Multiply(x).Get(0, 0))
}

// Result q_n on step n
func qn(k float64) float64 {
	return (2*math.Pow(rho_1(), k))/(1+math.Pow(rho_1(), 2*k))
}

func xi() float64 {
	return gamma1/gamma2
}

func rho_1() float64 {
	return (1-math.Sqrt(xi()))/(1+math.Sqrt(xi()))
}

// Direct sequence
func direct(max int, iterate func(i int) bool) {

	defer wg.Done()

	for i := 1; i <= max; i++ {
		if iterate(i) == false {
			return
		}
	}

	results <- Result{
		Iterations:max,
		X: []float64{},
	}
}

// Reverse sequence
func reverse(max int, iterate func(int) bool) {

	defer wg.Done()

	for i := max; i >= 1; i-- {
		if iterate(i) == false {
			return
		}
	}

	results <- Result{
		Iterations:max,
		X: []float64{},
	}
}

// Alternating equense
func alternating(max int, iterate func(int) bool)  {

	defer wg.Done()

	i := 1
	j := max
	k := 0

	for i != j {

		if k == 0 {

			if iterate(i) == false {
				return
			}

			i = i + 1
			k = 1
		} else {
			if iterate(j) == false {
				return
			}

			j = j - 1
			k = 0
		}
	}

	results <- Result{
		Iterations:max,
		X: []float64{},
	}
}