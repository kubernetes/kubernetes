package neldermead_test

import (
	"math"
	"testing"

	"github.com/influxdata/influxdb/influxql/neldermead"
)

func round(num float64, precision float64) float64 {
	rnum := num * math.Pow(10, precision)
	var tnum float64
	if rnum < 0 {
		tnum = math.Floor(rnum - 0.5)
	} else {
		tnum = math.Floor(rnum + 0.5)
	}
	rnum = tnum / math.Pow(10, precision)
	return rnum
}

func almostEqual(a, b, e float64) bool {
	return math.Abs(a-b) < e
}

func Test_Optimize(t *testing.T) {

	constraints := func(x []float64) {
		for i := range x {
			x[i] = round(x[i], 5)
		}
	}
	// 100*(b-a^2)^2 + (1-a)^2
	//
	// Obvious global minimum at (a,b) = (1,1)
	//
	// Useful visualization:
	// https://www.wolframalpha.com/input/?i=minimize(100*(b-a%5E2)%5E2+%2B+(1-a)%5E2)
	f := func(x []float64) float64 {
		constraints(x)
		// a = x[0]
		// b = x[1]
		return 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]) + (1.0-x[0])*(1.0-x[0])
	}

	start := []float64{-1.2, 1.0}

	opt := neldermead.New()
	epsilon := 1e-5
	min, parameters := opt.Optimize(f, start, epsilon, 1)

	if !almostEqual(min, 0, epsilon) {
		t.Errorf("unexpected min: got %f exp 0", min)
	}

	if !almostEqual(parameters[0], 1, 1e-2) {
		t.Errorf("unexpected parameters[0]: got %f exp 1", parameters[0])
	}

	if !almostEqual(parameters[1], 1, 1e-2) {
		t.Errorf("unexpected parameters[1]: got %f exp 1", parameters[1])
	}

}
