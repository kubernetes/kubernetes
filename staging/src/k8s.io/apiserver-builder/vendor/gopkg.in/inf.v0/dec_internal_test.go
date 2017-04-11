package inf

import (
	"math/big"
	"testing"
)

var decQuoRemZZZ = []struct {
	z, x, y  *Dec
	r        *big.Rat
	srA, srB int
}{
	// basic examples
	{NewDec(1, 0), NewDec(2, 0), NewDec(2, 0), big.NewRat(0, 1), 0, 1},
	{NewDec(15, 1), NewDec(3, 0), NewDec(2, 0), big.NewRat(0, 1), 0, 1},
	{NewDec(1, 1), NewDec(1, 0), NewDec(10, 0), big.NewRat(0, 1), 0, 1},
	{NewDec(0, 0), NewDec(2, 0), NewDec(3, 0), big.NewRat(2, 3), 1, 1},
	{NewDec(0, 0), NewDec(2, 0), NewDec(6, 0), big.NewRat(1, 3), 1, 1},
	{NewDec(1, 1), NewDec(2, 0), NewDec(12, 0), big.NewRat(2, 3), 1, 1},

	// examples from the Go Language Specification
	{NewDec(1, 0), NewDec(5, 0), NewDec(3, 0), big.NewRat(2, 3), 1, 1},
	{NewDec(-1, 0), NewDec(-5, 0), NewDec(3, 0), big.NewRat(-2, 3), -1, 1},
	{NewDec(-1, 0), NewDec(5, 0), NewDec(-3, 0), big.NewRat(-2, 3), 1, -1},
	{NewDec(1, 0), NewDec(-5, 0), NewDec(-3, 0), big.NewRat(2, 3), -1, -1},
}

func TestDecQuoRem(t *testing.T) {
	for i, a := range decQuoRemZZZ {
		z, rA, rB := new(Dec), new(big.Int), new(big.Int)
		s := scaleQuoExact{}.Scale(a.x, a.y)
		z.quoRem(a.x, a.y, s, true, rA, rB)
		if a.z.Cmp(z) != 0 || a.r.Cmp(new(big.Rat).SetFrac(rA, rB)) != 0 {
			t.Errorf("#%d QuoRemZZZ got %v, %v, %v; expected %v, %v", i, z, rA, rB, a.z, a.r)
		}
		if a.srA != rA.Sign() || a.srB != rB.Sign() {
			t.Errorf("#%d QuoRemZZZ wrong signs, got %v, %v; expected %v, %v", i, rA.Sign(), rB.Sign(), a.srA, a.srB)
		}
	}
}
