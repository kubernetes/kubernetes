package inf

import (
	"math/big"
)

// Rounder represents a method for rounding the (possibly infinite decimal)
// result of a division to a finite Dec. It is used by Dec.Round() and
// Dec.Quo().
//
// See the Example for results of using each Rounder with some sample values.
//
type Rounder rounder

// See http://speleotrove.com/decimal/damodel.html#refround for more detailed
// definitions of these rounding modes.
var (
	RoundDown     Rounder // towards 0
	RoundUp       Rounder // away from 0
	RoundFloor    Rounder // towards -infinity
	RoundCeil     Rounder // towards +infinity
	RoundHalfDown Rounder // to nearest; towards 0 if same distance
	RoundHalfUp   Rounder // to nearest; away from 0 if same distance
	RoundHalfEven Rounder // to nearest; even last digit if same distance
)

// RoundExact is to be used in the case when rounding is not necessary.
// When used with Quo or Round, it returns the result verbatim when it can be
// expressed exactly with the given precision, and it returns nil otherwise.
// QuoExact is a shorthand for using Quo with RoundExact.
var RoundExact Rounder

type rounder interface {

	// When UseRemainder() returns true, the Round() method is passed the
	// remainder of the division, expressed as the numerator and denominator of
	// a rational.
	UseRemainder() bool

	// Round sets the rounded value of a quotient to z, and returns z.
	// quo is rounded down (truncated towards zero) to the scale obtained from
	// the Scaler in Quo().
	//
	// When the remainder is not used, remNum and remDen are nil.
	// When used, the remainder is normalized between -1 and 1; that is:
	//
	//  -|remDen| < remNum < |remDen|
	//
	// remDen has the same sign as y, and remNum is zero or has the same sign
	// as x.
	Round(z, quo *Dec, remNum, remDen *big.Int) *Dec
}

type rndr struct {
	useRem bool
	round  func(z, quo *Dec, remNum, remDen *big.Int) *Dec
}

func (r rndr) UseRemainder() bool {
	return r.useRem
}

func (r rndr) Round(z, quo *Dec, remNum, remDen *big.Int) *Dec {
	return r.round(z, quo, remNum, remDen)
}

var intSign = []*big.Int{big.NewInt(-1), big.NewInt(0), big.NewInt(1)}

func roundHalf(f func(c int, odd uint) (roundUp bool)) func(z, q *Dec, rA, rB *big.Int) *Dec {
	return func(z, q *Dec, rA, rB *big.Int) *Dec {
		z.Set(q)
		brA, brB := rA.BitLen(), rB.BitLen()
		if brA < brB-1 {
			// brA < brB-1 => |rA| < |rB/2|
			return z
		}
		roundUp := false
		srA, srB := rA.Sign(), rB.Sign()
		s := srA * srB
		if brA == brB-1 {
			rA2 := new(big.Int).Lsh(rA, 1)
			if s < 0 {
				rA2.Neg(rA2)
			}
			roundUp = f(rA2.Cmp(rB)*srB, z.UnscaledBig().Bit(0))
		} else {
			// brA > brB-1 => |rA| > |rB/2|
			roundUp = true
		}
		if roundUp {
			z.UnscaledBig().Add(z.UnscaledBig(), intSign[s+1])
		}
		return z
	}
}

func init() {
	RoundExact = rndr{true,
		func(z, q *Dec, rA, rB *big.Int) *Dec {
			if rA.Sign() != 0 {
				return nil
			}
			return z.Set(q)
		}}
	RoundDown = rndr{false,
		func(z, q *Dec, rA, rB *big.Int) *Dec {
			return z.Set(q)
		}}
	RoundUp = rndr{true,
		func(z, q *Dec, rA, rB *big.Int) *Dec {
			z.Set(q)
			if rA.Sign() != 0 {
				z.UnscaledBig().Add(z.UnscaledBig(), intSign[rA.Sign()*rB.Sign()+1])
			}
			return z
		}}
	RoundFloor = rndr{true,
		func(z, q *Dec, rA, rB *big.Int) *Dec {
			z.Set(q)
			if rA.Sign()*rB.Sign() < 0 {
				z.UnscaledBig().Add(z.UnscaledBig(), intSign[0])
			}
			return z
		}}
	RoundCeil = rndr{true,
		func(z, q *Dec, rA, rB *big.Int) *Dec {
			z.Set(q)
			if rA.Sign()*rB.Sign() > 0 {
				z.UnscaledBig().Add(z.UnscaledBig(), intSign[2])
			}
			return z
		}}
	RoundHalfDown = rndr{true, roundHalf(
		func(c int, odd uint) bool {
			return c > 0
		})}
	RoundHalfUp = rndr{true, roundHalf(
		func(c int, odd uint) bool {
			return c >= 0
		})}
	RoundHalfEven = rndr{true, roundHalf(
		func(c int, odd uint) bool {
			return c > 0 || c == 0 && odd == 1
		})}
}
