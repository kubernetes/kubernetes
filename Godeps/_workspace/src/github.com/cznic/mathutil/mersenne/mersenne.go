// Copyright (c) 2014 The mersenne Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package mersenne collects utilities related to Mersenne numbers[1] and/or some
of their properties.

Exponent

In this documentation the term 'exponent' refers to 'n' of a Mersenne number Mn
equal to 2^n-1. This package supports only uint32 sized exponents. New()
currently supports exponents only up to math.MaxInt32 (31 bits, up to 256 MB
required to represent such Mn in memory as a big.Int).

Links

Referenced from above:
 [1] http://en.wikipedia.org/wiki/Mersenne_number
*/
package mersenne

import (
	"math"
	"math/big"

	"github.com/cznic/mathutil"
	"github.com/remyoudompheng/bigfft"
)

var (
	_0 = big.NewInt(0)
	_1 = big.NewInt(1)
	_2 = big.NewInt(2)
)

// Knowns list the exponent of currently (March 2012) known Mersenne primes
// exponents in order.  See also: http://oeis.org/A000043 for a partial list.
var Knowns = []uint32{
	2,  // #1
	3,  // #2
	5,  // #3
	7,  // #4
	13, // #5
	17, // #6
	19, // #7
	31, // #8
	61, // #9
	89, // #10

	107,  // #11
	127,  // #12
	521,  // #13
	607,  // #14
	1279, // #15
	2203, // #16
	2281, // #17
	3217, // #18
	4253, // #19
	4423, // #20

	9689,   // #21
	9941,   // #22
	11213,  // #23
	19937,  // #24
	21701,  // #25
	23209,  // #26
	44497,  // #27
	86243,  // #28
	110503, // #29
	132049, // #30

	216091,   // #31
	756839,   // #32
	859433,   // #33
	1257787,  // #34
	1398269,  // #35
	2976221,  // #36
	3021377,  // #37
	6972593,  // #38
	13466917, // #39
	20996011, // #40

	24036583, // #41
	25964951, // #42
	30402457, // #43
	32582657, // #44
	37156667, // #45
	42643801, // #46
	43112609, // #47
	57885161, // #48
}

// Known maps the exponent of known Mersenne primes its ordinal number/rank.
// Ranks > 41 are currently provisional.
var Known map[uint32]int

func init() {
	Known = map[uint32]int{}
	for i, v := range Knowns {
		Known[v] = i + 1
	}
}

// New returns Mn == 2^n-1 for n <= math.MaxInt32 or nil otherwise.
func New(n uint32) (m *big.Int) {
	if n > math.MaxInt32 {
		return
	}

	m = big.NewInt(0)
	return m.Sub(m.SetBit(m, int(n), 1), _1)
}

// HasFactorUint32 returns true if d | Mn. Typical run time for a 32 bit factor
// and a 32 bit exponent is < 1 µs.
func HasFactorUint32(d, n uint32) bool {
	return d == 1 || d&1 != 0 && mathutil.ModPowUint32(2, n, d) == 1
}

// HasFactorUint64 returns true if d | Mn. Typical run time for a 64 bit factor
// and a 32 bit exponent is < 30 µs.
func HasFactorUint64(d uint64, n uint32) bool {
	return d == 1 || d&1 != 0 && mathutil.ModPowUint64(2, uint64(n), d) == 1
}

// HasFactorBigInt returns true if d | Mn, d > 0. Typical run time for a 128
// bit factor and a 32 bit exponent is < 75 µs.
func HasFactorBigInt(d *big.Int, n uint32) bool {
	return d.Cmp(_1) == 0 || d.Sign() > 0 && d.Bit(0) == 1 &&
		mathutil.ModPowBigInt(_2, big.NewInt(int64(n)), d).Cmp(_1) == 0
}

// HasFactorBigInt2 returns true if d | Mn, d > 0
func HasFactorBigInt2(d, n *big.Int) bool {
	return d.Cmp(_1) == 0 || d.Sign() > 0 && d.Bit(0) == 1 &&
		mathutil.ModPowBigInt(_2, n, d).Cmp(_1) == 0
}

/*
FromFactorBigInt returns n such that d | Mn if n <= max and d is odd. In other
cases zero is returned.

It is conjectured that every odd d ∊ N divides infinitely many Mersenne numbers.
The returned n should be the exponent of smallest such Mn.

NOTE: The computation of n from a given d performs roughly in O(n). It is
thus highly recomended to use the 'max' argument to limit the "searched"
exponent upper bound as appropriate. Otherwise the computation can take a long
time as a large factor can be a divisor of a Mn with exponent above the uint32
limits.

The FromFactorBigInt function is a modification of the original Will
Edgington's "reverse method", discussed here:
http://tech.groups.yahoo.com/group/primenumbers/message/15061
*/
func FromFactorBigInt(d *big.Int, max uint32) (n uint32) {
	if d.Bit(0) == 0 {
		return
	}

	var m big.Int
	for n < max {
		m.Add(&m, d)
		i := 0
		for ; m.Bit(i) == 1; i++ {
			if n == math.MaxUint32 {
				return 0
			}

			n++
		}
		m.Rsh(&m, uint(i))
		if m.Sign() == 0 {
			if n > max {
				n = 0
			}
			return
		}
	}
	return 0
}

// Mod sets mod to n % Mexp and returns mod. It panics for exp == 0 || exp >=
// math.MaxInt32 || n < 0.
func Mod(mod, n *big.Int, exp uint32) *big.Int {
	if exp == 0 || exp >= math.MaxInt32 || n.Sign() < 0 {
		panic(0)
	}

	m := New(exp)
	mod.Set(n)
	var x big.Int
	for mod.BitLen() > int(exp) {
		x.Set(mod)
		x.Rsh(&x, uint(exp))
		mod.And(mod, m)
		mod.Add(mod, &x)
	}
	if mod.BitLen() == int(exp) && mod.Cmp(m) == 0 {
		mod.SetInt64(0)
	}
	return mod
}

// ModPow2 returns x such that 2^Me % Mm == 2^x. It panics for m < 2.  Typical
// run time is < 1 µs. Use instead of ModPow(2, e, m) wherever possible.
func ModPow2(e, m uint32) (x uint32) {
	/*
		m < 2 -> panic
		e == 0 -> x == 0
		e == 1 -> x == 1

		2^M1 % M2 == 2^1 %  3 == 2^1    10 // 2^1, 3, 5, 7 ...		+2k
		2^M1 % M3 == 2^1 %  7 == 2^1   010 // 2^1, 4, 7, ...		+3k
		2^M1 % M4 == 2^1 % 15 == 2^1  0010 // 2^1, 5, 9, 13...		+4k
		2^M1 % M5 == 2^1 % 31 == 2^1 00010 // 2^1, 6, 11, 16...		+5k

		2^M2 % M2 == 2^3 %  3 == 2^1   10.. // 2^3, 5, 7, 9, 11, ...	+2k
		2^M2 % M3 == 2^3 %  7 == 2^0 001... // 2^3, 6, 9, 12, 15, ...	+3k
		2^M2 % M4 == 2^3 % 15 == 2^3   1000 // 2^3, 7, 11, 15, 19, ...	+4k
		2^M2 % M5 == 2^3 % 31 == 2^3  01000 // 2^3, 8, 13, 18, 23, ...	+5k

		2^M3 % M2 == 2^7 %   3 == 2^1       10..--.. // 2^3, 5, 7...	+2k
		2^M3 % M3 == 2^7 %   7 == 2^1      010...--- //	2^1, 4, 7...	+3k
		2^M3 % M4 == 2^7 %  15 == 2^3       1000.... //			+4k
		2^M3 % M5 == 2^7 %  31 == 2^2     00100..... //			+5k
		2^M3 % M6 == 2^7 %  63 == 2^1   000010...... //			+6k
		2^M3 % M7 == 2^7 % 127 == 2^0 0000001.......
		2^M3 % M8 == 2^7 % 255 == 2^7       10000000
		2^M3 % M9 == 2^7 % 511 == 2^7      010000000

		2^M4 % M2 == 2^15 %   3 == 2^1 10..--..--..--..
		2^M4 % M3 == 2^15 %   7 == 2^0 1...---...---...
		2^M4 % M4 == 2^15 %  15 == 2^3 1000....----....
		2^M4 % M5 == 2^15 %  31 == 2^0 1.....-----.....
		2^M4 % M6 == 2^15 %  63 == 2^3 1000......------
		2^M4 % M7 == 2^15 % 127 == 2^1 10.......-------
		2^M4 % M8 == 2^15 % 255 == 2^7 10000000........
		2^M4 % M9 == 2^15 % 511 == 2^6 1000000.........
	*/
	switch {
	case m < 2:
		panic(0)
	case e < 2:
		return e
	}

	if x = mathutil.ModPowUint32(2, e, m); x == 0 {
		return m - 1
	}

	return x - 1
}

// ModPow returns b^Me % Mm. Run time grows quickly with 'e' and/or 'm' when b
// != 2 (then ModPow2 is used).
func ModPow(b, e, m uint32) (r *big.Int) {
	if m == 1 {
		return big.NewInt(0)
	}

	if b == 2 {
		x := ModPow2(e, m)
		r = big.NewInt(0)
		r.SetBit(r, int(x), 1)
		return
	}

	bb := big.NewInt(int64(b))
	r = big.NewInt(1)
	for ; e != 0; e-- {
		r = bigfft.Mul(r, bb)
		Mod(r, r, m)
		bb = bigfft.Mul(bb, bb)
		Mod(bb, bb, m)
	}
	return
}

// ProbablyPrime returns true if Mn is prime or is a pseudoprime to base a.
// Note: Every Mp, prime p, is a prime or is a pseudoprime to base 2, actually
// to every base 2^i, i ∊ [1, p). In contrast - it is conjectured (w/o any
// known counterexamples) that no composite Mp, prime p, is a pseudoprime to
// base 3.
func ProbablyPrime(n, a uint32) bool {
	//TODO +test, +bench
	if a == 2 {
		return ModPow2(n-1, n) == 0
	}

	nMinus1 := New(n)
	nMinus1.Sub(nMinus1, _1)
	x := ModPow(a, n-1, n)
	return x.Cmp(_1) == 0 || x.Cmp(nMinus1) == 0
}
