// Copyright (c) 2014 The mathutil Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mathutil

import (
	"math"
)

// IsPrimeUint16 returns true if n is prime. Typical run time is few ns.
func IsPrimeUint16(n uint16) bool {
	return n > 0 && primes16[n-1] == 1
}

// NextPrimeUint16 returns first prime > n and true if successful or an
// undefined value and false if there is no next prime in the uint16 limits.
// Typical run time is few ns.
func NextPrimeUint16(n uint16) (p uint16, ok bool) {
	return n + uint16(primes16[n]), n < 65521
}

// IsPrime returns true if n is prime. Typical run time is about 100 ns.
//
//TODO rename to IsPrimeUint32
func IsPrime(n uint32) bool {
	switch {
	case n&1 == 0:
		return n == 2
	case n%3 == 0:
		return n == 3
	case n%5 == 0:
		return n == 5
	case n%7 == 0:
		return n == 7
	case n%11 == 0:
		return n == 11
	case n%13 == 0:
		return n == 13
	case n%17 == 0:
		return n == 17
	case n%19 == 0:
		return n == 19
	case n%23 == 0:
		return n == 23
	case n%29 == 0:
		return n == 29
	case n%31 == 0:
		return n == 31
	case n%37 == 0:
		return n == 37
	case n%41 == 0:
		return n == 41
	case n%43 == 0:
		return n == 43
	case n%47 == 0:
		return n == 47
	case n%53 == 0:
		return n == 53 // Benchmarked optimum
	case n < 65536:
		// use table data
		return IsPrimeUint16(uint16(n))
	default:
		mod := ModPowUint32(2, (n+1)/2, n)
		if mod != 2 && mod != n-2 {
			return false
		}
		blk := &lohi[n>>24]
		lo, hi := blk.lo, blk.hi
		for lo <= hi {
			index := (lo + hi) >> 1
			liar := liars[index]
			switch {
			case n > liar:
				lo = index + 1
			case n < liar:
				hi = index - 1
			default:
				return false
			}
		}
		return true
	}
}

// IsPrimeUint64 returns true if n is prime. Typical run time is few tens of µs.
//
// SPRP bases: http://miller-rabin.appspot.com
func IsPrimeUint64(n uint64) bool {
	switch {
	case n%2 == 0:
		return n == 2
	case n%3 == 0:
		return n == 3
	case n%5 == 0:
		return n == 5
	case n%7 == 0:
		return n == 7
	case n%11 == 0:
		return n == 11
	case n%13 == 0:
		return n == 13
	case n%17 == 0:
		return n == 17
	case n%19 == 0:
		return n == 19
	case n%23 == 0:
		return n == 23
	case n%29 == 0:
		return n == 29
	case n%31 == 0:
		return n == 31
	case n%37 == 0:
		return n == 37
	case n%41 == 0:
		return n == 41
	case n%43 == 0:
		return n == 43
	case n%47 == 0:
		return n == 47
	case n%53 == 0:
		return n == 53
	case n%59 == 0:
		return n == 59
	case n%61 == 0:
		return n == 61
	case n%67 == 0:
		return n == 67
	case n%71 == 0:
		return n == 71
	case n%73 == 0:
		return n == 73
	case n%79 == 0:
		return n == 79
	case n%83 == 0:
		return n == 83
	case n%89 == 0:
		return n == 89 // Benchmarked optimum
	case n <= math.MaxUint16:
		return IsPrimeUint16(uint16(n))
	case n <= math.MaxUint32:
		return ProbablyPrimeUint32(uint32(n), 11000544) &&
			ProbablyPrimeUint32(uint32(n), 31481107)
	case n < 105936894253:
		return ProbablyPrimeUint64_32(n, 2) &&
			ProbablyPrimeUint64_32(n, 1005905886) &&
			ProbablyPrimeUint64_32(n, 1340600841)
	case n < 31858317218647:
		return ProbablyPrimeUint64_32(n, 2) &&
			ProbablyPrimeUint64_32(n, 642735) &&
			ProbablyPrimeUint64_32(n, 553174392) &&
			ProbablyPrimeUint64_32(n, 3046413974)
	case n < 3071837692357849:
		return ProbablyPrimeUint64_32(n, 2) &&
			ProbablyPrimeUint64_32(n, 75088) &&
			ProbablyPrimeUint64_32(n, 642735) &&
			ProbablyPrimeUint64_32(n, 203659041) &&
			ProbablyPrimeUint64_32(n, 3613982119)
	default:
		return ProbablyPrimeUint64_32(n, 2) &&
			ProbablyPrimeUint64_32(n, 325) &&
			ProbablyPrimeUint64_32(n, 9375) &&
			ProbablyPrimeUint64_32(n, 28178) &&
			ProbablyPrimeUint64_32(n, 450775) &&
			ProbablyPrimeUint64_32(n, 9780504) &&
			ProbablyPrimeUint64_32(n, 1795265022)
	}
}

// NextPrime returns first prime > n and true if successful or an undefined value and false if there
// is no next prime in the uint32 limits. Typical run time is about 2 µs.
//
//TODO rename to NextPrimeUint32
func NextPrime(n uint32) (p uint32, ok bool) {
	switch {
	case n < 65521:
		p16, _ := NextPrimeUint16(uint16(n))
		return uint32(p16), true
	case n >= math.MaxUint32-4:
		return
	}

	n++
	var d0, d uint32
	switch mod := n % 6; mod {
	case 0:
		d0, d = 1, 4
	case 1:
		d = 4
	case 2, 3, 4:
		d0, d = 5-mod, 2
	case 5:
		d = 2
	}

	p = n + d0
	if p < n { // overflow
		return
	}

	for {
		if IsPrime(p) {
			return p, true
		}

		p0 := p
		p += d
		if p < p0 { // overflow
			break
		}

		d ^= 6
	}
	return
}

// NextPrimeUint64 returns first prime > n and true if successful or an undefined value and false if there
// is no next prime in the uint64 limits. Typical run time is in hundreds of µs.
func NextPrimeUint64(n uint64) (p uint64, ok bool) {
	switch {
	case n < 65521:
		p16, _ := NextPrimeUint16(uint16(n))
		return uint64(p16), true
	case n >= 18446744073709551557: // last uint64 prime
		return
	}

	n++
	var d0, d uint64
	switch mod := n % 6; mod {
	case 0:
		d0, d = 1, 4
	case 1:
		d = 4
	case 2, 3, 4:
		d0, d = 5-mod, 2
	case 5:
		d = 2
	}

	p = n + d0
	if p < n { // overflow
		return
	}

	for {
		if ok = IsPrimeUint64(p); ok {
			break
		}

		p0 := p
		p += d
		if p < p0 { // overflow
			break
		}

		d ^= 6
	}
	return
}

// FactorTerm is one term of an integer factorization.
type FactorTerm struct {
	Prime uint32 // The divisor
	Power uint32 // Term == Prime^Power
}

// FactorTerms represent a factorization of an integer
type FactorTerms []FactorTerm

// FactorInt returns prime factorization of n > 1 or nil otherwise.
// Resulting factors are ordered by Prime. Typical run time is few µs.
func FactorInt(n uint32) (f FactorTerms) {
	switch {
	case n < 2:
		return
	case IsPrime(n):
		return []FactorTerm{{n, 1}}
	}

	f, w := make([]FactorTerm, 9), 0
	prime16 := uint16(0)
	for {
		var ok bool
		if prime16, ok = NextPrimeUint16(prime16); !ok {
			break
		}

		prime := uint32(prime16)
		if prime*prime > n {
			break
		}

		power := uint32(0)
		for n%prime == 0 {
			n /= prime
			power++
		}
		if power != 0 {
			f[w] = FactorTerm{prime, power}
			w++
		}
		if n == 1 {
			break
		}
	}
	if n != 1 {
		f[w] = FactorTerm{n, 1}
		w++
	}
	return f[:w]
}

// PrimorialProductsUint32 returns a slice of numbers in [lo, hi] which are a
// product of max 'max' primorials. The slice is not sorted.
//
// See also: http://en.wikipedia.org/wiki/Primorial
func PrimorialProductsUint32(lo, hi, max uint32) (r []uint32) {
	lo64, hi64 := int64(lo), int64(hi)
	if max > 31 { // N/A
		max = 31
	}

	var f func(int64, int64, uint32)
	f = func(n, p int64, emax uint32) {
		e := uint32(1)
		for n <= hi64 && e <= emax {
			n *= p
			if n >= lo64 && n <= hi64 {
				r = append(r, uint32(n))
			}
			if n < hi64 {
				p, _ := NextPrime(uint32(p))
				f(n, int64(p), e)
			}
			e++
		}
	}

	f(1, 2, max)
	return
}
