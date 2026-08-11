/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package resource

import (
	"math"
	"math/big"
	"sync"
)

var (
	// A sync pool to reduce allocation.
	intPool  sync.Pool
	maxInt64 = big.NewInt(math.MaxInt64)
)

func init() {
	intPool.New = func() interface{} {
		return &big.Int{}
	}
}

// scaledValue scales unscaled from scale to newScale and returns it as an int64,
// rounding away from zero to match the int64 backend (negativeScaleInt64). ok is
// false when the true value overflows int64, in which case the result saturates
// to mostNegative or mostPositive.
//
// scale, newScale represent the scale of the unscaled decimal.
// The mathematical value of the decimal is unscaled * 10**(-scale).
func scaledValue(unscaled *big.Int, scale, newScale int64) (int64, bool) {
	dif := scale - newScale
	if dif == 0 {
		return bigToInt64Saturated(unscaled)
	}

	// 10^log10MaxInt64 is the first power of ten that exceeds maxint.
	const log10MaxInt64 = 19

	// Scale up: multiply by 10^(-dif). Zero stays zero; once -dif reaches
	// log10MaxInt64 the power alone exceeds maxint, so any nonzero value
	// overflows. Below that an int64 coefficient scales on the int64 path and a
	// larger one already cannot fit, so no big.Int is built here.
	if dif < 0 {
		if unscaled.Sign() == 0 {
			return 0, true
		}
		// dif <= -log10MaxInt64 rather than -dif >= log10MaxInt64 keeps an
		// extreme (near MinInt64) dif from being negated before it is bounded.
		if dif <= -log10MaxInt64 {
			if unscaled.Sign() < 0 {
				return mostNegative, false
			}
			return mostPositive, false
		}
		// A coefficient already outside int64 only grows when multiplied by a
		// positive power of ten, so it stays out of range.
		if !unscaled.IsInt64() {
			if unscaled.Sign() < 0 {
				return mostNegative, false
			}
			return mostPositive, false
		}
		// -dif is below log10MaxInt64 and the coefficient fits int64, so scale it
		// on the int64 path without allocating a big.Int.
		return positiveScaleInt64(unscaled.Int64(), Scale(-dif))
	}

	// Scale down: divide by 10^dif, rounding the quotient away from zero.

	// Fast path when unscaled fits int64 and the divisor stays below it. The
	// quotient is then strictly smaller in magnitude, so it cannot overflow.
	if unscaled.IsInt64() && dif < log10MaxInt64 {
		u := unscaled.Int64()
		divide := int64(math.Pow10(int(dif)))
		q := u / divide
		if u%divide != 0 {
			if u < 0 {
				q--
			} else {
				q++
			}
		}
		return q, true
	}

	// When 10^dif already exceeds |unscaled| the quotient is zero, so only the
	// sign from away-from-zero rounding survives. Skip building a dif-digit
	// divisor, which an extreme scale could blow up: dif >= BitLen implies
	// 10^dif > 2^dif >= 2^BitLen > |unscaled|.
	if dif >= int64(unscaled.BitLen()) {
		switch unscaled.Sign() {
		case 0:
			return 0, true
		case -1:
			return -1, true
		default:
			return 1, true
		}
	}

	divisor := intPool.Get().(*big.Int)
	exp := intPool.Get().(*big.Int)
	quotient := intPool.Get().(*big.Int)
	remainder := intPool.Get().(*big.Int)
	defer func() {
		intPool.Put(divisor)
		intPool.Put(exp)
		intPool.Put(quotient)
		intPool.Put(remainder)
	}()

	// divisor = 10^(dif)
	divisor.Exp(bigTen, exp.SetInt64(dif), nil)
	// QuoRem truncates toward zero, so remainder carries unscaled's sign and the
	// rounding step below moves away from zero exactly as the fast path does.
	quotient.QuoRem(unscaled, divisor, remainder)
	if remainder.Sign() != 0 {
		if unscaled.Sign() < 0 {
			quotient.Sub(quotient, bigOne)
		} else {
			quotient.Add(quotient, bigOne)
		}
	}
	return bigToInt64Saturated(quotient)
}

// bigToInt64Saturated returns v as an int64, saturating to mostNegative or
// mostPositive when v does not fit. ok is false when saturation occurred.
func bigToInt64Saturated(v *big.Int) (int64, bool) {
	if v.IsInt64() {
		return v.Int64(), true
	}
	if v.Sign() < 0 {
		return mostNegative, false
	}
	return mostPositive, false
}
