/*
Copyright 2014 The Kubernetes Authors.

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
	"strconv"

	inf "gopkg.in/inf.v0"
)

// Scale is used for getting and setting the base-10 scaled value.
// Base-2 scales are omitted for mathematical simplicity.
//
// Scale is the power-of-10 exponent. 1E2 has scale 2. 1E-2 has scale -2.
// This is the opposite of inf.Scale which counts digits after the decimal
// point (for inf.Scale, 1E2 has scale -2. 1E-2 has scale 2).
//
// See Quantity.ScaledValue for more details.
type Scale int32

// infScale adapts a Scale value to an inf.Scale value, returning false in overflow cases.
func (s Scale) infScale() (inf.Scale, bool) {
	// inf.Scale is upside-down
	// TODO: saturate instead of wrap in overflow cases
	return inf.Scale(-s), s != math.MinInt32
}

// canInfScale reports whether infScale is faithful for s. Scale is int32 and
// infScale negates it, so math.MinInt32 is the one scale whose negation
// overflows back to itself and cannot be represented as an inf.Scale.
func (s Scale) canInfScale() bool {
	return s != math.MinInt32
}

// widenedScale is a Scale widened to int64, so it also holds a negated
// inf.Scale and the difference of two scales without overflow.
// Conversions should happen only via widenScale, widenInfScale and
// narrowScale to ensure the sign is converted safely.
type widenedScale int64

// widenScale returns s as a widenedScale.
func widenScale(s Scale) widenedScale {
	return widenedScale(s)
}

// narrowScale returns s as a Scale. fits is false when s is outside
// of the Scale range, in which case the result saturates to the nearest Scale bound.
func narrowScale(s widenedScale) (narrowed Scale, fits bool) {
	switch {
	case s < math.MinInt32:
		return math.MinInt32, false
	case s > math.MaxInt32:
		return math.MaxInt32, false
	default:
		return Scale(s), true
	}
}

// widenInfScale returns the base-10 exponent of an inf.Dec with scale s.
// This performs the negation required to switch from an inf.Scale to a Scale.
func widenInfScale(s inf.Scale) widenedScale {
	return -widenedScale(s)
}

const (
	Nano  Scale = -9
	Micro Scale = -6
	Milli Scale = -3
	Kilo  Scale = 3
	Mega  Scale = 6
	Giga  Scale = 9
	Tera  Scale = 12
	Peta  Scale = 15
	Exa   Scale = 18

	nanoInfScale = inf.Scale(9)
)

var (
	Zero = int64Amount{}

	// Used by quantity strings - treat as read only
	zeroBytes = []byte("0")
)

// int64Amount represents a fixed precision numerator and arbitrary scale exponent. It is faster
// than operations on inf.Dec for values that can be represented as int64.
// +k8s:openapi-gen=true
type int64Amount struct {
	value int64
	scale Scale
}

// Sign returns 0 if the value is zero, -1 if it is less than 0, or 1 if it is greater than 0.
func (a int64Amount) Sign() int {
	switch {
	case a.value == 0:
		return 0
	case a.value > 0:
		return 1
	default:
		return -1
	}
}

// AsInt64 returns the current amount as an int64 at scale 0, or false if the value cannot be
// represented in an int64 OR would result in a loss of precision. This method is intended as
// an optimization to avoid calling AsDec.
func (a int64Amount) AsInt64() (int64, bool) {
	if a.scale == 0 {
		return a.value, true
	}
	if a.scale < 0 {
		// TODO: attempt to reduce factors, although it is assumed that factors are reduced prior
		// to the int64Amount being created.
		return 0, false
	}
	return positiveScaleInt64(a.value, a.scale)
}

// AsScaledInt64 returns an int64 representing the value of this amount at the specified scale,
// rounding away from zero. ok is false when the value overflows int64, in which case the result
// saturates to mostNegative or mostPositive. (1e20).AsScaledInt64(1) overflows because 1e19 is not
// representable as an int64. Setting a scale larger than the current value may lose precision:
// (1e-6).AsScaledInt64(0) returns 1, because 0.000001 rounds away from zero to 1.
func (a int64Amount) AsScaledInt64(scale Scale) (result int64, ok bool) {
	if a.value == 0 {
		return 0, true
	}
	// Widen first: the delta of two int32 scales can need 33 bits.
	delta := a.widenedScale() - widenScale(scale)
	if delta < 0 {
		// Scaling down past 10^-log10MaxInt64 leaves only the rounding unit.
		down, fits := narrowScale(-delta)
		if !fits || down >= log10MaxInt64 {
			if a.value < 0 {
				return -1, true
			}
			return 1, true
		}
		result, _ = negativeScaleInt64(a.value, down)
		return result, true
	}
	// Scaling up: 10^log10MaxInt64 already exceeds int64.
	up, fits := narrowScale(delta)
	if !fits || up >= log10MaxInt64 {
		if a.value < 0 {
			return mostNegative, false
		}
		return mostPositive, false
	}
	return positiveScaleInt64(a.value, up)
}

// AsDec returns an inf.Dec representation of this value.
func (a int64Amount) AsDec() *inf.Dec {
	var base inf.Dec
	base.SetUnscaled(a.value)
	// TODO: handle !canInfScale case
	infScale, _ := a.scale.infScale()
	base.SetScale(infScale)
	return &base
}

// Cmp returns 0 if a and b are equal, 1 if a is greater than b, or -1 if a is less than b.
func (a int64Amount) Cmp(b int64Amount) int {
	switch {
	case a.scale == b.scale:
		// compare only the unscaled portion
	case a.scale > b.scale:
		// Widen before subtracting: the difference of two int32 scales does not
		// have to fit one, and a wrapped negative reaches a zero divisor.
		diff, fits := narrowScale(widenScale(a.scale) - widenScale(b.scale))
		if !fits || diff >= 18 {
			return cmpDec(a.AsDec(), b.AsDec())
		}
		result, remainder, exact := divideByScaleInt64(b.value, diff)
		if !exact {
			return cmpDec(a.AsDec(), b.AsDec())
		}
		if result == a.value {
			switch {
			case remainder == 0:
				return 0
			case remainder > 0:
				return -1
			default:
				return 1
			}
		}
		b.value = result
	default:
		diff, fits := narrowScale(widenScale(b.scale) - widenScale(a.scale))
		if !fits || diff >= 18 {
			return cmpDec(a.AsDec(), b.AsDec())
		}
		result, remainder, exact := divideByScaleInt64(a.value, diff)
		if !exact {
			return cmpDec(a.AsDec(), b.AsDec())
		}
		if result == b.value {
			switch {
			case remainder == 0:
				return 0
			case remainder > 0:
				return 1
			default:
				return -1
			}
		}
		a.value = result
	}

	switch {
	case a.value == b.value:
		return 0
	case a.value < b.value:
		return -1
	default:
		return 1
	}
}

// decimalExponentBounds brackets the e for which 10^(e-1) <= |c|*10^s < 10^e,
// for a non-zero c. An n-bit magnitude has at least n/4 and at most n/3+1
// decimal digits, so neither bound has to count them.
func decimalExponentBounds(bitLen int, s widenedScale) (lo, hi widenedScale) {
	n := widenedScale(bitLen)
	return n/4 + s, n/3 + 1 + s
}

// cmpDec compares x and y. inf.Dec.Cmp aligns the two scales by writing their
// difference out in digits, and a parsed exponent sets that difference, so
// settle what the magnitudes already decide before reaching it.
func cmpDec(x, y *inf.Dec) int {
	xSign, ySign := x.Sign(), y.Sign()
	switch {
	case xSign != ySign:
		if xSign > ySign {
			return 1
		}
		return -1
	case xSign == 0:
		return 0
	}
	xLo, xHi := decimalExponentBounds(x.UnscaledBig().BitLen(), widenInfScale(x.Scale()))
	yLo, yHi := decimalExponentBounds(y.UnscaledBig().BitLen(), widenInfScale(y.Scale()))
	switch {
	case xLo > yHi:
		return xSign
	case xHi < yLo:
		return -xSign
	}
	return x.Cmp(y)
}

// maxScaleGap bounds how many orders of magnitude may be between a non-zero
// coarser-scale operand and a finer-scale operand for Add and Sub without
// precision loss. If non-zero operands exceed this gap, the finer-scale operand
// is rounded away from zero to be within `maxScaleGap` of the coarser-scale
// operand. When either operand is zero, the non-zero operand's value and scale
// are preserved without rounding.
//
// This limit was selected for the following reasons:
//   - 29 digits is enough for the sum of two Quantities to be exact within
//     the int64 range (A single int64 value has at most 19 digits).
//   - Worst case running time is within 10x of normal expected running time.
const maxScaleGap = 29

// boundScaleGap returns x and y, replacing the finer-scale operand with a copy
// rounded away from zero when x and y are more than maxScaleGap orders of
// magnitude apart. Neither x nor y is mutated. Callers must handle zero
// operands before calling boundScaleGap.
func boundScaleGap(x, y *inf.Dec) (*inf.Dec, *inf.Dec) {
	xScale, yScale := widenInfScale(x.Scale()), widenInfScale(y.Scale())
	if xScale-yScale > maxScaleGap {
		rounded, _ := infDecAmount{y}.AsScale(Scale(xScale - maxScaleGap))
		return x, rounded.Dec
	}
	if yScale-xScale > maxScaleGap {
		rounded, _ := infDecAmount{x}.AsScale(Scale(yScale - maxScaleGap))
		return rounded.Dec, y
	}
	return x, y
}

// addDecBoundedScale adds x to y, but bounds the scale gap between the operands
// using boundScaleGap.
func addDecBoundedScale(x, y *inf.Dec) *inf.Dec {
	// short-circuit if one of the operands is zero
	if y.Sign() == 0 {
		return new(inf.Dec).Set(x)
	}
	if x.Sign() == 0 {
		return new(inf.Dec).Set(y)
	}

	// bound the scale between the operands if needed before adding
	x, y = boundScaleGap(x, y)
	return new(inf.Dec).Add(x, y)
}

// subDecBoundedScale subtracts y from x, but bounds the scale gap between the operands
// using boundScaleGap.
func subDecBoundedScale(x, y *inf.Dec) *inf.Dec {
	// short-circuit if one of the operands is zero
	if y.Sign() == 0 {
		return new(inf.Dec).Set(x)
	}
	if x.Sign() == 0 {
		return new(inf.Dec).Neg(y)
	}

	// bound the scale between the operands if needed before subtracting
	x, y = boundScaleGap(x, y)
	return new(inf.Dec).Sub(x, y)
}

// Add adds two int64Amounts together, matching scales. It will return false and not mutate
// a if overflow or underflow would result.
func (a *int64Amount) Add(b int64Amount) bool {
	switch {
	case b.value == 0:
		return true
	case a.value == 0:
		a.value = b.value
		a.scale = b.scale
		return true
	case a.scale == b.scale:
		c, ok := int64Add(a.value, b.value)
		if !ok {
			return false
		}
		a.value = c
	case a.scale > b.scale:
		// Widen first: the delta of two int32 scales can need 33 bits.
		up, fits := narrowScale(a.widenedScale() - b.widenedScale())
		if !fits {
			// a.value is non-zero (handled above), so this check is included for
			// clarity even though the result of the scale comparison would be
			// handled in the same way by positiveScaleInt64.
			return false
		}
		c, ok := positiveScaleInt64(a.value, up)
		if !ok {
			return false
		}
		c, ok = int64Add(c, b.value)
		if !ok {
			return false
		}
		a.scale = b.scale
		a.value = c
	default:
		// Widen first: the delta of two int32 scales can need 33 bits.
		up, fits := narrowScale(b.widenedScale() - a.widenedScale())
		if !fits {
			// b.value is non-zero (handled above), so this check is included for
			// clarity even though the result of the scale comparison would be
			// handled in the same way by positiveScaleInt64.
			return false
		}
		c, ok := positiveScaleInt64(b.value, up)
		if !ok {
			return false
		}
		c, ok = int64Add(a.value, c)
		if !ok {
			return false
		}
		a.value = c
	}
	return true
}

// Sub removes b from a. It returns false without mutating a when this
// subtraction cannot be performed safely on the int64 fast path and the caller
// can represent the operands in the decimal fallback.
func (a *int64Amount) Sub(b int64Amount) bool {
	if b.value == mostNegative && a.scale.canInfScale() && b.scale.canInfScale() {
		// -b.value overflows int64, so leave a unchanged and let the caller fall
		// back to inf.Dec when both scales can be represented there; otherwise try
		// the int64 path so same-scale MinInt32 subtraction keeps its existing
		// wrapped result.
		// TODO: Once int64Amount.AsDec handles math.MinInt32, drop the canInfScale
		// checks and always fall back to inf.Dec for mostNegative.
		return false
	}
	return a.Add(int64Amount{value: -b.value, scale: b.scale})
}

// Mul multiplies the provided b to the current amount, or
// returns false if overflow or underflow would result.
func (a *int64Amount) Mul(b int64) bool {
	switch {
	case a.value == 0:
		return true
	case b == 0:
		a.value = 0
		a.scale = 0
		return true
	case a.scale == 0:
		c, ok := int64Multiply(a.value, b)
		if !ok {
			return false
		}
		a.value = c
	case a.scale > 0:
		c, ok := int64Multiply(a.value, b)
		if !ok {
			return false
		}
		if _, ok = positiveScaleInt64(c, a.scale); !ok {
			return false
		}
		a.value = c
	default:
		c, ok := int64Multiply(a.value, b)
		if !ok {
			return false
		}
		if _, ok = negativeScaleInt64(c, -a.scale); !ok {
			return false
		}
		a.value = c
	}
	return true
}

// AsScale adjusts this amount to set a minimum scale, rounding away from zero,
// and returns true iff no precision was lost.
//
// For example:
//   - (1.1e5).AsScale(3) returns 1.1e5, true
//   - (1.1e5).AsScale(4) returns 1.1e5, true
//   - (1.1e5).AsScale(5) returns 2e5, false
//   - (1.1e5).AsScale(6) returns 1e6, false
//   - (-1.1e5).AsScale(5) returns -2e5, false
func (a int64Amount) AsScale(scale Scale) (int64Amount, bool) {
	if a.scale >= scale {
		return a, true
	}
	down, _ := narrowScale(widenScale(scale) - a.widenedScale())
	// We ignore the fits return value from narrowScale. negativeScaleInt64 scales an
	// int64 mantissa by no more than 19 digits, so if down is larger than 19, the
	// outcome is the same anyway, and it doesn't matter if it fit or got clamped.
	result, exact := negativeScaleInt64(a.value, down)
	return int64Amount{value: result, scale: scale}, exact
}

// widenedScale returns the base-10 scale exponent as a widenedScale.
func (a int64Amount) widenedScale() widenedScale {
	return widenScale(a.scale)
}

// floorToSIExponent returns the largest multiple of 3 that is less than or
// equal to w. fits is false when that multiple is outside the Scale range.
func floorToSIExponent(w widenedScale) (exponent Scale, fits bool) {
	switch w % 3 {
	case 1, -2:
		return narrowScale(w - 1)
	case 2, -1:
		return narrowScale(w - 2)
	default: // case 0
		return narrowScale(w)
	}
}

// AsCanonicalBytes accepts a buffer to write the base-10 string value of this
// field to, and returns either that buffer or a larger buffer and the current
// exponent of the value. The value is adjusted down until the exponent is a multiple
// of 3 to align with SI units. For example, 1.1e5 returns ("110", 3).
// math.MinInt32 and math.MinInt32+1 are returned as is because no lower
// multiple of 3 fits in an int32.
func (a int64Amount) AsCanonicalBytes(out []byte) (result []byte, exponent int32) {
	mantissa := a.value

	amount, times := removeInt64Factors(mantissa, 10)
	e := a.widenedScale() + widenedScale(times)
	c, fits := floorToSIExponent(e)
	if !fits {
		if e > 0 {
			return infDecAmount{a.AsDec()}.AsCanonicalBytes(out)
		}
		// 'times' is always positive, and the `e > 0` case above handles positive overflow,
		// so this means we could not round down to SI exponent without negative overflow, so keep e as-is
		if c, fits = narrowScale(e); !fits {
			assertUnreachable("int64Amount.AsCanonicalBytes: exponent does not fit in a Scale")
		}
	} else {
		// If the rounded-down exponent fits,
		// add the zeros that c no longer covers to the mantissa
		var ok bool
		switch e % 3 {
		case 0:
			// noop
		case 1, -2:
			amount, ok = int64MultiplyScale10(amount)
			if !ok {
				return infDecAmount{a.AsDec()}.AsCanonicalBytes(out)
			}
		case 2, -1:
			amount, ok = int64MultiplyScale100(amount)
			if !ok {
				return infDecAmount{a.AsDec()}.AsCanonicalBytes(out)
			}
		}
	}
	return strconv.AppendInt(out, amount, 10), int32(c)
}

// AsCanonicalBase1024Bytes accepts a buffer to write the base-1024 string value of this field to, and returns
// either that buffer or a larger buffer and the current exponent of the value. 2048 is 2 * 1024 ^ 1 and would
// return []byte("2048"), 1.
func (a int64Amount) AsCanonicalBase1024Bytes(out []byte) (result []byte, exponent int32) {
	value, ok := a.AsScaledInt64(0)
	if !ok {
		return infDecAmount{a.AsDec()}.AsCanonicalBase1024Bytes(out)
	}
	amount, exponent := removeInt64Factors(value, 1024)
	return strconv.AppendInt(out, amount, 10), exponent
}

// infDecAmount implements common operations over an inf.Dec that are specific to the quantity
// representation.
type infDecAmount struct {
	*inf.Dec
}

// AsScale adjusts this amount to set a minimum scale, rounding away from zero,
// and returns true iff no precision was lost.
//
// For example:
//   - (1.1e5).AsScale(3) returns 1.1e5, true
//   - (1.1e5).AsScale(4) returns 1.1e5, true
//   - (1.1e5).AsScale(5) returns 2e5, false
//   - (1.1e5).AsScale(6) returns 1e6, false
//   - (-1.1e5).AsScale(5) returns -2e5, false
func (a infDecAmount) AsScale(scale Scale) (infDecAmount, bool) {
	oldWidenedScale := a.widenedScale()
	newWidenedScale := widenScale(scale)
	infScale, canInfScale := scale.infScale()

	// Short-circuit if the scale is not being increased.
	// The scale.canInfScale() is check is included for clarity even
	// though the case is also caught by the scale comparison.
	if !canInfScale || oldWidenedScale >= newWidenedScale {
		return infDecAmount{new(inf.Dec).Set(a.Dec)}, true
	}

	sign := a.Dec.Sign()

	// Short-circuit if the value is zero, since we can skip the rounding.
	if sign == 0 {
		return infDecAmount{inf.NewDec(0, infScale)}, true
	}

	// Short-circuit if we can cheaply determine that the value is less than one unit of scale.
	_, hi := decimalExponentBounds(a.Dec.UnscaledBig().BitLen(), oldWidenedScale)
	if hi <= newWidenedScale {
		// The value is less than one unit of the desired scale, so to implement
		// AsScale, we round away from zero 1 or -1.
		return infDecAmount{inf.NewDec(int64(sign), infScale)}, false
	}

	tmp := &inf.Dec{}
	tmp.Round(a.Dec, infScale, inf.RoundUp)
	return infDecAmount{tmp}, tmp.Cmp(a.Dec) == 0
}

// widenedScale returns the base-10 scale exponent as a widenedScale.
func (a infDecAmount) widenedScale() widenedScale {
	return widenInfScale(a.Dec.Scale())
}

// AsCanonicalBytes accepts a buffer to write the base-10 string value of this
// field to, and returns either that buffer or a larger buffer and the current
// exponent of the value. The value is adjusted until the exponent is a multiple
// of 3 to align with SI units. For example, 1.1e5 returns ("110", 3).
// math.MinInt32 and math.MinInt32+1 are returned as is because no lower
// multiple of 3 fits in an int32.
func (a infDecAmount) AsCanonicalBytes(out []byte) (result []byte, exponent int32) {
	mantissa := a.Dec.UnscaledBig()
	amount := big.NewInt(0).Set(mantissa)
	// move all factors of 10 into the exponent for easy reasoning
	amount, times := removeBigIntFactors(amount, bigTen)
	e := a.widenedScale() + widenedScale(times)
	c, fits := floorToSIExponent(e)
	if !fits {
		if e > 0 {
			c = math.MaxInt32 - 1 // largest multiple of 3 in int32 is math.MaxInt32-1
		} else {
			// no multiple of 3 below e fits, so keep e
			if c, fits = narrowScale(e); !fits {
				assertUnreachable("infDecAmount.AsCanonicalBytes: exponent does not fit in a Scale")
			}
		}
	}

	// add the zeros that c no longer covers to the mantissa
	if c := widenScale(c); e > c {
		amount.Mul(amount, new(big.Int).Exp(bigTen, big.NewInt(int64(e-c)), nil))
	}

	return append(out, amount.String()...), int32(c)
}

// AsCanonicalBase1024Bytes accepts a buffer to write the base-1024 string value of this field to, and returns
// either that buffer or a larger buffer and the current exponent of the value. 2048 is 2 * 1024 ^ 1 and would
// return []byte("2048"), 1.
func (a infDecAmount) AsCanonicalBase1024Bytes(out []byte) (result []byte, exponent int32) {
	tmp := &inf.Dec{}
	tmp.Round(a.Dec, 0, inf.RoundUp)
	amount, exponent := removeBigIntFactors(tmp.UnscaledBig(), big1024)
	return append(out, amount.String()...), exponent
}
