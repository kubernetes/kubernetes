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
	"math/big"
	"strconv"

	inf "gopkg.in/inf.v0"
)

// Scale is used for getting and setting the base-10 scaled value.
// Base-2 scales are omitted for mathematical simplicity.
// See Quantity.ScaledValue for more details.
type Scale int32

// infScale adapts a Scale value to an inf.Scale value.
func (s Scale) infScale() inf.Scale {
	return inf.Scale(-s) // inf.Scale is upside-down
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
// rounding up, or false if that would result in overflow. (1e20).AsScaledInt64(1) would result
// in overflow because 1e19 is not representable as an int64. Note that setting a scale larger
// than the current value may result in loss of precision - i.e. (1e-6).AsScaledInt64(0) would
// return 1, because 0.000001 is rounded up to 1.
func (a int64Amount) AsScaledInt64(scale Scale) (result int64, ok bool) {
	if a.scale < scale {
		result, _ = negativeScaleInt64(a.value, scale-a.scale)
		return result, true
	}
	return positiveScaleInt64(a.value, a.scale-scale)
}

// AsDec returns an inf.Dec representation of this value.
func (a int64Amount) AsDec() *inf.Dec {
	var base inf.Dec
	base.SetUnscaled(a.value)
	base.SetScale(inf.Scale(-a.scale))
	return &base
}

// Cmp returns 0 if a and b are equal, 1 if a is greater than b, or -1 if a is less than b.
func (a int64Amount) Cmp(b int64Amount) int {
	switch {
	case a.scale == b.scale:
		// compare only the unscaled portion
	case a.scale > b.scale:
		result, remainder, exact := divideByScaleInt64(b.value, a.scale-b.scale)
		if !exact {
			return a.AsDec().Cmp(b.AsDec())
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
		result, remainder, exact := divideByScaleInt64(a.value, b.scale-a.scale)
		if !exact {
			return a.AsDec().Cmp(b.AsDec())
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
		c, ok := positiveScaleInt64(a.value, a.scale-b.scale)
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
		c, ok := positiveScaleInt64(b.value, b.scale-a.scale)
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

// Sub removes the value of b from the current amount, or returns false if underflow would result.
func (a *int64Amount) Sub(b int64Amount) bool {
	return a.Add(int64Amount{value: -b.value, scale: b.scale})
}

// AsScale adjusts this amount to set a minimum scale, rounding up, and returns true iff no precision
// was lost. (1.1e5).AsScale(5) would return 1.1e5, but (1.1e5).AsScale(6) would return 1e6.
func (a int64Amount) AsScale(scale Scale) (int64Amount, bool) {
	if a.scale >= scale {
		return a, true
	}
	result, exact := negativeScaleInt64(a.value, scale-a.scale)
	return int64Amount{value: result, scale: scale}, exact
}

// AsCanonicalBytes accepts a buffer to write the base-10 string value of this field to, and returns
// either that buffer or a larger buffer and the current exponent of the value. The value is adjusted
// until the exponent is a multiple of 3 - i.e. 1.1e5 would return "110", 3.
func (a int64Amount) AsCanonicalBytes(out []byte) (result []byte, exponent int32) {
	mantissa := a.value
	exponent = int32(a.scale)

	amount, times := removeInt64Factors(mantissa, 10)
	exponent += int32(times)

	// make sure exponent is a multiple of 3
	var ok bool
	switch exponent % 3 {
	case 1, -2:
		amount, ok = int64MultiplyScale10(amount)
		if !ok {
			return infDecAmount{a.AsDec()}.AsCanonicalBytes(out)
		}
		exponent = exponent - 1
	case 2, -1:
		amount, ok = int64MultiplyScale100(amount)
		if !ok {
			return infDecAmount{a.AsDec()}.AsCanonicalBytes(out)
		}
		exponent = exponent - 2
	}
	return strconv.AppendInt(out, amount, 10), exponent
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

// AsScale adjusts this amount to set a minimum scale, rounding up, and returns true iff no precision
// was lost. (1.1e5).AsScale(5) would return 1.1e5, but (1.1e5).AsScale(6) would return 1e6.
func (a infDecAmount) AsScale(scale Scale) (infDecAmount, bool) {
	tmp := &inf.Dec{}
	tmp.Round(a.Dec, scale.infScale(), inf.RoundUp)
	return infDecAmount{tmp}, tmp.Cmp(a.Dec) == 0
}

// AsCanonicalBytes accepts a buffer to write the base-10 string value of this field to, and returns
// either that buffer or a larger buffer and the current exponent of the value. The value is adjusted
// until the exponent is a multiple of 3 - i.e. 1.1e5 would return "110", 3.
func (a infDecAmount) AsCanonicalBytes(out []byte) (result []byte, exponent int32) {
	mantissa := a.Dec.UnscaledBig()
	exponent = int32(-a.Dec.Scale())
	amount := big.NewInt(0).Set(mantissa)
	// move all factors of 10 into the exponent for easy reasoning
	amount, times := removeBigIntFactors(amount, bigTen)
	exponent += times

	// make sure exponent is a multiple of 3
	for exponent%3 != 0 {
		amount.Mul(amount, bigTen)
		exponent--
	}

	return append(out, amount.String()...), exponent
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
