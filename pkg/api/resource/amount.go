/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"speter.net/go/exp/math/dec/inf"
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

	// MaxInt64Factors is the highest value that will be checked when removing factors of 10 from an int64
	maxInt64Factors = 18
)

var (
	Zero = int64Amount{}

	// Used by quantity strings - treat as read only
	zeroBytes = []byte("0")

	// Commonly needed big.Int values-- treat as read only!
	bigTen      = big.NewInt(10)
	bigZero     = big.NewInt(0)
	bigOne      = big.NewInt(1)
	bigThousand = big.NewInt(1000)
	big1024     = big.NewInt(1024)

	// Commonly needed inf.Dec values-- treat as read only!
	decZero      = inf.NewDec(0, 0)
	decOne       = inf.NewDec(1, 0)
	decMinusOne  = inf.NewDec(-1, 0)
	decThousand  = inf.NewDec(1000, 0)
	dec1024      = inf.NewDec(1024, 0)
	decMinus1024 = inf.NewDec(-1024, 0)

	// Largest (in magnitude) number allowed.
	maxAllowed = infDecAmount{inf.NewDec((1<<63)-1, 0)} // == max int64

	// The maximum value we can represent milli-units for.
	// Compare with the return value of Quantity.Value() to
	// see if it's safe to use Quantity.MilliValue().
	MaxMilliValue = int64(((1 << 63) - 1) / 1000)
)

type int64Amount struct {
	value int64
	scale Scale
}

func (a int64Amount) Sign() int {
	switch {
	case a.value > 0:
		return 1
	case a.value < 0:
		return -1
	default:
		return 0
	}
}

const mostNegative = -(mostPositive + 1)
const mostPositive = 1<<63 - 1

func int64Add(a, b int64) (int64, bool) {
	c := a + b
	switch {
	case a > 0 && b > 0:
		if c < 0 {
			return 0, false
		}
	case a < 0 && b < 0:
		if c > 0 {
			return 0, false
		}
	}
	return c, true
}

func int64Multiply(a, b int64) (int64, bool) {
	if a == 0 || b == 0 || a == 1 || b == 1 {
		return a * b, true
	}
	if a == mostNegative || b == mostNegative {
		return 0, false
	}
	c := a * b
	return c, c/b == a
}

func int64MultiplyScale(a int64, b int64) (int64, bool) {
	if a == 0 || a == 1 {
		return a * b, true
	}
	if a == mostNegative {
		return 0, false
	}
	c := a * b
	return c, c/b == a
}

func int64MultiplyScale10(a int64) (int64, bool) {
	if a == 0 || a == 1 {
		return a * 10, true
	}
	if a == mostNegative {
		return 0, false
	}
	c := a * 10
	return c, c/10 == a
}

func int64MultiplyScale100(a int64) (int64, bool) {
	if a == 0 || a == 1 {
		return a * 100, true
	}
	if a == mostNegative {
		return 0, false
	}
	c := a * 100
	return c, c/100 == a
}

func int64MultiplyScale1000(a int64) (int64, bool) {
	if a == 0 || a == 1 {
		return a * 1000, true
	}
	if a == mostNegative {
		return 0, false
	}
	c := a * 1000
	return c, c/1000 == a
}

// positiveScaleInt64 multiplies base by 10^scale, returning false if the
// value overflows.
func positiveScaleInt64(base int64, scale Scale) (int64, bool) {
	switch scale {
	case 0:
		return base, true
	case 1:
		return int64MultiplyScale10(base)
	case 2:
		return int64MultiplyScale100(base)
	case 3:
		return int64MultiplyScale1000(base)
	case 6:
		return int64MultiplyScale(base, 1000000)
	case 9:
		return int64MultiplyScale(base, 1000000000)
	default:
		value := base
		var ok bool
		for i := Scale(0); i < scale; i++ {
			if value, ok = int64MultiplyScale(value, 10); !ok {
				return 0, false
			}
		}
		return value, true
	}
}

// negativeScaleInt64 reduces base by the provided scale, rounding up, until the
// value is zero or the scale is reached.
func negativeScaleInt64(base int64, scale Scale) (result int64, exact bool) {
	switch scale {
	case 0:
		return base, true
	default:
		value := base
		var fraction bool
		for i := Scale(0); i < scale; i++ {
			if !fraction && value%10 != 0 {
				fraction = true
			}
			value = value / 10
			if value == 0 {
				if fraction {
					return 1, false
				}
				return 0, true
			}
		}
		return value, !fraction
	}
}

func (a int64Amount) AsInt64() (int64, bool) {
	if a.scale == 0 {
		return a.value, true
	}
	if a.scale < 0 {
		return 0, false
	}
	return positiveScaleInt64(a.value, a.scale)
}

func (a int64Amount) AsScaledInt64(scale Scale) (result int64, ok bool) {
	if a.scale < scale {
		result, _ = negativeScaleInt64(a.value, scale-a.scale)
		return result, true
	}
	return positiveScaleInt64(a.value, a.scale-scale)
}

func (a int64Amount) AsDec() *inf.Dec {
	var base inf.Dec
	base.SetUnscaled(a.value)
	base.SetScale(inf.Scale(-a.scale))
	return &base
}

func (a int64Amount) Cmp(b int64Amount) int {
	switch {
	case a.scale == b.scale:
		// compare only the unscaled portion
	case a.scale > b.scale:
		b.value, _ = negativeScaleInt64(b.value, a.scale-b.scale)
	default:
		a.value, _ = negativeScaleInt64(a.value, b.scale-a.scale)
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

func (a *int64Amount) Sub(b int64Amount) bool {
	return a.Add(int64Amount{value: -b.value, scale: b.scale})
}

func (a int64Amount) AsScale(scale Scale) (int64Amount, bool) {
	if a.scale >= scale {
		return a, true
	}
	result, exact := negativeScaleInt64(a.value, scale-a.scale)
	return int64Amount{value: result, scale: scale}, exact
}

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

func (a int64Amount) AsCanonicalBase1024Bytes(out []byte) (result []byte, exponent int32) {
	value, ok := a.AsScaledInt64(0)
	if !ok {
		return infDecAmount{a.AsDec()}.AsCanonicalBase1024Bytes(out)
	}
	amount, exponent := removeInt64Factors(value, 1024)
	return strconv.AppendInt(out, amount, 10), exponent
}

type infDecAmount struct {
	*inf.Dec
}

func (a infDecAmount) AsScale(scale Scale) (infDecAmount, bool) {
	tmp := &inf.Dec{}
	tmp.Round(a.Dec, 0, inf.RoundUp)
	return infDecAmount{tmp}, tmp.Cmp(a.Dec) == 0
}

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

func (a infDecAmount) AsCanonicalBase1024Bytes(out []byte) (result []byte, exponent int32) {
	tmp := &inf.Dec{}
	tmp.Round(a.Dec, 0, inf.RoundUp)
	amount, exponent := removeBigIntFactors(tmp.UnscaledBig(), big1024)
	return append(out, amount.String()...), exponent
}

// removeInt64Factors divides in a loop; the return values have the property that
// value == result * base ^ scale
func removeInt64Factors(value int64, base int64) (result int64, times int32) {
	times = 0
	result = value
	negative := result < 0
	if negative {
		result = -result
	}
	switch base {
	// allow the compiler to optimize the common cases
	case 10:
		for result >= 10 && result%10 == 0 {
			times++
			result = result / 10
		}
	// allow the compiler to optimize the common cases
	case 1024:
		for result >= 1024 && result%1024 == 0 {
			times++
			result = result / 1024
		}
	default:
		for result >= base && result%base == 0 {
			times++
			result = result / base
		}
	}
	if negative {
		result = -result
	}
	return result, times
}

// removeBigIntFactors divides in a loop; the return values have the property that
// d == result * factor ^ times
// d may be modified in place.
// If d == 0, then the return values will be (0, 0)
func removeBigIntFactors(d, factor *big.Int) (result *big.Int, times int32) {
	q := big.NewInt(0)
	m := big.NewInt(0)
	for d.Cmp(bigZero) != 0 {
		q.DivMod(d, factor, m)
		if m.Cmp(bigZero) != 0 {
			break
		}
		times++
		d, q = q, d
	}
	return d, times
}
