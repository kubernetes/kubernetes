/*
Copyright The Kubernetes Authors.

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
	"testing"
)

// pow10Big returns 10^n as a *big.Int.
func pow10Big(n int) *big.Int {
	return new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(n)), nil)
}

// TestPositiveScaleInt64Saturation pins the int64-path saturation: an overflow
// lands on the rail matching the sign of base, and ok reports the overflow.
func TestPositiveScaleInt64Saturation(t *testing.T) {
	for _, tc := range []struct {
		base   int64
		scale  Scale
		want   int64
		wantOK bool
	}{
		{0, 100, 0, true},                     // zero cannot overflow
		{0, 2000000000, 0, true},              // zero short-circuits an extreme scale (no hang)
		{1, 18, 1000000000000000000, true},    // 1e18 fits
		{-1, 18, -1000000000000000000, true},  // -1e18 fits
		{mostPositive, 0, mostPositive, true}, // scale 0 is identity
		{mostPositive / 10, 1, (mostPositive / 10) * 10, true},
		{mostPositive, 1, mostPositive, false}, // was -10 before the fix
		{mostNegative, 1, mostNegative, false},
		{mostNegative, 2, mostNegative, false},
		{5, 20, mostPositive, false},  // positive overflow -> +rail
		{-5, 20, mostNegative, false}, // negative overflow -> -rail
		{-1, 19, mostNegative, false}, // -1e19 overflows
	} {
		got, ok := positiveScaleInt64(tc.base, tc.scale)
		if got != tc.want || ok != tc.wantOK {
			t.Errorf("positiveScaleInt64(%d, %d) = (%d, %t), want (%d, %t)", tc.base, tc.scale, got, ok, tc.want, tc.wantOK)
		}
	}
}

// TestScaledValueSaturationAndRounding pins the Dec-path (big.Int) behavior:
// away-from-zero rounding that matches negativeScaleInt64, and saturation with
// ok=false on overflow, across the fast and big scale-down paths and scale-up.
func TestScaledValueSaturationAndRounding(t *testing.T) {
	neg := func(b *big.Int) *big.Int { return new(big.Int).Neg(b) }
	add := func(b *big.Int, n int64) *big.Int { return new(big.Int).Add(b, big.NewInt(n)) }
	for _, tc := range []struct {
		name     string
		unscaled *big.Int
		scale    int64
		newScale int64
		want     int64
		wantOK   bool
	}{
		{"pos round away", big.NewInt(95), 1, 0, 10, true},   // 9.5 -> 10
		{"neg round away", big.NewInt(-95), 1, 0, -10, true}, // -9.5 -> -10
		{"pos exact", big.NewInt(100), 1, 0, 10, true},
		{"neg exact", big.NewInt(-100), 1, 0, -10, true},
		{"pos tiny fraction", big.NewInt(1), 3, 0, 1, true},                  // 0.001 -> 1
		{"neg tiny fraction", big.NewInt(-1), 3, 0, -1, true},                // -0.001 -> -1
		{"scale-down pos overflow", pow10Big(30), 3, 0, mostPositive, false}, // 1e27 > maxint
		{"scale-down neg overflow", neg(pow10Big(30)), 3, 0, mostNegative, false},
		{"scale-up pos overflow", pow10Big(30), 0, 3, mostPositive, false},
		{"scale-up neg overflow", neg(pow10Big(30)), 0, 3, mostNegative, false},
		{"identity pos overflow", pow10Big(19), 0, 0, mostPositive, false}, // 1e19 > maxint
		{"identity neg overflow", neg(pow10Big(19)), 0, 0, mostNegative, false},
		{"identity fits", big.NewInt(12345), 0, 0, 12345, true},
		// big scale-down path (unscaled exceeds int64) with a surviving quotient
		{"big-path pos round fits", add(pow10Big(20), 5), 5, 0, 1000000000000001, true},
		{"big-path neg round fits", neg(add(pow10Big(20), 5)), 5, 0, -1000000000000001, true},
		// extreme scales must stay bounded, not build a giant power of ten
		{"scale-up extreme pos saturates", big.NewInt(7), 0, 2000000000, mostPositive, false},
		{"scale-up extreme neg saturates", big.NewInt(-7), 0, 2000000000, mostNegative, false},
		{"scale-up extreme zero", big.NewInt(0), 0, 2000000000, 0, true},
		{"scale-down extreme pos rounds to one", big.NewInt(7), 2000000000, 0, 1, true},
		{"scale-down extreme neg rounds to minus one", big.NewInt(-7), 2000000000, 0, -1, true},
		{"scale-down extreme zero", big.NewInt(0), 2000000000, 0, 0, true},
	} {
		got, ok := scaledValue(tc.unscaled, tc.scale, tc.newScale)
		if got != tc.want || ok != tc.wantOK {
			t.Errorf("%s: scaledValue = (%d, %t), want (%d, %t)", tc.name, got, ok, tc.want, tc.wantOK)
		}
	}
}

// TestQuantityAsScaledInt64Boundaries exercises the public accessor and locks
// the headline case: a scaled quantity whose integer value overflows returns
// the saturated rail with ok=false instead of a wrapped value.
func TestQuantityAsScaledInt64Boundaries(t *testing.T) {
	for _, tc := range []struct {
		name   string
		q      *Quantity
		scale  Scale
		want   int64
		wantOK bool
	}{
		{"maxint x10 saturates", NewScaledQuantity(mostPositive, 1), 0, mostPositive, false},
		{"minint x10 saturates", NewScaledQuantity(mostNegative, 1), 0, mostNegative, false},
		{"exact value", NewScaledQuantity(12345, 0), 0, 12345, true},
		{"milli fits", NewScaledQuantity(5, 0), Milli, 5000, true},
		{"milli pos overflow saturates", NewScaledQuantity(mostPositive, 0), Milli, mostPositive, false},
		{"milli neg overflow saturates", NewScaledQuantity(mostNegative, 0), Milli, mostNegative, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := tc.q.AsScaledInt64(tc.scale)
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("AsScaledInt64(%d) = (%d, %t), want (%d, %t)", tc.scale, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

// TestQuantityValueSaturates is the regression guard: before the fix,
// NewScaledQuantity(mostPositive, 1).Value() wrapped to -10.
func TestQuantityValueSaturates(t *testing.T) {
	if got := NewScaledQuantity(mostPositive, 1).Value(); got != mostPositive {
		t.Errorf("NewScaledQuantity(mostPositive, 1).Value() = %d, want %d", got, int64(mostPositive))
	}
	if got := NewScaledQuantity(mostNegative, 1).Value(); got != mostNegative {
		t.Errorf("NewScaledQuantity(mostNegative, 1).Value() = %d, want %d", got, int64(mostNegative))
	}
	if got := NewScaledQuantity(mostPositive, 0).MilliValue(); got != mostPositive {
		t.Errorf("NewScaledQuantity(mostPositive, 0).MilliValue() = %d, want %d", got, int64(mostPositive))
	}
}

// TestQuantityScaledInt64BackendsAgree is the brute-force differential: for a
// dense grid of boundary values and scales, the int64 backend and the promoted
// inf.Dec backend must return the same (value, ok) from AsScaledInt64. This is
// what guarantees q.Value() and q.ToDec().Value() no longer disagree.
func TestQuantityScaledInt64BackendsAgree(t *testing.T) {
	values := []int64{
		0, 1, -1, 2, -2, 9, -9, 10, -10, 12345, -12345,
		999999999999999999, -999999999999999999,
		mostPositive, mostNegative, mostPositive - 1, mostNegative + 1,
		mostPositive / 10, mostNegative / 10,
		mostPositive / 1000, mostNegative / 1000,
		1 << 62, -(1 << 62),
	}
	for _, v := range values {
		for scale := Scale(-18); scale <= 18; scale++ {
			for target := Scale(-18); target <= 18; target++ {
				qi := NewScaledQuantity(v, scale)
				v1, ok1 := qi.AsScaledInt64(target)
				qi.ToDec()
				v2, ok2 := qi.AsScaledInt64(target)
				if v1 != v2 || ok1 != ok2 {
					t.Fatalf("backends disagree for value=%d scale=%d target=%d: int64=(%d,%t) dec=(%d,%t)",
						v, scale, target, v1, ok1, v2, ok2)
				}
			}
		}
	}
}

// TestQuantityDecOnlyOverflow covers unscaled magnitudes that cannot be built
// from an int64 (parsed decimals), where only the Dec path is exercised.
func TestQuantityDecOnlyOverflow(t *testing.T) {
	for _, tc := range []struct {
		in     string
		want   int64
		wantOK bool
	}{
		{"1e100", mostPositive, false},
		{"-1e100", mostNegative, false},
		{"9223372036854775807", mostPositive, true},   // exactly maxint
		{"-9223372036854775808", mostNegative, true},  // exactly minint
		{"9223372036854775808", mostPositive, false},  // maxint + 1
		{"-9223372036854775809", mostNegative, false}, // minint - 1
	} {
		q := MustParse(tc.in)
		q.ToDec()
		got, ok := q.AsScaledInt64(0)
		if got != tc.want || ok != tc.wantOK {
			t.Errorf("%q Dec AsScaledInt64(0) = (%d, %t), want (%d, %t)", tc.in, got, ok, tc.want, tc.wantOK)
		}
	}
}

// TestQuantityExtremeScaleStaysBounded drives the public accessor with scales in
// the billions. These must return promptly (the guards keep the int64 loop and
// the big.Int power bounded) and the two backends must still agree.
func TestQuantityExtremeScaleStaysBounded(t *testing.T) {
	for _, tc := range []struct {
		value  int64
		target Scale
	}{
		{7, 2000000000},
		{-7, 2000000000},
		{7, -2000000000},
		{-7, -2000000000},
		{0, 2000000000},
		{0, -2000000000},
		{mostPositive, -2000000000},
		{mostNegative, -2000000000},
	} {
		qi := NewScaledQuantity(tc.value, 0)
		v1, ok1 := qi.AsScaledInt64(tc.target)
		qi.ToDec()
		v2, ok2 := qi.AsScaledInt64(tc.target)
		if v1 != v2 || ok1 != ok2 {
			t.Errorf("value=%d target=%d: int64=(%d,%t) dec=(%d,%t)", tc.value, tc.target, v1, ok1, v2, ok2)
		}
	}
}

// FuzzQuantityScaledInt64BackendsAgree drives the same differential with random
// values and scales so the int64 and Dec backends are checked far past the
// hand-picked grid.
func FuzzQuantityScaledInt64BackendsAgree(f *testing.F) {
	f.Add(int64(0), int8(0), int8(0))
	f.Add(int64(mostPositive), int8(1), int8(0))
	f.Add(int64(mostNegative), int8(1), int8(0))
	f.Add(int64(-95), int8(1), int8(0))
	f.Fuzz(func(t *testing.T, value int64, scaleByte, targetByte int8) {
		scale := Scale(int(scaleByte) % 40)
		target := Scale(int(targetByte) % 40)
		qi := NewScaledQuantity(value, scale)
		v1, ok1 := qi.AsScaledInt64(target)
		qi.ToDec()
		v2, ok2 := qi.AsScaledInt64(target)
		if v1 != v2 || ok1 != ok2 {
			t.Fatalf("backends disagree: value=%d scale=%d target=%d int64=(%d,%t) dec=(%d,%t)",
				value, scale, target, v1, ok1, v2, ok2)
		}
	})
}

// TestQuantityAsScaledInt64ExtremeScales covers scale deltas that overflow int32
// when computed the naive way: a source and target near opposite ends of the
// range, and the int32 endpoints themselves.
func TestQuantityAsScaledInt64ExtremeScales(t *testing.T) {
	for _, tc := range []struct {
		name   string
		value  int64
		scale  Scale
		target Scale
		want   int64
		wantOK bool
	}{
		{"neg-source-pos-target-rounds-up", 7, Scale(-2000000000), Scale(2000000000), 1, true},
		{"neg-source-pos-target-negative", -7, Scale(-2000000000), Scale(2000000000), -1, true},
		{"pos-source-neg-target-saturates", 7, Scale(2000000000), Scale(-2000000000), mostPositive, false},
		{"neg-source-neg-target-saturates", -7, Scale(2000000000), Scale(-2000000000), mostNegative, false},
		{"minint32-target-saturates", 7, 0, Scale(math.MinInt32), mostPositive, false},
		{"maxint32-target-rounds-up", 7, 0, Scale(math.MaxInt32), 1, true},
		{"minint32-source-rounds-up", 1, Scale(math.MinInt32), 0, 1, true},
		{"zero-value-any-scale", 0, Scale(math.MinInt32), Scale(math.MaxInt32), 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := NewScaledQuantity(tc.value, tc.scale).AsScaledInt64(tc.target)
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("value=%d scale=%d target=%d: got (%d,%t), want (%d,%t)",
					tc.value, tc.scale, tc.target, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

// TestQuantityDecExtremeTarget checks that an int32-endpoint target scale on the
// Dec backend is handled without the infScale() negation overflow.
func TestQuantityDecExtremeTarget(t *testing.T) {
	for _, tc := range []struct {
		in     string
		target Scale
		want   int64
		wantOK bool
	}{
		{"7", Scale(math.MinInt32), mostPositive, false},
		{"7", Scale(math.MaxInt32), 1, true},
		{"-7", Scale(math.MinInt32), mostNegative, false},
	} {
		q := MustParse(tc.in)
		q.ToDec()
		got, ok := q.AsScaledInt64(tc.target)
		if got != tc.want || ok != tc.wantOK {
			t.Errorf("%q ToDec().AsScaledInt64(%d) = (%d,%t), want (%d,%t)", tc.in, tc.target, got, ok, tc.want, tc.wantOK)
		}
	}
}

// TestQuantityWrapperParity locks the wrappers to their checked accessors on both
// backends, and exercises AsMilliInt64 directly.
func TestQuantityWrapperParity(t *testing.T) {
	for _, vs := range []struct {
		v int64
		s Scale
	}{
		{0, 0}, {5, 0}, {-5, 0}, {12345, 0},
		{mostPositive, 0}, {mostNegative, 0}, {mostPositive, 1}, {mostNegative, 1},
		{7, -6}, {-7, -6},
	} {
		for _, dec := range []bool{false, true} {
			q := NewScaledQuantity(vs.v, vs.s)
			if dec {
				q.ToDec()
			}
			if v, _ := q.AsScaledInt64(0); q.Value() != v {
				t.Errorf("dec=%t value=%d scale=%d: Value()=%d, AsScaledInt64(0)=%d", dec, vs.v, vs.s, q.Value(), v)
			}
			milli, mok := q.AsMilliInt64()
			if s, sok := q.AsScaledInt64(Milli); milli != s || mok != sok {
				t.Errorf("dec=%t value=%d scale=%d: AsMilliInt64()=(%d,%t), AsScaledInt64(Milli)=(%d,%t)", dec, vs.v, vs.s, milli, mok, s, sok)
			}
			if q.MilliValue() != milli {
				t.Errorf("dec=%t value=%d scale=%d: MilliValue()=%d, AsMilliInt64()=%d", dec, vs.v, vs.s, q.MilliValue(), milli)
			}
			if v, _ := q.AsScaledInt64(Kilo); q.ScaledValue(Kilo) != v {
				t.Errorf("dec=%t value=%d scale=%d: ScaledValue(Kilo)=%d, AsScaledInt64(Kilo)=%d", dec, vs.v, vs.s, q.ScaledValue(Kilo), v)
			}
		}
	}
}
