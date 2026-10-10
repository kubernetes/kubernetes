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

package exact

import (
	"errors"
	"math"
	"math/big"
	"strings"
	"testing"
)

func TestParseAndCanonical(t *testing.T) {
	for _, tc := range []struct {
		in, value, canonical, format string
	}{
		{"0", "0", "0", DecimalSI},
		{"1.5", "15e-1", "1500m", DecimalSI},
		{"1.5Gi", "1610612736e0", "1536Mi", BinarySI},
		{"0.5Ki", "512e0", "512", BinarySI},
		{"0.0001Ki", "1024e-4", "102400u", DecimalSI},
		{"0.1n", "1e-9", "1n", DecimalSI},
		{"-0.1n", "-1e-9", "-1n", DecimalSI},
		{"8Ei", "9223372036854775807e0", "9223372036854775807", BinarySI},
		{"1e2147483648", "1e2147483648", "100e2147483646", DecimalExponent},
		{"7e-2147483648", "1e-9", "1e-9", DecimalExponent},
		{"1e-9223372036854775808", "1e-9", "1e-9", DecimalExponent},
		{"12e6", "12e6", "12e6", DecimalExponent},
		{"1000k", "1e6", "1M", DecimalSI},
		{"10000000Ki", "1024e7", "10000000Ki", BinarySI},
	} {
		q, err := Parse(tc.in)
		if err != nil {
			t.Errorf("Parse(%q): %v", tc.in, err)
			continue
		}
		if got := q.ExactString(); got != tc.value {
			t.Errorf("Parse(%q) = %s, want %s", tc.in, got, tc.value)
		}
		if got := q.String(); got != tc.canonical {
			t.Errorf("Parse(%q).String() = %q, want %q", tc.in, got, tc.canonical)
		}
		if got := q.Format(); got != tc.format {
			t.Errorf("Parse(%q).Format() = %q, want %q", tc.in, got, tc.format)
		}
	}
	for _, in := range []string{"", ".", "-", "e3", "1..5", "1.1.M", "1ki", "1e", "1e+-3", "1K", "0x10"} {
		if _, err := Parse(in); !errors.Is(err, ErrFormatWrong) {
			t.Errorf("Parse(%q) error = %v, want ErrFormatWrong", in, err)
		}
	}
	if _, err := Parse("1e99999999999999999999"); !errors.Is(err, ErrOutOfDomain) {
		t.Errorf("Parse(1e99999999999999999999) error = %v, want ErrOutOfDomain", err)
	}
}

func TestArithmetic(t *testing.T) {
	huge, tiny := New(big.NewInt(1), math.MaxInt32, DecimalExponent), New(big.NewInt(7), math.MinInt32, DecimalExponent)
	if huge.Cmp(tiny) != 1 || tiny.Cmp(huge) != -1 || tiny.CmpInt64(1) != -1 {
		t.Errorf("Cmp across the int32 edges is wrong")
	}
	if _, ok := huge.Add(tiny); ok {
		t.Errorf("Add across the int32 edges should report that it cannot align")
	}
	sum, ok := MustParse("1.5").Add(MustParse("-2"))
	if !ok || sum.ExactString() != "-5e-1" {
		t.Errorf("1.5 + -2 = %s, %t", sum.ExactString(), ok)
	}
	if v, ok := MustParse("1.5").AsScaledInt64(0); v != 2 || !ok {
		t.Errorf("AsScaledInt64(0) of 1.5 = %d, %t", v, ok)
	}
	if v, ok := MustParse("-1.5").AsScaledInt64(0); v != -2 || !ok {
		t.Errorf("AsScaledInt64(0) of -1.5 = %d, %t", v, ok)
	}
	if v, ok := MustParse("1e19").AsInt64(); v != math.MaxInt64 || ok {
		t.Errorf("AsInt64 of 1e19 = %d, %t", v, ok)
	}
	if r, ok := MustParse("1.5").Rat(); !ok || r.Cmp(big.NewRat(3, 2)) != 0 {
		t.Errorf("Rat of 1.5 = %v, %t", r, ok)
	}
	if _, ok := huge.Rat(); ok {
		t.Errorf("Rat of 1e2147483647 should be unavailable")
	}
	if f := MustParse("1.5").AsApproximateFloat64(); f != 1.5 {
		t.Errorf("AsApproximateFloat64 of 1.5 = %v", f)
	}
}

// TestEdges checks the operations that work on values beyond big.Rat, at the
// int32 edges of resource.Quantity's scale, and the boundary between the two.
func TestEdges(t *testing.T) {
	if _, ok := MustParse("1e1000000").Rat(); !ok {
		t.Errorf("Rat of 1e1000000 should be available")
	}
	if _, ok := MustParse("1e1000001").Rat(); ok {
		t.Errorf("Rat of 1e1000001 should be unavailable")
	}
	if MustParse("1e1000001").Cmp(MustParse("1e1000000")) != 1 || MustParse("-1e1000001").Cmp(MustParse("-1e1000000")) != -1 {
		t.Errorf("Cmp across RatLimit is wrong")
	}
	huge := New(big.NewInt(9), math.MaxInt32, DecimalExponent)
	tiny := New(big.NewInt(-7), math.MinInt32, DecimalExponent)
	for _, tc := range []struct {
		q       Quantity
		scale   int32
		v       int64
		ok      bool
		rounded string
		exact   bool
	}{
		{huge, 0, math.MaxInt64, false, "9e2147483647", true},
		{huge, -9, math.MaxInt64, false, "9e2147483647", true},
		{huge.Neg(), 3, math.MinInt64, false, "-9e2147483647", true},
		{tiny, 0, -1, true, "-1e0", false},
		{tiny, -9, -1, true, "-1e-9", false},
		{tiny.Neg(), 18, 1, true, "1e18", false},
	} {
		if v, ok := tc.q.AsScaledInt64(tc.scale); v != tc.v || ok != tc.ok {
			t.Errorf("%s.AsScaledInt64(%d) = %d, %t, want %d, %t", tc.q.ExactString(), tc.scale, v, ok, tc.v, tc.ok)
		}
		if r, exact := tc.q.RoundToScale(tc.scale); r.ExactString() != tc.rounded || exact != tc.exact {
			t.Errorf("%s.RoundToScale(%d) = %s, %t, want %s, %t", tc.q.ExactString(), tc.scale, r.ExactString(), exact, tc.rounded, tc.exact)
		}
	}
	if f := huge.AsApproximateFloat64(); !math.IsInf(f, 1) {
		t.Errorf("AsApproximateFloat64 of %s = %v", huge.ExactString(), f)
	}
	// A mantissa of a million digits brings an exponent beyond -RatLimit back
	// into the float64 range.
	long, _ := new(big.Int).SetString("1"+strings.Repeat("0", RatLimit+5), 10)
	if f := New(long, -RatLimit-3, DecimalSI).AsApproximateFloat64(); f != 100 {
		t.Errorf("AsApproximateFloat64 of 10^%d * 10^%d = %v, want 100", RatLimit+5, -RatLimit-3, f)
	}
	if f := tiny.AsApproximateFloat64(); f != 0 || !math.Signbit(f) {
		t.Errorf("AsApproximateFloat64 of %s = %v", tiny.ExactString(), f)
	}
	// Around ±RatLimit, fraction digits move the exponent across it, and a
	// long enough mantissa brings a value with an exponent below -RatLimit up
	// to 1n or more.
	zeros := strings.Repeat("0", RatLimit)
	for _, tc := range []struct{ name, in, value string }{
		{"1.5e-1000000", "1.5e-1000000", "1e-9"},
		{"-0.1e-1000000", "-0.1e-1000000", "-1e-9"},
		{"12.34e-999999", "12.34e-999999", "1e-9"},
		{"0.1e1000001", "0.1e1000001", "1e1000000"},
		{"12.5e1000000", "12.5e1000000", "125e999999"},
		{"0.<10^6 zeros>1", "0." + zeros + "1", "1e-9"},
		{"1234.<10^6 zeros>5", "1234." + zeros + "5", "1234000000001e-9"},
		{"1.<10^6 zeros>5Ki", "1." + zeros + "5Ki", "1024000000001e-9"},
		{"1<10^6+10 zeros>e-1000001", "1" + zeros + "0000000000e-1000001", "1e9"},
	} {
		if got := MustParse(tc.in).ExactString(); got != tc.value {
			t.Errorf("Parse(%s) = %s, want %s", tc.name, got, tc.value)
		}
	}
}

// TestFormat checks the formats that follow resource.Quantity's rules: Add on
// a zero receiver takes the other operand's format, and the zero value has no
// format and serializes without a suffix.
func TestFormat(t *testing.T) {
	if s, _ := MustParse("0").Add(MustParse("2Ki")); s.Format() != BinarySI {
		t.Errorf("0 + 2Ki format = %q, want BinarySI", s.Format())
	}
	if s, _ := MustParse("0").Sub(MustParse("2Ki")); s.Format() != BinarySI || s.ExactString() != "-2048e0" {
		t.Errorf("0 - 2Ki = %s %q, want -2048e0 BinarySI", s.ExactString(), s.Format())
	}
	if s, _ := MustParse("2Ki").Add(MustParse("0")); s.Format() != BinarySI {
		t.Errorf("2Ki + 0 format = %q, want BinarySI", s.Format())
	}
	if p := New(big.NewInt(7), math.MinInt32, DecimalExponent).Mul(-30); p.ExactString() != "-21e-2147483647" {
		t.Errorf("7e-2147483648 * -30 = %s", p.ExactString())
	}
	if p := MustParse("1.5Ki").Mul(0); !p.IsZero() || p.Format() != BinarySI {
		t.Errorf("1.5Ki * 0 = %s %q", p.ExactString(), p.Format())
	}
	var zero Quantity
	if zero.Format() != "" || zero.String() != "0" {
		t.Errorf("zero value: format %q, string %q", zero.Format(), zero.String())
	}
	if got := New(big.NewInt(1500), -3, "").String(); got != "1500e-3" {
		t.Errorf("1.5 with no format = %q, want 1500e-3", got)
	}
}

// TestDomain checks that values outside ±MaxExponent are refused rather than
// computed with an exponent that wrapped.
func TestDomain(t *testing.T) {
	for _, tc := range []struct {
		mantissa int64
		exponent int64
	}{{10, MaxExponent}, {1, MaxExponent + 1}, {1, -MaxExponent - 1}, {10, -MaxExponent - 2}, {10, math.MaxInt64}, {1, math.MinInt64}} {
		func() {
			defer func() {
				if _, ok := recover().(Unjudgeable); !ok {
					t.Errorf("New(%d, %d) did not panic with Unjudgeable", tc.mantissa, tc.exponent)
				}
			}()
			New(big.NewInt(tc.mantissa), tc.exponent, DecimalSI)
		}()
	}
	if got := New(big.NewInt(1), MaxExponent, DecimalSI).ExactString(); got != "1e1099511627776" {
		t.Errorf("New(1, MaxExponent) = %s", got)
	}
	// The domain is on the value's exponent, after trailing zeros move it.
	if got := New(big.NewInt(10), -MaxExponent-1, DecimalSI).ExactString(); got != "1e-1099511627776" {
		t.Errorf("New(10, -MaxExponent-1) = %s", got)
	}
	if _, err := Parse("1000e1099511627776"); !errors.Is(err, ErrOutOfDomain) {
		t.Errorf("Parse(1000e2^40) error = %v, want ErrOutOfDomain", err)
	}
}

// TestInt64Rails checks both ends of the int64 range, held as a big.Rat and
// held as a mantissa and exponent.
func TestInt64Rails(t *testing.T) {
	maxPlus := new(big.Int).Add(big.NewInt(math.MaxInt64), big.NewInt(1))
	minMinus := new(big.Int).Sub(big.NewInt(math.MinInt64), big.NewInt(1))
	for _, scale := range []int64{-10, -2_000_000, 2_000_000} {
		for _, tc := range []struct {
			mantissa *big.Int
			v        int64
			ok       bool
		}{
			{big.NewInt(math.MaxInt64), math.MaxInt64, true},
			{big.NewInt(math.MinInt64), math.MinInt64, true},
			{maxPlus, math.MaxInt64, false},
			{minMinus, math.MinInt64, false},
		} {
			q := New(tc.mantissa, scale, DecimalSI)
			if v, ok := q.AsScaledInt64(int32(scale)); v != tc.v || ok != tc.ok {
				t.Errorf("%s.AsScaledInt64(%d) = %d, %t, want %d, %t", q.ExactString(), scale, v, ok, tc.v, tc.ok)
			}
		}
	}
	if v, ok := New(big.NewInt(15), math.MaxInt32, DecimalSI).AsScaledInt64(math.MaxInt32); v != 15 || !ok {
		t.Errorf("15e2147483647.AsScaledInt64(MaxInt32) = %d, %t", v, ok)
	}
	if r, exact := New(big.NewInt(math.MaxInt64), -2_000_000, DecimalSI).RoundToScale(-1_999_999); r.ExactString() != "922337203685477581e-1999999" || exact {
		t.Errorf("RoundToScale across the edge = %s, %t", r.ExactString(), exact)
	}
}

// TestRepresentation checks that the representation of a value does not
// depend on how it was computed.
func TestRepresentation(t *testing.T) {
	sum, _ := MustParse("9e1000000").Add(MustParse("1e1000000"))
	for _, q := range []Quantity{MustParse("1e1000001"), New(big.NewInt(10), RatLimit, DecimalSI), sum} {
		if _, ok := q.Rat(); ok {
			t.Errorf("%s is held as a big.Rat", q.ExactString())
		}
	}
	diff, _ := MustParse("2e1000000").Sub(MustParse("1e1000000"))
	if _, ok := diff.Rat(); !ok {
		t.Errorf("%s is not held as a big.Rat", diff.ExactString())
	}
}

// propertyValues spans both representations, both signs and both int32 edges.
func propertyValues() []Quantity {
	vs := []Quantity{{}}
	for _, s := range []string{"1", "1.5", "1n", "1023", "1024", "1Ki", "1.5Gi", "8Ei", "9223372036854775807",
		"123456789e-30", "9e1000000", "1e1000001", "1e2147483647", "7e2147483646", "99e-1000000"} {
		q := MustParse(s)
		vs = append(vs, q, q.Neg())
	}
	for _, e := range []int64{math.MinInt32, math.MinInt32 + 1, -RatLimit - 1, RatLimit + 1, math.MaxInt32} {
		q := New(big.NewInt(7), e, DecimalExponent)
		vs = append(vs, q, q.Neg())
	}
	return vs
}

// TestProperties checks package exact against itself: Cmp is antisymmetric
// and transitive, the canonical string parses back to the value, and rounding
// agrees with the scaled accessor and is monotonic in the value.
func TestProperties(t *testing.T) {
	vs := propertyValues()
	cmp := make([][]int, len(vs))
	for i, a := range vs {
		cmp[i] = make([]int, len(vs))
		for j, b := range vs {
			cmp[i][j] = a.Cmp(b)
		}
	}
	for i := range vs {
		for j := range vs {
			if cmp[i][j] != -cmp[j][i] {
				t.Errorf("Cmp(%s, %s) = %d, reverse %d", vs[i].ExactString(), vs[j].ExactString(), cmp[i][j], cmp[j][i])
			}
			for k := range vs {
				if cmp[i][j] <= 0 && cmp[j][k] <= 0 && cmp[i][k] > 0 {
					t.Errorf("Cmp not transitive: %s <= %s <= %s", vs[i].ExactString(), vs[j].ExactString(), vs[k].ExactString())
				}
			}
		}
	}
	nano := NewInt64(1, -9)
	for _, q := range vs {
		for _, f := range []string{DecimalSI, BinarySI, DecimalExponent, ""} {
			s := q.Canonical(f)
			back, err := Parse(s)
			switch {
			case cmpMag(q, nano) < 0 && !q.IsZero():
				// Parsing rounds below 1n.
			case f == BinarySI && cmpMag(q, NewInt64(math.MaxInt64, 0)) > 0 && strings.HasSuffix(s, "i"):
				// A binarySI spelling is capped at 2^63-1 when parsed.
			case err != nil || !back.Equal(q):
				t.Errorf("Parse(Canonical(%s, %q) = %q) = %s, %v", q.ExactString(), f, s, back.ExactString(), err)
			}
		}
	}
	scales := []int32{math.MinInt32, -RatLimit - 1, -19, -18, -9, -3, 0, 3, 9, 18, 19, RatLimit + 1, math.MaxInt32}
	for _, s := range scales {
		scaled := make([]int64, len(vs))
		for i, q := range vs {
			r, exact := q.RoundToScale(s)
			if r.Cmp(q)*q.Sign() < 0 || exact != r.Equal(q) {
				t.Errorf("%s.RoundToScale(%d) = %s, %t: not away from zero or exactness wrong", q.ExactString(), s, r.ExactString(), exact)
			}
			v, ok := q.AsScaledInt64(s)
			if ok && !New(big.NewInt(v), int64(s), DecimalSI).Equal(r) {
				t.Errorf("%s at scale %d: AsScaledInt64 %d disagrees with RoundToScale %s", q.ExactString(), s, v, r.ExactString())
			}
			scaled[i] = v
		}
		for i := range vs {
			for j := range vs {
				if cmp[i][j] < 0 && scaled[i] > scaled[j] {
					t.Errorf("AsScaledInt64(%d) not monotonic: %s -> %d, %s -> %d", s, vs[i].ExactString(), scaled[i], vs[j].ExactString(), scaled[j])
				}
			}
		}
	}
}
