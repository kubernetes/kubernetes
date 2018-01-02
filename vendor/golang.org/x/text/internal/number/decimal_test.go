// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"
)

func mkfloat(num string) float64 {
	u, _ := strconv.ParseUint(num, 10, 32)
	return float64(u)
}

// mkdec creates a decimal from a string. All ASCII digits are converted to
// digits in the decimal. The dot is used to indicate the scale by which the
// digits are shifted. Numbers may have an additional exponent or be the special
// value NaN, Inf, or -Inf.
func mkdec(num string) (d Decimal) {
	if num[0] == '-' {
		d.Neg = true
		num = num[1:]
	}
	switch num {
	case "NaN":
		d.NaN = true
		return
	case "Inf":
		d.Inf = true
		return
	}
	if p := strings.IndexAny(num, "eE"); p != -1 {
		i64, err := strconv.ParseInt(num[p+1:], 10, 32)
		if err != nil {
			panic(err)
		}
		d.Exp = int32(i64)
		num = num[:p]
	}
	if p := strings.IndexByte(num, '.'); p != -1 {
		d.Exp += int32(p)
		num = num[:p] + num[p+1:]
	} else {
		d.Exp += int32(len(num))
	}
	d.Digits = []byte(num)
	for i := range d.Digits {
		d.Digits[i] -= '0'
	}
	return d.normalize()
}

func byteNum(s string) []byte {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		if c := s[i]; '0' <= c && c <= '9' {
			b[i] = s[i] - '0'
		} else {
			b[i] = s[i] - 'a' + 10
		}
	}
	return b
}

func strNum(s string) string {
	return string(byteNum(s))
}

func TestDecimalString(t *testing.T) {
	for _, test := range []struct {
		x    Decimal
		want string
	}{
		{want: "0"},
		{Decimal{Digits: nil, Exp: 1000}, "0"}, // exponent of 1000 is ignored
		{Decimal{Digits: byteNum("12345"), Exp: 0}, "0.12345"},
		{Decimal{Digits: byteNum("12345"), Exp: -3}, "0.00012345"},
		{Decimal{Digits: byteNum("12345"), Exp: +3}, "123.45"},
		{Decimal{Digits: byteNum("12345"), Exp: +10}, "1234500000"},
	} {
		if got := test.x.String(); got != test.want {
			t.Errorf("%v == %q; want %q", test.x, got, test.want)
		}
	}
}

func TestRounding(t *testing.T) {
	testCases := []struct {
		x string
		n int
		// modes is the result for modes. Signs are left out of the result.
		// The results are stored in the following order:
		// zero, negInf
		// nearZero, nearEven, nearAway
		// away, posInf
		modes [numModes]string
	}{
		{"0", 1, [numModes]string{
			"0", "0",
			"0", "0", "0",
			"0", "0"}},
		{"1", 1, [numModes]string{
			"1", "1",
			"1", "1", "1",
			"1", "1"}},
		{"5", 1, [numModes]string{
			"5", "5",
			"5", "5", "5",
			"5", "5"}},
		{"15", 1, [numModes]string{
			"10", "10",
			"10", "20", "20",
			"20", "20"}},
		{"45", 1, [numModes]string{
			"40", "40",
			"40", "40", "50",
			"50", "50"}},
		{"95", 1, [numModes]string{
			"90", "90",
			"90", "100", "100",
			"100", "100"}},

		{"12344999", 4, [numModes]string{
			"12340000", "12340000",
			"12340000", "12340000", "12340000",
			"12350000", "12350000"}},
		{"12345000", 4, [numModes]string{
			"12340000", "12340000",
			"12340000", "12340000", "12350000",
			"12350000", "12350000"}},
		{"12345001", 4, [numModes]string{
			"12340000", "12340000",
			"12350000", "12350000", "12350000",
			"12350000", "12350000"}},
		{"12345100", 4, [numModes]string{
			"12340000", "12340000",
			"12350000", "12350000", "12350000",
			"12350000", "12350000"}},
		{"23454999", 4, [numModes]string{
			"23450000", "23450000",
			"23450000", "23450000", "23450000",
			"23460000", "23460000"}},
		{"23455000", 4, [numModes]string{
			"23450000", "23450000",
			"23450000", "23460000", "23460000",
			"23460000", "23460000"}},
		{"23455001", 4, [numModes]string{
			"23450000", "23450000",
			"23460000", "23460000", "23460000",
			"23460000", "23460000"}},
		{"23455100", 4, [numModes]string{
			"23450000", "23450000",
			"23460000", "23460000", "23460000",
			"23460000", "23460000"}},

		{"99994999", 4, [numModes]string{
			"99990000", "99990000",
			"99990000", "99990000", "99990000",
			"100000000", "100000000"}},
		{"99995000", 4, [numModes]string{
			"99990000", "99990000",
			"99990000", "100000000", "100000000",
			"100000000", "100000000"}},
		{"99999999", 4, [numModes]string{
			"99990000", "99990000",
			"100000000", "100000000", "100000000",
			"100000000", "100000000"}},

		{"12994999", 4, [numModes]string{
			"12990000", "12990000",
			"12990000", "12990000", "12990000",
			"13000000", "13000000"}},
		{"12995000", 4, [numModes]string{
			"12990000", "12990000",
			"12990000", "13000000", "13000000",
			"13000000", "13000000"}},
		{"12999999", 4, [numModes]string{
			"12990000", "12990000",
			"13000000", "13000000", "13000000",
			"13000000", "13000000"}},
	}
	modes := []RoundingMode{
		ToZero, ToNegativeInf,
		ToNearestZero, ToNearestEven, ToNearestAway,
		AwayFromZero, ToPositiveInf,
	}
	for _, tc := range testCases {
		// Create negative counterpart tests: the sign is reversed and
		// ToPositiveInf and ToNegativeInf swapped.
		negModes := tc.modes
		negModes[1], negModes[6] = negModes[6], negModes[1]
		for i, res := range negModes {
			negModes[i] = "-" + res
		}
		for i, m := range modes {
			t.Run(fmt.Sprintf("x:%s/n:%d/%s", tc.x, tc.n, m), func(t *testing.T) {
				d := mkdec(tc.x)
				d.round(m, tc.n)
				if got := d.String(); got != tc.modes[i] {
					t.Errorf("pos decimal: got %q; want %q", d.String(), tc.modes[i])
				}

				mult := math.Pow(10, float64(len(tc.x)-tc.n))
				f := mkfloat(tc.x)
				f = m.roundFloat(f/mult) * mult
				if got := fmt.Sprintf("%.0f", f); got != tc.modes[i] {
					t.Errorf("pos float: got %q; want %q", got, tc.modes[i])
				}

				// Test the negative case. This is the same as the positive
				// case, but with ToPositiveInf and ToNegativeInf swapped.
				d = mkdec(tc.x)
				d.Neg = true
				d.round(m, tc.n)
				if got, want := d.String(), negModes[i]; got != want {
					t.Errorf("neg decimal: got %q; want %q", d.String(), want)
				}

				f = -mkfloat(tc.x)
				f = m.roundFloat(f/mult) * mult
				if got := fmt.Sprintf("%.0f", f); got != negModes[i] {
					t.Errorf("neg float: got %q; want %q", got, negModes[i])
				}
			})
		}
	}
}

func TestConvert(t *testing.T) {
	scale2 := &RoundingContext{Scale: 2}
	scale2away := &RoundingContext{Scale: 2, Mode: AwayFromZero}
	inc0_05 := &RoundingContext{Increment: 5, Scale: 2}
	inc50 := &RoundingContext{Increment: 50}
	prec3 := &RoundingContext{Precision: 3}
	testCases := []struct {
		x   interface{}
		rc  *RoundingContext
		out string
	}{
		{int8(-34), scale2, "-34"},
		{int16(-234), scale2, "-234"},
		{int32(-234), scale2, "-234"},
		{int64(-234), scale2, "-234"},
		{int(-234), scale2, "-234"},
		{uint8(234), scale2, "234"},
		{uint16(234), scale2, "234"},
		{uint32(234), scale2, "234"},
		{uint64(234), scale2, "234"},
		{uint(234), scale2, "234"},
		{-0.001, scale2, "-0"},
		{-1e9, scale2, "-1000000000.00"},
		{0.234, scale2, "0.23"},
		{0.234, scale2away, "0.24"},
		{0.1234, prec3, "0.123"},
		{1234.0, prec3, "1230"},
		{1.2345e10, prec3, "12300000000"},

		{0.03, inc0_05, "0.05"},
		{0.025, inc0_05, "0"},
		{0.075, inc0_05, "0.10"},
		{325, inc50, "300"},
		{375, inc50, "400"},

		{converter(3), scale2, "100"},

		{math.Inf(1), inc50, "Inf"},
		{math.Inf(-1), inc50, "-Inf"},
		{math.NaN(), inc50, "NaN"},
	}
	for _, tc := range testCases {
		var d Decimal
		t.Run(fmt.Sprintf("%T:%v-%v", tc.x, tc.x, tc.rc), func(t *testing.T) {
			d.Convert(tc.rc, tc.x)
			if got := d.String(); got != tc.out {
				t.Errorf("got %q; want %q", got, tc.out)
			}
		})
	}
}

type converter int

func (c converter) Convert(d *Decimal, r *RoundingContext) {
	d.Digits = append(d.Digits, 1, 0, 0)
	d.Exp = 3
}
