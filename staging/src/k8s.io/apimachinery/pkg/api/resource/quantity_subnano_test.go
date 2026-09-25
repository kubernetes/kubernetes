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
	"reflect"
	"testing"
)

// TestParseQuantitySubNanoRoundsUp pins that a magnitude below 1n rounds away from
// zero to 1n, extreme exponents included, without building a 10^scale big.Int (a
// regression there shows as a timeout). Expected values are compared with Cmp.
func TestParseQuantitySubNanoRoundsUp(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want string
	}{
		// magnitude below 1n rounds away from zero to the minimum unit
		{"1e-2147483647", "1n"},
		{"-1e-2147483647", "-1n"},
		// fractional mantissa at the extreme exponent, on the Dec path: still 1n
		{"1.5e-2147483647", "1n"},
		{"-1.5e-2147483647", "-1n"},
		{"0.5e-2147483647", "1n"},
		{"0.1e-2147483647", "1n"},
		{"-0.1e-2147483647", "-1n"},
		{"1e-100", "1n"},
		{"-1e-100", "-1n"},
		{"1e-10", "1n"},
		{"9e-10", "1n"},
		{"-9e-10", "-1n"},
		// at and above 1n: unchanged, still routed through Round
		{"1e-9", "1n"},
		{"-1e-9", "-1n"},
		{"2e-9", "2n"},
		{"15e-10", "2n"},
		// long mantissas reach Round through the Dec path (1e-9 and 2e-9 above take the int64 path)
		{"1000000000000000000000e-30", "1n"},
		{"1000000000000000000001e-30", "2n"},
		{"999999999999999999999e-30", "1n"},
		// zero is never rounded up, whatever the exponent
		{"0e-2147483647", "0"},
		{"0.0e-100", "0"},
		// a fractional zero at the extreme exponent reaches the branch that records 1n
		{"0.0e-2147483647", "0"},
		{"-0.0e-2147483647", "0"},
		// a BinarySI value below 1n rounds to 1n and its format flips to DecimalSI
		{"0.00000000000000000000001Ki", "1n"},
	} {
		q, err := ParseQuantity(tc.in)
		if err != nil {
			t.Errorf("ParseQuantity(%q): unexpected error %v", tc.in, err)
			continue
		}
		want := MustParse(tc.want)
		if q.Cmp(want) != 0 {
			t.Errorf("ParseQuantity(%q) = %v, want %v", tc.in, q.String(), tc.want)
		}
	}
}

// TestParseQuantitySubNanoMatchesRound pins the serialized form and confirms the
// shortcut produces the same Quantity the Round path produces.
func TestParseQuantitySubNanoMatchesRound(t *testing.T) {
	for _, tc := range []struct {
		in, want string
		format   Format
	}{
		{"1e-100", "1e-9", DecimalExponent},
		{"0.0000000000001", "1n", DecimalSI},
		{"-0.0000000000001", "-1n", DecimalSI},
		{"0.00000000000000000000001Ki", "1n", DecimalSI},
		// fractional mantissas at the extreme exponent, through the branch that records 1n
		{"1.5e-2147483647", "1e-9", DecimalExponent},
		{"-0.1e-2147483647", "-1e-9", DecimalExponent},
		{"0.0e-2147483647", "0", DecimalExponent},
		{"-0.0e-2147483647", "0", DecimalExponent},
	} {
		if q := MustParse(tc.in); q.String() != tc.want || q.Format != tc.format {
			t.Errorf("ParseQuantity(%q) = (%q, %v), want (%q, %v)", tc.in, q.String(), q.Format, tc.want, tc.format)
		}
	}
	for _, tc := range []struct{ shortcut, roundPath string }{
		{"1e-100", "9e-10"},
		{"-1e-2147483647", "-9e-10"},
		{"0.0000000000001", "0.0000000009"},
	} {
		if a, b := MustParse(tc.shortcut), MustParse(tc.roundPath); !reflect.DeepEqual(a, b) {
			t.Errorf("ParseQuantity(%q) = %#v, want the Round-path result of %q", tc.shortcut, a, tc.roundPath)
		}
	}
}
