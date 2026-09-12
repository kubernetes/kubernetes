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

import "testing"

// TestParseQuantitySubNanoRoundsUp locks the rounding of magnitudes below the
// minimum representable unit (1n): they round away from zero to 1n. A large
// negative exponent hits the same rule, so parsing it must resolve promptly
// rather than materializing a 10^scale big.Int; if the fast path regresses,
// this test times out instead of failing. Values are compared by magnitude so
// the assertion does not depend on the canonical suffix vs exponent spelling.
func TestParseQuantitySubNanoRoundsUp(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want string
	}{
		// magnitude below 1n rounds away from zero to the minimum unit
		{"1e-2147483647", "1n"},
		{"-1e-2147483647", "-1n"},
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
