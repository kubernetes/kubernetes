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
	"testing"
)

// Cmp is bounded and correct when the two operands' scales differ by more than
// an int32 can represent.
func TestCmpLargeScaleDifference(t *testing.T) {
	// ParseQuantity caps values at 1.7976931348623157e308, so build the larger
	// ones directly.
	quantities := map[string]Quantity{
		"1e2147483647":  *NewScaledQuantity(1, math.MaxInt32),
		"-1e2147483647": *NewScaledQuantity(-1, math.MaxInt32),
		"1e2147483646":  *NewScaledQuantity(1, math.MaxInt32-1),
		"1e10000000":    *NewScaledQuantity(1, 10000000),
	}
	quantity := func(s string) Quantity {
		if q, ok := quantities[s]; ok {
			return q
		}
		return MustParse(s)
	}
	cases := []struct {
		a, b string
		want int
	}{
		{"1e2147483647", "1n", 1},
		{"1n", "1e2147483647", -1},
		{"1e2147483647", "500m", 1},
		{"1e2147483647", "1", 1},
		{"-1e2147483647", "1n", -1},
		{"-1e2147483647", "-1n", -1},
		{"1e2147483647", "1e2147483647", 0},
		{"1e2147483647", "1e2147483646", 1},
		{"1e10000000", "1Gi", 1},
		{"1Gi", "1e10000000", -1},
	}
	for _, tc := range cases {
		a, b := quantity(tc.a), quantity(tc.b)
		if got := a.Cmp(b); got != tc.want {
			t.Errorf("%s.Cmp(%s) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
}

// CmpInt64 is bounded the same way for a decimal-backed receiver.
func TestCmpInt64LargeScaleDifference(t *testing.T) {
	q := *NewScaledQuantity(1, math.MaxInt32)
	q.ToDec()
	if got := q.CmpInt64(1); got != 1 {
		t.Errorf("CmpInt64(1) = %d, want 1", got)
	}
}
