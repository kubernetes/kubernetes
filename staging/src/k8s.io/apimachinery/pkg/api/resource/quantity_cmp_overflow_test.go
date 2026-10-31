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
	// The operands with exponents above 308 are capped when parsed, so we load
	// them directly to test how Cmp handles them.
	cases := []struct {
		a    string
		aVal *Quantity // if set, use this value instead of parsing the a string
		b    string
		bVal *Quantity // if set, use this value instead of parsing the b string
		want int
	}{
		{aVal: NewScaledQuantity(1, math.MaxInt32), b: "1n", want: 1},
		{a: "1n", bVal: NewScaledQuantity(1, math.MaxInt32), want: -1},
		{aVal: NewScaledQuantity(1, math.MaxInt32), b: "500m", want: 1},
		{aVal: NewScaledQuantity(1, math.MaxInt32), b: "1", want: 1},
		{aVal: NewScaledQuantity(-1, math.MaxInt32), b: "1n", want: -1},
		{aVal: NewScaledQuantity(-1, math.MaxInt32), b: "-1n", want: -1},
		{aVal: NewScaledQuantity(1, math.MaxInt32), bVal: NewScaledQuantity(1, math.MaxInt32), want: 0},
		{aVal: NewScaledQuantity(1, math.MaxInt32), bVal: NewScaledQuantity(1, math.MaxInt32-1), want: 1},
		{aVal: NewScaledQuantity(1, 10000000), b: "1Gi", want: 1},
		{a: "1Gi", bVal: NewScaledQuantity(1, 10000000), want: -1},
	}
	// quantity returns val if it is set, and otherwise parses s.
	quantity := func(s string, val *Quantity) Quantity {
		if val != nil {
			return *val
		}
		return MustParse(s)
	}
	for _, tc := range cases {
		a, b := quantity(tc.a, tc.aVal), quantity(tc.b, tc.bVal)
		if got := a.Cmp(b); got != tc.want {
			t.Errorf("%s.Cmp(%s) = %d, want %d", a.String(), b.String(), got, tc.want)
		}
	}
}

// CmpInt64 is bounded the same way for a decimal-backed receiver.
func TestCmpInt64LargeScaleDifference(t *testing.T) {
	// 1e2147483647 is capped when parsed, so we load it directly to test how
	// CmpInt64 handles it.
	q := *NewScaledQuantity(1, math.MaxInt32)
	q.ToDec()
	if got := q.CmpInt64(1); got != 1 {
		t.Errorf("CmpInt64(1) = %d, want 1", got)
	}
}
