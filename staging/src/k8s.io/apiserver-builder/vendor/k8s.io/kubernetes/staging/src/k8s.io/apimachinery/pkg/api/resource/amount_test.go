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
	"testing"
)

func TestInt64AmountAsInt64(t *testing.T) {
	for _, test := range []struct {
		value  int64
		scale  Scale
		result int64
		ok     bool
	}{
		{100, 0, 100, true},
		{100, 1, 1000, true},
		{100, -5, 0, false},
		{100, 100, 0, false},
	} {
		r, ok := int64Amount{value: test.value, scale: test.scale}.AsInt64()
		if r != test.result {
			t.Errorf("%v: unexpected result: %d", test, r)
		}
		if ok != test.ok {
			t.Errorf("%v: unexpected ok: %t", test, ok)
		}
	}
}

func TestInt64AmountAdd(t *testing.T) {
	for _, test := range []struct {
		a, b, c int64Amount
		ok      bool
	}{
		{int64Amount{value: 100, scale: 1}, int64Amount{value: 10, scale: 2}, int64Amount{value: 200, scale: 1}, true},
		{int64Amount{value: 100, scale: 1}, int64Amount{value: 1, scale: 2}, int64Amount{value: 110, scale: 1}, true},
		{int64Amount{value: 100, scale: 1}, int64Amount{value: 1, scale: 100}, int64Amount{value: 1, scale: 100}, false},
		{int64Amount{value: -5, scale: 2}, int64Amount{value: 50, scale: 1}, int64Amount{value: 0, scale: 1}, true},
		{int64Amount{value: -5, scale: 2}, int64Amount{value: 5, scale: 2}, int64Amount{value: 0, scale: 2}, true},

		{int64Amount{value: mostPositive, scale: -1}, int64Amount{value: 1, scale: -1}, int64Amount{value: 0, scale: -1}, false},
		{int64Amount{value: mostPositive, scale: -1}, int64Amount{value: 0, scale: -1}, int64Amount{value: mostPositive, scale: -1}, true},
		{int64Amount{value: mostPositive / 10, scale: 1}, int64Amount{value: 10, scale: 0}, int64Amount{value: mostPositive, scale: -1}, false},
	} {
		c := test.a
		ok := c.Add(test.b)
		if ok != test.ok {
			t.Errorf("%v: unexpected ok: %t", test, ok)
		}
		if ok {
			if c != test.c {
				t.Errorf("%v: unexpected result: %d", test, c)
			}
		} else {
			if c != test.a {
				t.Errorf("%v: overflow addition mutated source: %d", test, c)
			}
		}

		// addition is commutative
		c = test.b
		if ok := c.Add(test.a); ok != test.ok {
			t.Errorf("%v: unexpected ok: %t", test, ok)
		}
		if ok {
			if c != test.c {
				t.Errorf("%v: unexpected result: %d", test, c)
			}
		} else {
			if c != test.b {
				t.Errorf("%v: overflow addition mutated source: %d", test, c)
			}
		}
	}
}
func TestInt64AsCanonicalString(t *testing.T) {
	for _, test := range []struct {
		value    int64
		scale    Scale
		result   string
		exponent int32
	}{
		{100, 0, "100", 0},
		{100, 1, "1", 3},
		{100, -1, "10", 0},
		{10800, -10, "1080", -9},
	} {
		r, exp := int64Amount{value: test.value, scale: test.scale}.AsCanonicalBytes(nil)
		if string(r) != test.result {
			t.Errorf("%v: unexpected result: %s", test, r)
		}
		if exp != test.exponent {
			t.Errorf("%v: unexpected exponent: %d", test, exp)
		}
	}
}
