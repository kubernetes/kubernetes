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
	"testing"
)

func TestRemoveInt64Factors(t *testing.T) {
	for _, test := range []struct {
		value  int64
		max    int64
		result int64
		scale  int32
	}{
		{100, 10, 1, 2},
		{100, 10, 1, 2},
		{100, 100, 1, 1},
		{1, 10, 1, 0},
	} {
		r, s := removeInt64Factors(test.value, test.max)
		if r != test.result {
			t.Errorf("%v: unexpected result: %d", test, r)
		}
		if s != test.scale {
			t.Errorf("%v: unexpected scale: %d", test, s)
		}
	}
}

func TestNegativeScaleInt64(t *testing.T) {
	for _, test := range []struct {
		base   int64
		scale  Scale
		result int64
		exact  bool
	}{
		{1234567, 3, 1234, false},
	} {
		result, exact := negativeScaleInt64(test.base, test.scale)
		if result != test.result {
			t.Errorf("%v: unexpected result: %d", test, result)
		}
		if exact != test.exact {
			t.Errorf("%v: unexpected exact: %t", test, exact)
		}
	}
}

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
