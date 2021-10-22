// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package value_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/internal/value"
)

func TestSortKeys(t *testing.T) {
	type (
		MyString string
		MyArray  [2]int
		MyStruct struct {
			A MyString
			B MyArray
			C chan float64
		}
		EmptyStruct struct{}
	)

	opts := []cmp.Option{
		cmp.Comparer(func(x, y float64) bool {
			if math.IsNaN(x) && math.IsNaN(y) {
				return true
			}
			return x == y
		}),
		cmp.Comparer(func(x, y complex128) bool {
			rx, ix, ry, iy := real(x), imag(x), real(y), imag(y)
			if math.IsNaN(rx) && math.IsNaN(ry) {
				rx, ry = 0, 0
			}
			if math.IsNaN(ix) && math.IsNaN(iy) {
				ix, iy = 0, 0
			}
			return rx == ry && ix == iy
		}),
		cmp.Comparer(func(x, y chan bool) bool { return true }),
		cmp.Comparer(func(x, y chan int) bool { return true }),
		cmp.Comparer(func(x, y chan float64) bool { return true }),
		cmp.Comparer(func(x, y chan interface{}) bool { return true }),
		cmp.Comparer(func(x, y *int) bool { return true }),
	}

	tests := []struct {
		in   map[interface{}]bool // Set of keys to sort
		want []interface{}
	}{{
		in:   map[interface{}]bool{1: true, 2: true, 3: true},
		want: []interface{}{1, 2, 3},
	}, {
		in: map[interface{}]bool{
			nil:                    true,
			true:                   true,
			false:                  true,
			-5:                     true,
			-55:                    true,
			-555:                   true,
			uint(1):                true,
			uint(11):               true,
			uint(111):              true,
			"abc":                  true,
			"abcd":                 true,
			"abcde":                true,
			"foo":                  true,
			"bar":                  true,
			MyString("abc"):        true,
			MyString("abcd"):       true,
			MyString("abcde"):      true,
			new(int):               true,
			new(int):               true,
			make(chan bool):        true,
			make(chan bool):        true,
			make(chan int):         true,
			make(chan interface{}): true,
			math.Inf(+1):           true,
			math.Inf(-1):           true,
			1.2345:                 true,
			12.345:                 true,
			123.45:                 true,
			1234.5:                 true,
			0 + 0i:                 true,
			1 + 0i:                 true,
			2 + 0i:                 true,
			0 + 1i:                 true,
			0 + 2i:                 true,
			0 + 3i:                 true,
			[2]int{2, 3}:           true,
			[2]int{4, 0}:           true,
			[2]int{2, 4}:           true,
			MyArray([2]int{2, 4}):  true,
			EmptyStruct{}:          true,
			MyStruct{
				"bravo", [2]int{2, 3}, make(chan float64),
			}: true,
			MyStruct{
				"alpha", [2]int{3, 3}, make(chan float64),
			}: true,
		},
		want: []interface{}{
			nil, false, true,
			-555, -55, -5, uint(1), uint(11), uint(111),
			math.Inf(-1), 1.2345, 12.345, 123.45, 1234.5, math.Inf(+1),
			(0 + 0i), (0 + 1i), (0 + 2i), (0 + 3i), (1 + 0i), (2 + 0i),
			[2]int{2, 3}, [2]int{2, 4}, [2]int{4, 0}, MyArray([2]int{2, 4}),
			make(chan bool), make(chan bool), make(chan int), make(chan interface{}),
			new(int), new(int),
			"abc", "abcd", "abcde", "bar", "foo",
			MyString("abc"), MyString("abcd"), MyString("abcde"),
			EmptyStruct{},
			MyStruct{"alpha", [2]int{3, 3}, make(chan float64)},
			MyStruct{"bravo", [2]int{2, 3}, make(chan float64)},
		},
	}, {
		// NaN values cannot be properly deduplicated.
		// This is okay since map entries with NaN in the keys cannot be
		// retrieved anyways.
		in: map[interface{}]bool{
			math.NaN():                      true,
			math.NaN():                      true,
			complex(0, math.NaN()):          true,
			complex(0, math.NaN()):          true,
			complex(math.NaN(), 0):          true,
			complex(math.NaN(), 0):          true,
			complex(math.NaN(), math.NaN()): true,
		},
		want: []interface{}{
			math.NaN(),
			complex(math.NaN(), math.NaN()),
			complex(math.NaN(), 0),
			complex(0, math.NaN()),
		},
	}}

	for i, tt := range tests {
		// Intentionally pass the map via an unexported field to detect panics.
		// Unfortunately, we cannot actually test the keys without using unsafe.
		v := reflect.ValueOf(struct{ x map[interface{}]bool }{tt.in}).Field(0)
		value.SortKeys(append(v.MapKeys(), v.MapKeys()...))

		// Try again, with keys that have read-write access in reflect.
		v = reflect.ValueOf(tt.in)
		keys := append(v.MapKeys(), v.MapKeys()...)
		var got []interface{}
		for _, k := range value.SortKeys(keys) {
			got = append(got, k.Interface())
		}
		if d := cmp.Diff(got, tt.want, opts...); d != "" {
			t.Errorf("test %d, Sort() mismatch (-got +want):\n%s", i, d)
		}
	}
}
