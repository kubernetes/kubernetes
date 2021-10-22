// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmpopts

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/xerrors"
)

type (
	MyInt    int
	MyInts   []int
	MyFloat  float32
	MyString string
	MyTime   struct{ time.Time }
	MyStruct struct {
		A, B []int
		C, D map[time.Time]string
	}

	Foo1 struct{ Alpha, Bravo, Charlie int }
	Foo2 struct{ *Foo1 }
	Foo3 struct{ *Foo2 }
	Bar1 struct{ Foo3 }
	Bar2 struct {
		Bar1
		*Foo3
		Bravo float32
	}
	Bar3 struct {
		Bar1
		Bravo *Bar2
		Delta struct{ Echo Foo1 }
		*Foo3
		Alpha string
	}

	privateStruct struct{ Public, private int }
	PublicStruct  struct{ Public, private int }
	ParentStruct  struct {
		*privateStruct
		*PublicStruct
		Public  int
		private int
	}

	Everything struct {
		MyInt
		MyFloat
		MyTime
		MyStruct
		Bar3
		ParentStruct
	}

	EmptyInterface interface{}
)

func TestOptions(t *testing.T) {
	createBar3X := func() *Bar3 {
		return &Bar3{
			Bar1: Bar1{Foo3{&Foo2{&Foo1{Bravo: 2}}}},
			Bravo: &Bar2{
				Bar1:  Bar1{Foo3{&Foo2{&Foo1{Charlie: 7}}}},
				Foo3:  &Foo3{&Foo2{&Foo1{Bravo: 5}}},
				Bravo: 4,
			},
			Delta: struct{ Echo Foo1 }{Foo1{Charlie: 3}},
			Foo3:  &Foo3{&Foo2{&Foo1{Alpha: 1}}},
			Alpha: "alpha",
		}
	}
	createBar3Y := func() *Bar3 {
		return &Bar3{
			Bar1: Bar1{Foo3{&Foo2{&Foo1{Bravo: 3}}}},
			Bravo: &Bar2{
				Bar1:  Bar1{Foo3{&Foo2{&Foo1{Charlie: 8}}}},
				Foo3:  &Foo3{&Foo2{&Foo1{Bravo: 6}}},
				Bravo: 5,
			},
			Delta: struct{ Echo Foo1 }{Foo1{Charlie: 4}},
			Foo3:  &Foo3{&Foo2{&Foo1{Alpha: 2}}},
			Alpha: "ALPHA",
		}
	}

	tests := []struct {
		label     string       // Test name
		x, y      interface{}  // Input values to compare
		opts      []cmp.Option // Input options
		wantEqual bool         // Whether the inputs are equal
		wantPanic bool         // Whether Equal should panic
		reason    string       // The reason for the expected outcome
	}{{
		label:     "EquateEmpty",
		x:         []int{},
		y:         []int(nil),
		wantEqual: false,
		reason:    "not equal because empty non-nil and nil slice differ",
	}, {
		label:     "EquateEmpty",
		x:         []int{},
		y:         []int(nil),
		opts:      []cmp.Option{EquateEmpty()},
		wantEqual: true,
		reason:    "equal because EquateEmpty equates empty slices",
	}, {
		label:     "SortSlices",
		x:         []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
		y:         []int{1, 0, 5, 2, 8, 9, 4, 3, 6, 7},
		wantEqual: false,
		reason:    "not equal because element order differs",
	}, {
		label:     "SortSlices",
		x:         []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
		y:         []int{1, 0, 5, 2, 8, 9, 4, 3, 6, 7},
		opts:      []cmp.Option{SortSlices(func(x, y int) bool { return x < y })},
		wantEqual: true,
		reason:    "equal because SortSlices sorts the slices",
	}, {
		label:     "SortSlices",
		x:         []MyInt{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
		y:         []MyInt{1, 0, 5, 2, 8, 9, 4, 3, 6, 7},
		opts:      []cmp.Option{SortSlices(func(x, y int) bool { return x < y })},
		wantEqual: false,
		reason:    "not equal because MyInt is not the same type as int",
	}, {
		label:     "SortSlices",
		x:         []float64{0, 1, 1, 2, 2, 2},
		y:         []float64{2, 0, 2, 1, 2, 1},
		opts:      []cmp.Option{SortSlices(func(x, y float64) bool { return x < y })},
		wantEqual: true,
		reason:    "equal even when sorted with duplicate elements",
	}, {
		label:     "SortSlices",
		x:         []float64{0, 1, 1, 2, 2, 2, math.NaN(), 3, 3, 3, 3, 4, 4, 4, 4},
		y:         []float64{2, 0, 4, 4, 3, math.NaN(), 4, 1, 3, 2, 3, 3, 4, 1, 2},
		opts:      []cmp.Option{SortSlices(func(x, y float64) bool { return x < y })},
		wantPanic: true,
		reason:    "panics because SortSlices used with non-transitive less function",
	}, {
		label: "SortSlices",
		x:     []float64{0, 1, 1, 2, 2, 2, math.NaN(), 3, 3, 3, 3, 4, 4, 4, 4},
		y:     []float64{2, 0, 4, 4, 3, math.NaN(), 4, 1, 3, 2, 3, 3, 4, 1, 2},
		opts: []cmp.Option{SortSlices(func(x, y float64) bool {
			return (!math.IsNaN(x) && math.IsNaN(y)) || x < y
		})},
		wantEqual: false,
		reason:    "no panics because SortSlices used with valid less function; not equal because NaN != NaN",
	}, {
		label: "SortSlices+EquateNaNs",
		x:     []float64{0, 1, 1, 2, 2, 2, math.NaN(), 3, 3, 3, math.NaN(), 3, 4, 4, 4, 4},
		y:     []float64{2, 0, 4, 4, 3, math.NaN(), 4, 1, 3, 2, 3, 3, 4, 1, math.NaN(), 2},
		opts: []cmp.Option{
			EquateNaNs(),
			SortSlices(func(x, y float64) bool {
				return (!math.IsNaN(x) && math.IsNaN(y)) || x < y
			}),
		},
		wantEqual: true,
		reason:    "no panics because SortSlices used with valid less function; equal because EquateNaNs is used",
	}, {
		label: "SortMaps",
		x: map[time.Time]string{
			time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC): "0th birthday",
			time.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC): "1st birthday",
			time.Date(2011, time.November, 10, 23, 0, 0, 0, time.UTC): "2nd birthday",
		},
		y: map[time.Time]string{
			time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local): "0th birthday",
			time.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local): "1st birthday",
			time.Date(2011, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local): "2nd birthday",
		},
		wantEqual: false,
		reason:    "not equal because timezones differ",
	}, {
		label: "SortMaps",
		x: map[time.Time]string{
			time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC): "0th birthday",
			time.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC): "1st birthday",
			time.Date(2011, time.November, 10, 23, 0, 0, 0, time.UTC): "2nd birthday",
		},
		y: map[time.Time]string{
			time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local): "0th birthday",
			time.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local): "1st birthday",
			time.Date(2011, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local): "2nd birthday",
		},
		opts:      []cmp.Option{SortMaps(func(x, y time.Time) bool { return x.Before(y) })},
		wantEqual: true,
		reason:    "equal because SortMaps flattens to a slice where Time.Equal can be used",
	}, {
		label: "SortMaps",
		x: map[MyTime]string{
			{time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)}: "0th birthday",
			{time.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC)}: "1st birthday",
			{time.Date(2011, time.November, 10, 23, 0, 0, 0, time.UTC)}: "2nd birthday",
		},
		y: map[MyTime]string{
			{time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local)}: "0th birthday",
			{time.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local)}: "1st birthday",
			{time.Date(2011, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local)}: "2nd birthday",
		},
		opts:      []cmp.Option{SortMaps(func(x, y time.Time) bool { return x.Before(y) })},
		wantEqual: false,
		reason:    "not equal because MyTime is not assignable to time.Time",
	}, {
		label: "SortMaps",
		x:     map[int]string{-3: "", -2: "", -1: "", 0: "", 1: "", 2: "", 3: ""},
		// => {0, 1, 2, 3, -1, -2, -3},
		y: map[int]string{300: "", 200: "", 100: "", 0: "", 1: "", 2: "", 3: ""},
		// => {0, 1, 2, 3, 100, 200, 300},
		opts: []cmp.Option{SortMaps(func(a, b int) bool {
			if -10 < a && a <= 0 {
				a *= -100
			}
			if -10 < b && b <= 0 {
				b *= -100
			}
			return a < b
		})},
		wantEqual: false,
		reason:    "not equal because values differ even though SortMap provides valid ordering",
	}, {
		label: "SortMaps",
		x:     map[int]string{-3: "", -2: "", -1: "", 0: "", 1: "", 2: "", 3: ""},
		// => {0, 1, 2, 3, -1, -2, -3},
		y: map[int]string{300: "", 200: "", 100: "", 0: "", 1: "", 2: "", 3: ""},
		// => {0, 1, 2, 3, 100, 200, 300},
		opts: []cmp.Option{
			SortMaps(func(x, y int) bool {
				if -10 < x && x <= 0 {
					x *= -100
				}
				if -10 < y && y <= 0 {
					y *= -100
				}
				return x < y
			}),
			cmp.Comparer(func(x, y int) bool {
				if -10 < x && x <= 0 {
					x *= -100
				}
				if -10 < y && y <= 0 {
					y *= -100
				}
				return x == y
			}),
		},
		wantEqual: true,
		reason:    "equal because Comparer used to equate differences",
	}, {
		label: "SortMaps",
		x:     map[int]string{-3: "", -2: "", -1: "", 0: "", 1: "", 2: "", 3: ""},
		y:     map[int]string{},
		opts: []cmp.Option{SortMaps(func(x, y int) bool {
			return x < y && x >= 0 && y >= 0
		})},
		wantPanic: true,
		reason:    "panics because SortMaps used with non-transitive less function",
	}, {
		label: "SortMaps",
		x:     map[int]string{-3: "", -2: "", -1: "", 0: "", 1: "", 2: "", 3: ""},
		y:     map[int]string{},
		opts: []cmp.Option{SortMaps(func(x, y int) bool {
			return math.Abs(float64(x)) < math.Abs(float64(y))
		})},
		wantPanic: true,
		reason:    "panics because SortMaps used with partial less function",
	}, {
		label: "EquateEmpty+SortSlices+SortMaps",
		x: MyStruct{
			A: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
			C: map[time.Time]string{
				time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC): "0th birthday",
				time.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC): "1st birthday",
			},
			D: map[time.Time]string{},
		},
		y: MyStruct{
			A: []int{1, 0, 5, 2, 8, 9, 4, 3, 6, 7},
			B: []int{},
			C: map[time.Time]string{
				time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local): "0th birthday",
				time.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC).In(time.Local): "1st birthday",
			},
		},
		opts: []cmp.Option{
			EquateEmpty(),
			SortSlices(func(x, y int) bool { return x < y }),
			SortMaps(func(x, y time.Time) bool { return x.Before(y) }),
		},
		wantEqual: true,
		reason:    "no panics because EquateEmpty should compose with the sort options",
	}, {
		label:     "EquateApprox",
		x:         3.09,
		y:         3.10,
		wantEqual: false,
		reason:    "not equal because floats do not exactly matches",
	}, {
		label:     "EquateApprox",
		x:         3.09,
		y:         3.10,
		opts:      []cmp.Option{EquateApprox(0, 0)},
		wantEqual: false,
		reason:    "not equal because EquateApprox(0 ,0) is equivalent to using ==",
	}, {
		label:     "EquateApprox",
		x:         3.09,
		y:         3.10,
		opts:      []cmp.Option{EquateApprox(0.003, 0.009)},
		wantEqual: false,
		reason:    "not equal because EquateApprox is too strict",
	}, {
		label:     "EquateApprox",
		x:         3.09,
		y:         3.10,
		opts:      []cmp.Option{EquateApprox(0, 0.011)},
		wantEqual: true,
		reason:    "equal because margin is loose enough to match",
	}, {
		label:     "EquateApprox",
		x:         3.09,
		y:         3.10,
		opts:      []cmp.Option{EquateApprox(0.004, 0)},
		wantEqual: true,
		reason:    "equal because fraction is loose enough to match",
	}, {
		label:     "EquateApprox",
		x:         3.09,
		y:         3.10,
		opts:      []cmp.Option{EquateApprox(0.004, 0.011)},
		wantEqual: true,
		reason:    "equal because both the margin and fraction are loose enough to match",
	}, {
		label:     "EquateApprox",
		x:         float32(3.09),
		y:         float64(3.10),
		opts:      []cmp.Option{EquateApprox(0.004, 0)},
		wantEqual: false,
		reason:    "not equal because the types differ",
	}, {
		label:     "EquateApprox",
		x:         float32(3.09),
		y:         float32(3.10),
		opts:      []cmp.Option{EquateApprox(0.004, 0)},
		wantEqual: true,
		reason:    "equal because EquateApprox also applies on float32s",
	}, {
		label:     "EquateApprox",
		x:         []float64{math.Inf(+1), math.Inf(-1)},
		y:         []float64{math.Inf(+1), math.Inf(-1)},
		opts:      []cmp.Option{EquateApprox(0, 1)},
		wantEqual: true,
		reason:    "equal because we fall back on == which matches Inf (EquateApprox does not apply on Inf) ",
	}, {
		label:     "EquateApprox",
		x:         []float64{math.Inf(+1), -1e100},
		y:         []float64{+1e100, math.Inf(-1)},
		opts:      []cmp.Option{EquateApprox(0, 1)},
		wantEqual: false,
		reason:    "not equal because we fall back on == where Inf != 1e100 (EquateApprox does not apply on Inf)",
	}, {
		label:     "EquateApprox",
		x:         float64(+1e100),
		y:         float64(-1e100),
		opts:      []cmp.Option{EquateApprox(math.Inf(+1), 0)},
		wantEqual: true,
		reason:    "equal because infinite fraction matches everything",
	}, {
		label:     "EquateApprox",
		x:         float64(+1e100),
		y:         float64(-1e100),
		opts:      []cmp.Option{EquateApprox(0, math.Inf(+1))},
		wantEqual: true,
		reason:    "equal because infinite margin matches everything",
	}, {
		label:     "EquateApprox",
		x:         math.Pi,
		y:         math.Pi,
		opts:      []cmp.Option{EquateApprox(0, 0)},
		wantEqual: true,
		reason:    "equal because EquateApprox(0, 0) is equivalent to ==",
	}, {
		label:     "EquateApprox",
		x:         math.Pi,
		y:         math.Nextafter(math.Pi, math.Inf(+1)),
		opts:      []cmp.Option{EquateApprox(0, 0)},
		wantEqual: false,
		reason:    "not equal because EquateApprox(0, 0) is equivalent to ==",
	}, {
		label:     "EquateNaNs",
		x:         []float64{1.0, math.NaN(), math.E, -0.0, +0.0, math.Inf(+1), math.Inf(-1)},
		y:         []float64{1.0, math.NaN(), math.E, -0.0, +0.0, math.Inf(+1), math.Inf(-1)},
		wantEqual: false,
		reason:    "not equal because NaN != NaN",
	}, {
		label:     "EquateNaNs",
		x:         []float64{1.0, math.NaN(), math.E, -0.0, +0.0, math.Inf(+1), math.Inf(-1)},
		y:         []float64{1.0, math.NaN(), math.E, -0.0, +0.0, math.Inf(+1), math.Inf(-1)},
		opts:      []cmp.Option{EquateNaNs()},
		wantEqual: true,
		reason:    "equal because EquateNaNs allows NaN == NaN",
	}, {
		label:     "EquateNaNs",
		x:         []float32{1.0, float32(math.NaN()), math.E, -0.0, +0.0},
		y:         []float32{1.0, float32(math.NaN()), math.E, -0.0, +0.0},
		opts:      []cmp.Option{EquateNaNs()},
		wantEqual: true,
		reason:    "equal because EquateNaNs operates on float32",
	}, {
		label: "EquateApprox+EquateNaNs",
		x:     []float64{1.0, math.NaN(), math.E, -0.0, +0.0, math.Inf(+1), math.Inf(-1), 1.01, 5001},
		y:     []float64{1.0, math.NaN(), math.E, -0.0, +0.0, math.Inf(+1), math.Inf(-1), 1.02, 5002},
		opts: []cmp.Option{
			EquateNaNs(),
			EquateApprox(0.01, 0),
		},
		wantEqual: true,
		reason:    "equal because EquateNaNs and EquateApprox compose together",
	}, {
		label: "EquateApprox+EquateNaNs",
		x:     []MyFloat{1.0, MyFloat(math.NaN()), MyFloat(math.E), -0.0, +0.0, MyFloat(math.Inf(+1)), MyFloat(math.Inf(-1)), 1.01, 5001},
		y:     []MyFloat{1.0, MyFloat(math.NaN()), MyFloat(math.E), -0.0, +0.0, MyFloat(math.Inf(+1)), MyFloat(math.Inf(-1)), 1.02, 5002},
		opts: []cmp.Option{
			EquateNaNs(),
			EquateApprox(0.01, 0),
		},
		wantEqual: false,
		reason:    "not equal because EquateApprox and EquateNaNs do not apply on a named type",
	}, {
		label: "EquateApprox+EquateNaNs+Transform",
		x:     []MyFloat{1.0, MyFloat(math.NaN()), MyFloat(math.E), -0.0, +0.0, MyFloat(math.Inf(+1)), MyFloat(math.Inf(-1)), 1.01, 5001},
		y:     []MyFloat{1.0, MyFloat(math.NaN()), MyFloat(math.E), -0.0, +0.0, MyFloat(math.Inf(+1)), MyFloat(math.Inf(-1)), 1.02, 5002},
		opts: []cmp.Option{
			cmp.Transformer("", func(x MyFloat) float64 { return float64(x) }),
			EquateNaNs(),
			EquateApprox(0.01, 0),
		},
		wantEqual: true,
		reason:    "equal because named type is transformed to float64",
	}, {
		label:     "EquateApproxTime",
		x:         time.Date(2009, 11, 10, 23, 0, 0, 0, time.UTC),
		y:         time.Date(2009, 11, 10, 23, 0, 0, 0, time.UTC),
		opts:      []cmp.Option{EquateApproxTime(0)},
		wantEqual: true,
		reason:    "equal because times are identical",
	}, {
		label:     "EquateApproxTime",
		x:         time.Date(2009, 11, 10, 23, 0, 0, 0, time.UTC),
		y:         time.Date(2009, 11, 10, 23, 0, 3, 0, time.UTC),
		opts:      []cmp.Option{EquateApproxTime(3 * time.Second)},
		wantEqual: true,
		reason:    "equal because time is exactly at the allowed margin",
	}, {
		label:     "EquateApproxTime",
		x:         time.Date(2009, 11, 10, 23, 0, 3, 0, time.UTC),
		y:         time.Date(2009, 11, 10, 23, 0, 0, 0, time.UTC),
		opts:      []cmp.Option{EquateApproxTime(3 * time.Second)},
		wantEqual: true,
		reason:    "equal because time is exactly at the allowed margin (negative)",
	}, {
		label:     "EquateApproxTime",
		x:         time.Date(2009, 11, 10, 23, 0, 3, 0, time.UTC),
		y:         time.Date(2009, 11, 10, 23, 0, 0, 0, time.UTC),
		opts:      []cmp.Option{EquateApproxTime(3*time.Second - 1)},
		wantEqual: false,
		reason:    "not equal because time is outside allowed margin",
	}, {
		label:     "EquateApproxTime",
		x:         time.Date(2009, 11, 10, 23, 0, 0, 0, time.UTC),
		y:         time.Date(2009, 11, 10, 23, 0, 3, 0, time.UTC),
		opts:      []cmp.Option{EquateApproxTime(3*time.Second - 1)},
		wantEqual: false,
		reason:    "not equal because time is outside allowed margin (negative)",
	}, {
		label:     "EquateApproxTime",
		x:         time.Time{},
		y:         time.Time{},
		opts:      []cmp.Option{EquateApproxTime(3 * time.Second)},
		wantEqual: true,
		reason:    "equal because both times are zero",
	}, {
		label:     "EquateApproxTime",
		x:         time.Time{},
		y:         time.Time{}.Add(1),
		opts:      []cmp.Option{EquateApproxTime(3 * time.Second)},
		wantEqual: false,
		reason:    "not equal because zero time is always not equal not non-zero",
	}, {
		label:     "EquateApproxTime",
		x:         time.Time{}.Add(1),
		y:         time.Time{},
		opts:      []cmp.Option{EquateApproxTime(3 * time.Second)},
		wantEqual: false,
		reason:    "not equal because zero time is always not equal not non-zero",
	}, {
		label:     "EquateApproxTime",
		x:         time.Date(2409, 11, 10, 23, 0, 0, 0, time.UTC),
		y:         time.Date(2000, 11, 10, 23, 0, 3, 0, time.UTC),
		opts:      []cmp.Option{EquateApproxTime(3 * time.Second)},
		wantEqual: false,
		reason:    "time difference overflows time.Duration",
	}, {
		label:     "EquateErrors",
		x:         nil,
		y:         nil,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "nil values are equal",
	}, {
		label:     "EquateErrors",
		x:         errors.New("EOF"),
		y:         io.EOF,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: false,
		reason:    "user-defined EOF is not exactly equal",
	}, {
		label:     "EquateErrors",
		x:         xerrors.Errorf("wrapped: %w", io.EOF),
		y:         io.EOF,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "wrapped io.EOF is equal according to errors.Is",
	}, {
		label:     "EquateErrors",
		x:         xerrors.Errorf("wrapped: %w", io.EOF),
		y:         io.EOF,
		wantEqual: false,
		reason:    "wrapped io.EOF is not equal without EquateErrors option",
	}, {
		label:     "EquateErrors",
		x:         io.EOF,
		y:         io.EOF,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "sentinel errors are equal",
	}, {
		label:     "EquateErrors",
		x:         io.EOF,
		y:         AnyError,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "AnyError is equal to any non-nil error",
	}, {
		label:     "EquateErrors",
		x:         io.EOF,
		y:         AnyError,
		wantEqual: false,
		reason:    "AnyError is not equal to any non-nil error without EquateErrors option",
	}, {
		label:     "EquateErrors",
		x:         nil,
		y:         AnyError,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: false,
		reason:    "AnyError is not equal to nil value",
	}, {
		label:     "EquateErrors",
		x:         nil,
		y:         nil,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "nil values are equal",
	}, {
		label:     "EquateErrors",
		x:         errors.New("EOF"),
		y:         io.EOF,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: false,
		reason:    "user-defined EOF is not exactly equal",
	}, {
		label:     "EquateErrors",
		x:         xerrors.Errorf("wrapped: %w", io.EOF),
		y:         io.EOF,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "wrapped io.EOF is equal according to errors.Is",
	}, {
		label:     "EquateErrors",
		x:         xerrors.Errorf("wrapped: %w", io.EOF),
		y:         io.EOF,
		wantEqual: false,
		reason:    "wrapped io.EOF is not equal without EquateErrors option",
	}, {
		label:     "EquateErrors",
		x:         io.EOF,
		y:         io.EOF,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "sentinel errors are equal",
	}, {
		label:     "EquateErrors",
		x:         io.EOF,
		y:         AnyError,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "AnyError is equal to any non-nil error",
	}, {
		label:     "EquateErrors",
		x:         io.EOF,
		y:         AnyError,
		wantEqual: false,
		reason:    "AnyError is not equal to any non-nil error without EquateErrors option",
	}, {
		label:     "EquateErrors",
		x:         nil,
		y:         AnyError,
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: false,
		reason:    "AnyError is not equal to nil value",
	}, {
		label:     "EquateErrors",
		x:         struct{ E error }{nil},
		y:         struct{ E error }{nil},
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "nil values are equal",
	}, {
		label:     "EquateErrors",
		x:         struct{ E error }{errors.New("EOF")},
		y:         struct{ E error }{io.EOF},
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: false,
		reason:    "user-defined EOF is not exactly equal",
	}, {
		label:     "EquateErrors",
		x:         struct{ E error }{xerrors.Errorf("wrapped: %w", io.EOF)},
		y:         struct{ E error }{io.EOF},
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "wrapped io.EOF is equal according to errors.Is",
	}, {
		label:     "EquateErrors",
		x:         struct{ E error }{xerrors.Errorf("wrapped: %w", io.EOF)},
		y:         struct{ E error }{io.EOF},
		wantEqual: false,
		reason:    "wrapped io.EOF is not equal without EquateErrors option",
	}, {
		label:     "EquateErrors",
		x:         struct{ E error }{io.EOF},
		y:         struct{ E error }{io.EOF},
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "sentinel errors are equal",
	}, {
		label:     "EquateErrors",
		x:         struct{ E error }{io.EOF},
		y:         struct{ E error }{AnyError},
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: true,
		reason:    "AnyError is equal to any non-nil error",
	}, {
		label:     "EquateErrors",
		x:         struct{ E error }{io.EOF},
		y:         struct{ E error }{AnyError},
		wantEqual: false,
		reason:    "AnyError is not equal to any non-nil error without EquateErrors option",
	}, {
		label:     "EquateErrors",
		x:         struct{ E error }{nil},
		y:         struct{ E error }{AnyError},
		opts:      []cmp.Option{EquateErrors()},
		wantEqual: false,
		reason:    "AnyError is not equal to nil value",
	}, {
		label:     "IgnoreFields",
		x:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 5}}}},
		y:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 6}}}},
		wantEqual: false,
		reason:    "not equal because values do not match in deeply embedded field",
	}, {
		label:     "IgnoreFields",
		x:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 5}}}},
		y:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 6}}}},
		opts:      []cmp.Option{IgnoreFields(Bar1{}, "Alpha")},
		wantEqual: true,
		reason:    "equal because IgnoreField ignores deeply embedded field: Alpha",
	}, {
		label:     "IgnoreFields",
		x:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 5}}}},
		y:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 6}}}},
		opts:      []cmp.Option{IgnoreFields(Bar1{}, "Foo1.Alpha")},
		wantEqual: true,
		reason:    "equal because IgnoreField ignores deeply embedded field: Foo1.Alpha",
	}, {
		label:     "IgnoreFields",
		x:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 5}}}},
		y:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 6}}}},
		opts:      []cmp.Option{IgnoreFields(Bar1{}, "Foo2.Alpha")},
		wantEqual: true,
		reason:    "equal because IgnoreField ignores deeply embedded field: Foo2.Alpha",
	}, {
		label:     "IgnoreFields",
		x:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 5}}}},
		y:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 6}}}},
		opts:      []cmp.Option{IgnoreFields(Bar1{}, "Foo3.Alpha")},
		wantEqual: true,
		reason:    "equal because IgnoreField ignores deeply embedded field: Foo3.Alpha",
	}, {
		label:     "IgnoreFields",
		x:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 5}}}},
		y:         Bar1{Foo3{&Foo2{&Foo1{Alpha: 6}}}},
		opts:      []cmp.Option{IgnoreFields(Bar1{}, "Foo3.Foo2.Alpha")},
		wantEqual: true,
		reason:    "equal because IgnoreField ignores deeply embedded field: Foo3.Foo2.Alpha",
	}, {
		label:     "IgnoreFields",
		x:         createBar3X(),
		y:         createBar3Y(),
		wantEqual: false,
		reason:    "not equal because many deeply nested or embedded fields differ",
	}, {
		label:     "IgnoreFields",
		x:         createBar3X(),
		y:         createBar3Y(),
		opts:      []cmp.Option{IgnoreFields(Bar3{}, "Bar1", "Bravo", "Delta", "Foo3", "Alpha")},
		wantEqual: true,
		reason:    "equal because IgnoreFields ignores fields at the highest levels",
	}, {
		label: "IgnoreFields",
		x:     createBar3X(),
		y:     createBar3Y(),
		opts: []cmp.Option{
			IgnoreFields(Bar3{},
				"Bar1.Foo3.Bravo",
				"Bravo.Bar1.Foo3.Foo2.Foo1.Charlie",
				"Bravo.Foo3.Foo2.Foo1.Bravo",
				"Bravo.Bravo",
				"Delta.Echo.Charlie",
				"Foo3.Foo2.Foo1.Alpha",
				"Alpha",
			),
		},
		wantEqual: true,
		reason:    "equal because IgnoreFields ignores fields using fully-qualified field",
	}, {
		label: "IgnoreFields",
		x:     createBar3X(),
		y:     createBar3Y(),
		opts: []cmp.Option{
			IgnoreFields(Bar3{},
				"Bar1.Foo3.Bravo",
				"Bravo.Foo3.Foo2.Foo1.Bravo",
				"Bravo.Bravo",
				"Delta.Echo.Charlie",
				"Foo3.Foo2.Foo1.Alpha",
				"Alpha",
			),
		},
		wantEqual: false,
		reason:    "not equal because one fully-qualified field is not ignored: Bravo.Bar1.Foo3.Foo2.Foo1.Charlie",
	}, {
		label:     "IgnoreFields",
		x:         createBar3X(),
		y:         createBar3Y(),
		opts:      []cmp.Option{IgnoreFields(Bar3{}, "Bar1", "Bravo", "Delta", "Alpha")},
		wantEqual: false,
		reason:    "not equal because highest-level field is not ignored: Foo3",
	}, {
		label: "IgnoreFields",
		x: ParentStruct{
			privateStruct: &privateStruct{private: 1},
			PublicStruct:  &PublicStruct{private: 2},
			private:       3,
		},
		y: ParentStruct{
			privateStruct: &privateStruct{private: 10},
			PublicStruct:  &PublicStruct{private: 20},
			private:       30,
		},
		opts:      []cmp.Option{cmp.AllowUnexported(ParentStruct{}, PublicStruct{}, privateStruct{})},
		wantEqual: false,
		reason:    "not equal because unexported fields mismatch",
	}, {
		label: "IgnoreFields",
		x: ParentStruct{
			privateStruct: &privateStruct{private: 1},
			PublicStruct:  &PublicStruct{private: 2},
			private:       3,
		},
		y: ParentStruct{
			privateStruct: &privateStruct{private: 10},
			PublicStruct:  &PublicStruct{private: 20},
			private:       30,
		},
		opts: []cmp.Option{
			cmp.AllowUnexported(ParentStruct{}, PublicStruct{}, privateStruct{}),
			IgnoreFields(ParentStruct{}, "PublicStruct.private", "privateStruct.private", "private"),
		},
		wantEqual: true,
		reason:    "equal because mismatching unexported fields are ignored",
	}, {
		label:     "IgnoreTypes",
		x:         []interface{}{5, "same"},
		y:         []interface{}{6, "same"},
		wantEqual: false,
		reason:    "not equal because 5 != 6",
	}, {
		label:     "IgnoreTypes",
		x:         []interface{}{5, "same"},
		y:         []interface{}{6, "same"},
		opts:      []cmp.Option{IgnoreTypes(0)},
		wantEqual: true,
		reason:    "equal because ints are ignored",
	}, {
		label:     "IgnoreTypes+IgnoreInterfaces",
		x:         []interface{}{5, "same", new(bytes.Buffer)},
		y:         []interface{}{6, "same", new(bytes.Buffer)},
		opts:      []cmp.Option{IgnoreTypes(0)},
		wantPanic: true,
		reason:    "panics because bytes.Buffer has unexported fields",
	}, {
		label: "IgnoreTypes+IgnoreInterfaces",
		x:     []interface{}{5, "same", new(bytes.Buffer)},
		y:     []interface{}{6, "diff", new(bytes.Buffer)},
		opts: []cmp.Option{
			IgnoreTypes(0, ""),
			IgnoreInterfaces(struct{ io.Reader }{}),
		},
		wantEqual: true,
		reason:    "equal because bytes.Buffer is ignored by match on interface type",
	}, {
		label: "IgnoreTypes+IgnoreInterfaces",
		x:     []interface{}{5, "same", new(bytes.Buffer)},
		y:     []interface{}{6, "same", new(bytes.Buffer)},
		opts: []cmp.Option{
			IgnoreTypes(0, ""),
			IgnoreInterfaces(struct {
				io.Reader
				io.Writer
				fmt.Stringer
			}{}),
		},
		wantEqual: true,
		reason:    "equal because bytes.Buffer is ignored by match on multiple interface types",
	}, {
		label:     "IgnoreInterfaces",
		x:         struct{ mu sync.Mutex }{},
		y:         struct{ mu sync.Mutex }{},
		wantPanic: true,
		reason:    "panics because sync.Mutex has unexported fields",
	}, {
		label:     "IgnoreInterfaces",
		x:         struct{ mu sync.Mutex }{},
		y:         struct{ mu sync.Mutex }{},
		opts:      []cmp.Option{IgnoreInterfaces(struct{ sync.Locker }{})},
		wantEqual: true,
		reason:    "equal because IgnoreInterfaces applies on values (with pointer receiver)",
	}, {
		label:     "IgnoreInterfaces",
		x:         struct{ mu *sync.Mutex }{},
		y:         struct{ mu *sync.Mutex }{},
		opts:      []cmp.Option{IgnoreInterfaces(struct{ sync.Locker }{})},
		wantEqual: true,
		reason:    "equal because IgnoreInterfaces applies on pointers",
	}, {
		label:     "IgnoreUnexported",
		x:         ParentStruct{Public: 1, private: 2},
		y:         ParentStruct{Public: 1, private: -2},
		opts:      []cmp.Option{cmp.AllowUnexported(ParentStruct{})},
		wantEqual: false,
		reason:    "not equal because ParentStruct.private differs with AllowUnexported",
	}, {
		label:     "IgnoreUnexported",
		x:         ParentStruct{Public: 1, private: 2},
		y:         ParentStruct{Public: 1, private: -2},
		opts:      []cmp.Option{IgnoreUnexported(ParentStruct{})},
		wantEqual: true,
		reason:    "equal because IgnoreUnexported ignored ParentStruct.private",
	}, {
		label: "IgnoreUnexported",
		x:     ParentStruct{Public: 1, private: 2, PublicStruct: &PublicStruct{Public: 3, private: 4}},
		y:     ParentStruct{Public: 1, private: -2, PublicStruct: &PublicStruct{Public: 3, private: 4}},
		opts: []cmp.Option{
			cmp.AllowUnexported(PublicStruct{}),
			IgnoreUnexported(ParentStruct{}),
		},
		wantEqual: true,
		reason:    "equal because ParentStruct.private is ignored",
	}, {
		label: "IgnoreUnexported",
		x:     ParentStruct{Public: 1, private: 2, PublicStruct: &PublicStruct{Public: 3, private: 4}},
		y:     ParentStruct{Public: 1, private: -2, PublicStruct: &PublicStruct{Public: 3, private: -4}},
		opts: []cmp.Option{
			cmp.AllowUnexported(PublicStruct{}),
			IgnoreUnexported(ParentStruct{}),
		},
		wantEqual: false,
		reason:    "not equal because ParentStruct.PublicStruct.private differs and not ignored by IgnoreUnexported(ParentStruct{})",
	}, {
		label: "IgnoreUnexported",
		x:     ParentStruct{Public: 1, private: 2, PublicStruct: &PublicStruct{Public: 3, private: 4}},
		y:     ParentStruct{Public: 1, private: -2, PublicStruct: &PublicStruct{Public: 3, private: -4}},
		opts: []cmp.Option{
			IgnoreUnexported(ParentStruct{}, PublicStruct{}),
		},
		wantEqual: true,
		reason:    "equal because both ParentStruct.PublicStruct and ParentStruct.PublicStruct.private are ignored",
	}, {
		label: "IgnoreUnexported",
		x:     ParentStruct{Public: 1, private: 2, privateStruct: &privateStruct{Public: 3, private: 4}},
		y:     ParentStruct{Public: 1, private: 2, privateStruct: &privateStruct{Public: -3, private: -4}},
		opts: []cmp.Option{
			cmp.AllowUnexported(privateStruct{}, PublicStruct{}, ParentStruct{}),
		},
		wantEqual: false,
		reason:    "not equal since ParentStruct.privateStruct differs",
	}, {
		label: "IgnoreUnexported",
		x:     ParentStruct{Public: 1, private: 2, privateStruct: &privateStruct{Public: 3, private: 4}},
		y:     ParentStruct{Public: 1, private: 2, privateStruct: &privateStruct{Public: -3, private: -4}},
		opts: []cmp.Option{
			cmp.AllowUnexported(privateStruct{}, PublicStruct{}),
			IgnoreUnexported(ParentStruct{}),
		},
		wantEqual: true,
		reason:    "equal because ParentStruct.privateStruct ignored by IgnoreUnexported(ParentStruct{})",
	}, {
		label: "IgnoreUnexported",
		x:     ParentStruct{Public: 1, private: 2, privateStruct: &privateStruct{Public: 3, private: 4}},
		y:     ParentStruct{Public: 1, private: 2, privateStruct: &privateStruct{Public: 3, private: -4}},
		opts: []cmp.Option{
			cmp.AllowUnexported(PublicStruct{}, ParentStruct{}),
			IgnoreUnexported(privateStruct{}),
		},
		wantEqual: true,
		reason:    "equal because privateStruct.private ignored by IgnoreUnexported(privateStruct{})",
	}, {
		label: "IgnoreUnexported",
		x:     ParentStruct{Public: 1, private: 2, privateStruct: &privateStruct{Public: 3, private: 4}},
		y:     ParentStruct{Public: 1, private: 2, privateStruct: &privateStruct{Public: -3, private: -4}},
		opts: []cmp.Option{
			cmp.AllowUnexported(PublicStruct{}, ParentStruct{}),
			IgnoreUnexported(privateStruct{}),
		},
		wantEqual: false,
		reason:    "not equal because privateStruct.Public differs and not ignored by IgnoreUnexported(privateStruct{})",
	}, {
		label: "IgnoreFields+IgnoreTypes+IgnoreUnexported",
		x: &Everything{
			MyInt:   5,
			MyFloat: 3.3,
			MyTime:  MyTime{time.Now()},
			Bar3:    *createBar3X(),
			ParentStruct: ParentStruct{
				Public: 1, private: 2, PublicStruct: &PublicStruct{Public: 3, private: 4},
			},
		},
		y: &Everything{
			MyInt:   -5,
			MyFloat: 3.3,
			MyTime:  MyTime{time.Now()},
			Bar3:    *createBar3Y(),
			ParentStruct: ParentStruct{
				Public: 1, private: -2, PublicStruct: &PublicStruct{Public: -3, private: -4},
			},
		},
		opts: []cmp.Option{
			IgnoreFields(Everything{}, "MyTime", "Bar3.Foo3"),
			IgnoreFields(Bar3{}, "Bar1", "Bravo", "Delta", "Alpha"),
			IgnoreTypes(MyInt(0), PublicStruct{}),
			IgnoreUnexported(ParentStruct{}),
		},
		wantEqual: true,
		reason:    "equal because all Ignore options can be composed together",
	}, {
		label: "IgnoreSliceElements",
		x:     []int{1, 0, 2, 3, 0, 4, 0, 0},
		y:     []int{0, 0, 0, 0, 1, 2, 3, 4},
		opts: []cmp.Option{
			IgnoreSliceElements(func(v int) bool { return v == 0 }),
		},
		wantEqual: true,
		reason:    "equal because zero elements are ignored",
	}, {
		label: "IgnoreSliceElements",
		x:     []MyInt{1, 0, 2, 3, 0, 4, 0, 0},
		y:     []MyInt{0, 0, 0, 0, 1, 2, 3, 4},
		opts: []cmp.Option{
			IgnoreSliceElements(func(v int) bool { return v == 0 }),
		},
		wantEqual: false,
		reason:    "not equal because MyInt is not assignable to int",
	}, {
		label: "IgnoreSliceElements",
		x:     MyInts{1, 0, 2, 3, 0, 4, 0, 0},
		y:     MyInts{0, 0, 0, 0, 1, 2, 3, 4},
		opts: []cmp.Option{
			IgnoreSliceElements(func(v int) bool { return v == 0 }),
		},
		wantEqual: true,
		reason:    "equal because the element type of MyInts is assignable to int",
	}, {
		label: "IgnoreSliceElements+EquateEmpty",
		x:     []MyInt{},
		y:     []MyInt{0, 0, 0, 0},
		opts: []cmp.Option{
			IgnoreSliceElements(func(v int) bool { return v == 0 }),
			EquateEmpty(),
		},
		wantEqual: false,
		reason:    "not equal because ignored elements does not imply empty slice",
	}, {
		label: "IgnoreMapEntries",
		x:     map[string]int{"one": 1, "TWO": 2, "three": 3, "FIVE": 5},
		y:     map[string]int{"one": 1, "three": 3, "TEN": 10},
		opts: []cmp.Option{
			IgnoreMapEntries(func(k string, v int) bool { return strings.ToUpper(k) == k }),
		},
		wantEqual: true,
		reason:    "equal because uppercase keys are ignored",
	}, {
		label: "IgnoreMapEntries",
		x:     map[MyString]int{"one": 1, "TWO": 2, "three": 3, "FIVE": 5},
		y:     map[MyString]int{"one": 1, "three": 3, "TEN": 10},
		opts: []cmp.Option{
			IgnoreMapEntries(func(k string, v int) bool { return strings.ToUpper(k) == k }),
		},
		wantEqual: false,
		reason:    "not equal because MyString is not assignable to string",
	}, {
		label: "IgnoreMapEntries",
		x:     map[string]MyInt{"one": 1, "TWO": 2, "three": 3, "FIVE": 5},
		y:     map[string]MyInt{"one": 1, "three": 3, "TEN": 10},
		opts: []cmp.Option{
			IgnoreMapEntries(func(k string, v int) bool { return strings.ToUpper(k) == k }),
		},
		wantEqual: false,
		reason:    "not equal because MyInt is not assignable to int",
	}, {
		label: "IgnoreMapEntries+EquateEmpty",
		x:     map[string]MyInt{"ONE": 1, "TWO": 2, "THREE": 3},
		y:     nil,
		opts: []cmp.Option{
			IgnoreMapEntries(func(k string, v int) bool { return strings.ToUpper(k) == k }),
			EquateEmpty(),
		},
		wantEqual: false,
		reason:    "not equal because ignored entries does not imply empty map",
	}, {
		label: "AcyclicTransformer",
		x:     "a\nb\nc\nd",
		y:     "a\nb\nd\nd",
		opts: []cmp.Option{
			AcyclicTransformer("", func(s string) []string { return strings.Split(s, "\n") }),
		},
		wantEqual: false,
		reason:    "not equal because 3rd line differs, but should not recurse infinitely",
	}, {
		label: "AcyclicTransformer",
		x:     []string{"foo", "Bar", "BAZ"},
		y:     []string{"Foo", "BAR", "baz"},
		opts: []cmp.Option{
			AcyclicTransformer("", strings.ToUpper),
		},
		wantEqual: true,
		reason:    "equal because of strings.ToUpper; AcyclicTransformer unnecessary, but check this still works",
	}, {
		label: "AcyclicTransformer",
		x:     "this is a sentence",
		y: "this   			is a 			sentence",
		opts: []cmp.Option{
			AcyclicTransformer("", strings.Fields),
		},
		wantEqual: true,
		reason:    "equal because acyclic transformer splits on any contiguous whitespace",
	}}

	for _, tt := range tests {
		t.Run(tt.label, func(t *testing.T) {
			var gotEqual bool
			var gotPanic string
			func() {
				defer func() {
					if ex := recover(); ex != nil {
						gotPanic = fmt.Sprint(ex)
					}
				}()
				gotEqual = cmp.Equal(tt.x, tt.y, tt.opts...)
			}()
			switch {
			case tt.reason == "":
				t.Errorf("reason must be provided")
			case gotPanic == "" && tt.wantPanic:
				t.Errorf("expected Equal panic\nreason: %s", tt.reason)
			case gotPanic != "" && !tt.wantPanic:
				t.Errorf("unexpected Equal panic: got %v\nreason: %v", gotPanic, tt.reason)
			case gotEqual != tt.wantEqual:
				t.Errorf("Equal = %v, want %v\nreason: %v", gotEqual, tt.wantEqual, tt.reason)
			}
		})
	}
}

func TestPanic(t *testing.T) {
	args := func(x ...interface{}) []interface{} { return x }
	tests := []struct {
		label     string        // Test name
		fnc       interface{}   // Option function to call
		args      []interface{} // Arguments to pass in
		wantPanic string        // Expected panic message
		reason    string        // The reason for the expected outcome
	}{{
		label:  "EquateApprox",
		fnc:    EquateApprox,
		args:   args(0.0, 0.0),
		reason: "zero margin and fraction is equivalent to exact equality",
	}, {
		label:     "EquateApprox",
		fnc:       EquateApprox,
		args:      args(-0.1, 0.0),
		wantPanic: "margin or fraction must be a non-negative number",
		reason:    "negative inputs are invalid",
	}, {
		label:     "EquateApprox",
		fnc:       EquateApprox,
		args:      args(0.0, -0.1),
		wantPanic: "margin or fraction must be a non-negative number",
		reason:    "negative inputs are invalid",
	}, {
		label:     "EquateApprox",
		fnc:       EquateApprox,
		args:      args(math.NaN(), 0.0),
		wantPanic: "margin or fraction must be a non-negative number",
		reason:    "NaN inputs are invalid",
	}, {
		label:  "EquateApprox",
		fnc:    EquateApprox,
		args:   args(1.0, 0.0),
		reason: "fraction of 1.0 or greater is valid",
	}, {
		label:  "EquateApprox",
		fnc:    EquateApprox,
		args:   args(0.0, math.Inf(+1)),
		reason: "margin of infinity is valid",
	}, {
		label:     "EquateApproxTime",
		fnc:       EquateApproxTime,
		args:      args(time.Duration(-1)),
		wantPanic: "margin must be a non-negative number",
		reason:    "negative duration is invalid",
	}, {
		label:     "SortSlices",
		fnc:       SortSlices,
		args:      args(strings.Compare),
		wantPanic: "invalid less function",
		reason:    "func(x, y string) int is wrong signature for less",
	}, {
		label:     "SortSlices",
		fnc:       SortSlices,
		args:      args((func(_, _ int) bool)(nil)),
		wantPanic: "invalid less function",
		reason:    "nil value is not valid",
	}, {
		label:     "SortMaps",
		fnc:       SortMaps,
		args:      args(strings.Compare),
		wantPanic: "invalid less function",
		reason:    "func(x, y string) int is wrong signature for less",
	}, {
		label:     "SortMaps",
		fnc:       SortMaps,
		args:      args((func(_, _ int) bool)(nil)),
		wantPanic: "invalid less function",
		reason:    "nil value is not valid",
	}, {
		label:     "IgnoreFields",
		fnc:       IgnoreFields,
		args:      args(Foo1{}, ""),
		wantPanic: "name must not be empty",
		reason:    "empty selector is invalid",
	}, {
		label:     "IgnoreFields",
		fnc:       IgnoreFields,
		args:      args(Foo1{}, "."),
		wantPanic: "name must not be empty",
		reason:    "single dot selector is invalid",
	}, {
		label:  "IgnoreFields",
		fnc:    IgnoreFields,
		args:   args(Foo1{}, ".Alpha"),
		reason: "dot-prefix is okay since Foo1.Alpha reads naturally",
	}, {
		label:     "IgnoreFields",
		fnc:       IgnoreFields,
		args:      args(Foo1{}, "Alpha."),
		wantPanic: "name must not be empty",
		reason:    "dot-suffix is invalid",
	}, {
		label:     "IgnoreFields",
		fnc:       IgnoreFields,
		args:      args(Foo1{}, "Alpha "),
		wantPanic: "does not exist",
		reason:    "identifiers must not have spaces",
	}, {
		label:     "IgnoreFields",
		fnc:       IgnoreFields,
		args:      args(Foo1{}, "Zulu"),
		wantPanic: "does not exist",
		reason:    "name of non-existent field is invalid",
	}, {
		label:     "IgnoreFields",
		fnc:       IgnoreFields,
		args:      args(Foo1{}, "Alpha.NoExist"),
		wantPanic: "must be a struct",
		reason:    "cannot select into a non-struct",
	}, {
		label:     "IgnoreFields",
		fnc:       IgnoreFields,
		args:      args(&Foo1{}, "Alpha"),
		wantPanic: "must be a non-pointer struct",
		reason:    "the type must be a struct (not pointer to a struct)",
	}, {
		label:  "IgnoreFields",
		fnc:    IgnoreFields,
		args:   args(struct{ privateStruct }{}, "privateStruct"),
		reason: "privateStruct field permitted since it is the default name of the embedded type",
	}, {
		label:  "IgnoreFields",
		fnc:    IgnoreFields,
		args:   args(struct{ privateStruct }{}, "Public"),
		reason: "Public field permitted since it is a forwarded field that is exported",
	}, {
		label:     "IgnoreFields",
		fnc:       IgnoreFields,
		args:      args(struct{ privateStruct }{}, "private"),
		wantPanic: "does not exist",
		reason:    "private field not permitted since it is a forwarded field that is unexported",
	}, {
		label:  "IgnoreTypes",
		fnc:    IgnoreTypes,
		reason: "empty input is valid",
	}, {
		label:     "IgnoreTypes",
		fnc:       IgnoreTypes,
		args:      args(nil),
		wantPanic: "cannot determine type",
		reason:    "input must not be nil value",
	}, {
		label:  "IgnoreTypes",
		fnc:    IgnoreTypes,
		args:   args(0, 0, 0),
		reason: "duplicate inputs of the same type is valid",
	}, {
		label:     "IgnoreInterfaces",
		fnc:       IgnoreInterfaces,
		args:      args(nil),
		wantPanic: "input must be an anonymous struct",
		reason:    "input must not be nil value",
	}, {
		label:     "IgnoreInterfaces",
		fnc:       IgnoreInterfaces,
		args:      args(Foo1{}),
		wantPanic: "input must be an anonymous struct",
		reason:    "input must not be a named struct type",
	}, {
		label:     "IgnoreInterfaces",
		fnc:       IgnoreInterfaces,
		args:      args(struct{ _ io.Reader }{}),
		wantPanic: "struct cannot have named fields",
		reason:    "input must not have named fields",
	}, {
		label:     "IgnoreInterfaces",
		fnc:       IgnoreInterfaces,
		args:      args(struct{ Foo1 }{}),
		wantPanic: "embedded field must be an interface type",
		reason:    "field types must be interfaces",
	}, {
		label:     "IgnoreInterfaces",
		fnc:       IgnoreInterfaces,
		args:      args(struct{ EmptyInterface }{}),
		wantPanic: "cannot ignore empty interface",
		reason:    "field types must not be the empty interface",
	}, {
		label: "IgnoreInterfaces",
		fnc:   IgnoreInterfaces,
		args: args(struct {
			io.Reader
			io.Writer
			io.Closer
			io.ReadWriteCloser
		}{}),
		reason: "multiple interfaces may be specified, even if they overlap",
	}, {
		label:  "IgnoreUnexported",
		fnc:    IgnoreUnexported,
		reason: "empty input is valid",
	}, {
		label:     "IgnoreUnexported",
		fnc:       IgnoreUnexported,
		args:      args(nil),
		wantPanic: "must be a non-pointer struct",
		reason:    "input must not be nil value",
	}, {
		label:     "IgnoreUnexported",
		fnc:       IgnoreUnexported,
		args:      args(&Foo1{}),
		wantPanic: "must be a non-pointer struct",
		reason:    "input must be a struct type (not a pointer to a struct)",
	}, {
		label:  "IgnoreUnexported",
		fnc:    IgnoreUnexported,
		args:   args(Foo1{}, struct{ x, X int }{}),
		reason: "input may be named or unnamed structs",
	}, {
		label:     "AcyclicTransformer",
		fnc:       AcyclicTransformer,
		args:      args("", "not a func"),
		wantPanic: "invalid transformer function",
		reason:    "AcyclicTransformer has same input requirements as Transformer",
	}}

	for _, tt := range tests {
		t.Run(tt.label, func(t *testing.T) {
			// Prepare function arguments.
			vf := reflect.ValueOf(tt.fnc)
			var vargs []reflect.Value
			for i, arg := range tt.args {
				if arg == nil {
					tf := vf.Type()
					if i == tf.NumIn()-1 && tf.IsVariadic() {
						vargs = append(vargs, reflect.Zero(tf.In(i).Elem()))
					} else {
						vargs = append(vargs, reflect.Zero(tf.In(i)))
					}
				} else {
					vargs = append(vargs, reflect.ValueOf(arg))
				}
			}

			// Call the function and capture any panics.
			var gotPanic string
			func() {
				defer func() {
					if ex := recover(); ex != nil {
						if s, ok := ex.(string); ok {
							gotPanic = s
						} else {
							panic(ex)
						}
					}
				}()
				vf.Call(vargs)
			}()

			switch {
			case tt.reason == "":
				t.Errorf("reason must be provided")
			case tt.wantPanic == "" && gotPanic != "":
				t.Errorf("unexpected panic message: %s\nreason: %s", gotPanic, tt.reason)
			case tt.wantPanic != "" && !strings.Contains(gotPanic, tt.wantPanic):
				t.Errorf("panic message:\ngot:  %s\nwant: %s\nreason: %s", gotPanic, tt.wantPanic, tt.reason)
			}
		})
	}
}
