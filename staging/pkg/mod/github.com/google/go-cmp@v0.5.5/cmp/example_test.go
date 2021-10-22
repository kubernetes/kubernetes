// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmp_test

import (
	"fmt"
	"math"
	"net"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/google/go-cmp/cmp"
)

// TODO: Re-write these examples in terms of how you actually use the
// fundamental options and filters and not in terms of what cool things you can
// do with them since that overlaps with cmp/cmpopts.

// Use Diff to print out a human-readable report of differences for tests
// comparing nested or structured data.
func ExampleDiff_testing() {
	// Let got be the hypothetical value obtained from some logic under test
	// and want be the expected golden data.
	got, want := MakeGatewayInfo()

	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("MakeGatewayInfo() mismatch (-want +got):\n%s", diff)
	}

	// Output:
	// MakeGatewayInfo() mismatch (-want +got):
	//   cmp_test.Gateway{
	//   	SSID:      "CoffeeShopWiFi",
	// - 	IPAddress: s"192.168.0.2",
	// + 	IPAddress: s"192.168.0.1",
	//   	NetMask:   {0xff, 0xff, 0x00, 0x00},
	//   	Clients: []cmp_test.Client{
	//   		... // 2 identical elements
	//   		{Hostname: "macchiato", IPAddress: s"192.168.0.153", LastSeen: s"2009-11-10 23:39:43 +0000 UTC"},
	//   		{Hostname: "espresso", IPAddress: s"192.168.0.121"},
	//   		{
	//   			Hostname:  "latte",
	// - 			IPAddress: s"192.168.0.221",
	// + 			IPAddress: s"192.168.0.219",
	//   			LastSeen:  s"2009-11-10 23:00:23 +0000 UTC",
	//   		},
	// + 		{
	// + 			Hostname:  "americano",
	// + 			IPAddress: s"192.168.0.188",
	// + 			LastSeen:  s"2009-11-10 23:03:05 +0000 UTC",
	// + 		},
	//   	},
	//   }
}

// Approximate equality for floats can be handled by defining a custom
// comparer on floats that determines two values to be equal if they are within
// some range of each other.
//
// This example is for demonstrative purposes; use cmpopts.EquateApprox instead.
func ExampleOption_approximateFloats() {
	// This Comparer only operates on float64.
	// To handle float32s, either define a similar function for that type
	// or use a Transformer to convert float32s into float64s.
	opt := cmp.Comparer(func(x, y float64) bool {
		delta := math.Abs(x - y)
		mean := math.Abs(x+y) / 2.0
		return delta/mean < 0.00001
	})

	x := []float64{1.0, 1.1, 1.2, math.Pi}
	y := []float64{1.0, 1.1, 1.2, 3.14159265359} // Accurate enough to Pi
	z := []float64{1.0, 1.1, 1.2, 3.1415}        // Diverges too far from Pi

	fmt.Println(cmp.Equal(x, y, opt))
	fmt.Println(cmp.Equal(y, z, opt))
	fmt.Println(cmp.Equal(z, x, opt))

	// Output:
	// true
	// false
	// false
}

// Normal floating-point arithmetic defines == to be false when comparing
// NaN with itself. In certain cases, this is not the desired property.
//
// This example is for demonstrative purposes; use cmpopts.EquateNaNs instead.
func ExampleOption_equalNaNs() {
	// This Comparer only operates on float64.
	// To handle float32s, either define a similar function for that type
	// or use a Transformer to convert float32s into float64s.
	opt := cmp.Comparer(func(x, y float64) bool {
		return (math.IsNaN(x) && math.IsNaN(y)) || x == y
	})

	x := []float64{1.0, math.NaN(), math.E, -0.0, +0.0}
	y := []float64{1.0, math.NaN(), math.E, -0.0, +0.0}
	z := []float64{1.0, math.NaN(), math.Pi, -0.0, +0.0} // Pi constant instead of E

	fmt.Println(cmp.Equal(x, y, opt))
	fmt.Println(cmp.Equal(y, z, opt))
	fmt.Println(cmp.Equal(z, x, opt))

	// Output:
	// true
	// false
	// false
}

// To have floating-point comparisons combine both properties of NaN being
// equal to itself and also approximate equality of values, filters are needed
// to restrict the scope of the comparison so that they are composable.
//
// This example is for demonstrative purposes;
// use cmpopts.EquateNaNs and cmpopts.EquateApprox instead.
func ExampleOption_equalNaNsAndApproximateFloats() {
	alwaysEqual := cmp.Comparer(func(_, _ interface{}) bool { return true })

	opts := cmp.Options{
		// This option declares that a float64 comparison is equal only if
		// both inputs are NaN.
		cmp.FilterValues(func(x, y float64) bool {
			return math.IsNaN(x) && math.IsNaN(y)
		}, alwaysEqual),

		// This option declares approximate equality on float64s only if
		// both inputs are not NaN.
		cmp.FilterValues(func(x, y float64) bool {
			return !math.IsNaN(x) && !math.IsNaN(y)
		}, cmp.Comparer(func(x, y float64) bool {
			delta := math.Abs(x - y)
			mean := math.Abs(x+y) / 2.0
			return delta/mean < 0.00001
		})),
	}

	x := []float64{math.NaN(), 1.0, 1.1, 1.2, math.Pi}
	y := []float64{math.NaN(), 1.0, 1.1, 1.2, 3.14159265359} // Accurate enough to Pi
	z := []float64{math.NaN(), 1.0, 1.1, 1.2, 3.1415}        // Diverges too far from Pi

	fmt.Println(cmp.Equal(x, y, opts))
	fmt.Println(cmp.Equal(y, z, opts))
	fmt.Println(cmp.Equal(z, x, opts))

	// Output:
	// true
	// false
	// false
}

// Sometimes, an empty map or slice is considered equal to an allocated one
// of zero length.
//
// This example is for demonstrative purposes; use cmpopts.EquateEmpty instead.
func ExampleOption_equalEmpty() {
	alwaysEqual := cmp.Comparer(func(_, _ interface{}) bool { return true })

	// This option handles slices and maps of any type.
	opt := cmp.FilterValues(func(x, y interface{}) bool {
		vx, vy := reflect.ValueOf(x), reflect.ValueOf(y)
		return (vx.IsValid() && vy.IsValid() && vx.Type() == vy.Type()) &&
			(vx.Kind() == reflect.Slice || vx.Kind() == reflect.Map) &&
			(vx.Len() == 0 && vy.Len() == 0)
	}, alwaysEqual)

	type S struct {
		A []int
		B map[string]bool
	}
	x := S{nil, make(map[string]bool, 100)}
	y := S{make([]int, 0, 200), nil}
	z := S{[]int{0}, nil} // []int has a single element (i.e., not empty)

	fmt.Println(cmp.Equal(x, y, opt))
	fmt.Println(cmp.Equal(y, z, opt))
	fmt.Println(cmp.Equal(z, x, opt))

	// Output:
	// true
	// false
	// false
}

// Two slices may be considered equal if they have the same elements,
// regardless of the order that they appear in. Transformations can be used
// to sort the slice.
//
// This example is for demonstrative purposes; use cmpopts.SortSlices instead.
func ExampleOption_sortedSlice() {
	// This Transformer sorts a []int.
	trans := cmp.Transformer("Sort", func(in []int) []int {
		out := append([]int(nil), in...) // Copy input to avoid mutating it
		sort.Ints(out)
		return out
	})

	x := struct{ Ints []int }{[]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}
	y := struct{ Ints []int }{[]int{2, 8, 0, 9, 6, 1, 4, 7, 3, 5}}
	z := struct{ Ints []int }{[]int{0, 0, 1, 2, 3, 4, 5, 6, 7, 8}}

	fmt.Println(cmp.Equal(x, y, trans))
	fmt.Println(cmp.Equal(y, z, trans))
	fmt.Println(cmp.Equal(z, x, trans))

	// Output:
	// true
	// false
	// false
}

type otherString string

func (x otherString) Equal(y otherString) bool {
	return strings.ToLower(string(x)) == strings.ToLower(string(y))
}

// If the Equal method defined on a type is not suitable, the type can be
// dynamically transformed to be stripped of the Equal method (or any method
// for that matter).
func ExampleOption_avoidEqualMethod() {
	// Suppose otherString.Equal performs a case-insensitive equality,
	// which is too loose for our needs.
	// We can avoid the methods of otherString by declaring a new type.
	type myString otherString

	// This transformer converts otherString to myString, allowing Equal to use
	// other Options to determine equality.
	trans := cmp.Transformer("", func(in otherString) myString {
		return myString(in)
	})

	x := []otherString{"foo", "bar", "baz"}
	y := []otherString{"fOO", "bAr", "Baz"} // Same as before, but with different case

	fmt.Println(cmp.Equal(x, y))        // Equal because of case-insensitivity
	fmt.Println(cmp.Equal(x, y, trans)) // Not equal because of more exact equality

	// Output:
	// true
	// false
}

func roundF64(z float64) float64 {
	if z < 0 {
		return math.Ceil(z - 0.5)
	}
	return math.Floor(z + 0.5)
}

// The complex numbers complex64 and complex128 can really just be decomposed
// into a pair of float32 or float64 values. It would be convenient to be able
// define only a single comparator on float64 and have float32, complex64, and
// complex128 all be able to use that comparator. Transformations can be used
// to handle this.
func ExampleOption_transformComplex() {
	opts := []cmp.Option{
		// This transformer decomposes complex128 into a pair of float64s.
		cmp.Transformer("T1", func(in complex128) (out struct{ Real, Imag float64 }) {
			out.Real, out.Imag = real(in), imag(in)
			return out
		}),
		// This transformer converts complex64 to complex128 to allow the
		// above transform to take effect.
		cmp.Transformer("T2", func(in complex64) complex128 {
			return complex128(in)
		}),
		// This transformer converts float32 to float64.
		cmp.Transformer("T3", func(in float32) float64 {
			return float64(in)
		}),
		// This equality function compares float64s as rounded integers.
		cmp.Comparer(func(x, y float64) bool {
			return roundF64(x) == roundF64(y)
		}),
	}

	x := []interface{}{
		complex128(3.0), complex64(5.1 + 2.9i), float32(-1.2), float64(12.3),
	}
	y := []interface{}{
		complex128(3.1), complex64(4.9 + 3.1i), float32(-1.3), float64(11.7),
	}
	z := []interface{}{
		complex128(3.8), complex64(4.9 + 3.1i), float32(-1.3), float64(11.7),
	}

	fmt.Println(cmp.Equal(x, y, opts...))
	fmt.Println(cmp.Equal(y, z, opts...))
	fmt.Println(cmp.Equal(z, x, opts...))

	// Output:
	// true
	// false
	// false
}

type (
	Gateway struct {
		SSID      string
		IPAddress net.IP
		NetMask   net.IPMask
		Clients   []Client
	}
	Client struct {
		Hostname  string
		IPAddress net.IP
		LastSeen  time.Time
	}
)

func MakeGatewayInfo() (x, y Gateway) {
	x = Gateway{
		SSID:      "CoffeeShopWiFi",
		IPAddress: net.IPv4(192, 168, 0, 1),
		NetMask:   net.IPv4Mask(255, 255, 0, 0),
		Clients: []Client{{
			Hostname:  "ristretto",
			IPAddress: net.IPv4(192, 168, 0, 116),
		}, {
			Hostname:  "aribica",
			IPAddress: net.IPv4(192, 168, 0, 104),
			LastSeen:  time.Date(2009, time.November, 10, 23, 6, 32, 0, time.UTC),
		}, {
			Hostname:  "macchiato",
			IPAddress: net.IPv4(192, 168, 0, 153),
			LastSeen:  time.Date(2009, time.November, 10, 23, 39, 43, 0, time.UTC),
		}, {
			Hostname:  "espresso",
			IPAddress: net.IPv4(192, 168, 0, 121),
		}, {
			Hostname:  "latte",
			IPAddress: net.IPv4(192, 168, 0, 219),
			LastSeen:  time.Date(2009, time.November, 10, 23, 0, 23, 0, time.UTC),
		}, {
			Hostname:  "americano",
			IPAddress: net.IPv4(192, 168, 0, 188),
			LastSeen:  time.Date(2009, time.November, 10, 23, 3, 5, 0, time.UTC),
		}},
	}
	y = Gateway{
		SSID:      "CoffeeShopWiFi",
		IPAddress: net.IPv4(192, 168, 0, 2),
		NetMask:   net.IPv4Mask(255, 255, 0, 0),
		Clients: []Client{{
			Hostname:  "ristretto",
			IPAddress: net.IPv4(192, 168, 0, 116),
		}, {
			Hostname:  "aribica",
			IPAddress: net.IPv4(192, 168, 0, 104),
			LastSeen:  time.Date(2009, time.November, 10, 23, 6, 32, 0, time.UTC),
		}, {
			Hostname:  "macchiato",
			IPAddress: net.IPv4(192, 168, 0, 153),
			LastSeen:  time.Date(2009, time.November, 10, 23, 39, 43, 0, time.UTC),
		}, {
			Hostname:  "espresso",
			IPAddress: net.IPv4(192, 168, 0, 121),
		}, {
			Hostname:  "latte",
			IPAddress: net.IPv4(192, 168, 0, 221),
			LastSeen:  time.Date(2009, time.November, 10, 23, 0, 23, 0, time.UTC),
		}},
	}
	return x, y
}

var t fakeT

type fakeT struct{}

func (t fakeT) Errorf(format string, args ...interface{}) { fmt.Printf(format+"\n", args...) }
