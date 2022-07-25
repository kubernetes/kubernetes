// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine safe

package mat

import "reflect"

var sizeOfFloat64 = int(reflect.TypeOf(float64(0)).Size())

// offset returns the number of float64 values b[0] is after a[0].
func offset(a, b []float64) int {
	va0 := reflect.ValueOf(a).Index(0)
	vb0 := reflect.ValueOf(b).Index(0)
	if va0.Addr() == vb0.Addr() {
		return 0
	}
	// This expression must be atomic with respect to GC moves.
	// At this stage this is true, because the GC does not
	// move. See https://golang.org/issue/12445.
	return int(vb0.UnsafeAddr()-va0.UnsafeAddr()) / sizeOfFloat64
}

var sizeOfComplex128 = int(reflect.TypeOf(complex128(0)).Size())

// offsetComplex returns the number of complex128 values b[0] is after a[0].
func offsetComplex(a, b []complex128) int {
	va0 := reflect.ValueOf(a).Index(0)
	vb0 := reflect.ValueOf(b).Index(0)
	if va0.Addr() == vb0.Addr() {
		return 0
	}
	// This expression must be atomic with respect to GC moves.
	// At this stage this is true, because the GC does not
	// move. See https://golang.org/issue/12445.
	return int(vb0.UnsafeAddr()-va0.UnsafeAddr()) / sizeOfComplex128
}
