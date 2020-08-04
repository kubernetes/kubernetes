// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

package value

import (
	"fmt"
	"math"
	"reflect"
	"sort"
)

// SortKeys sorts a list of map keys, deduplicating keys if necessary.
// The type of each value must be comparable.
func SortKeys(vs []reflect.Value) []reflect.Value {
	if len(vs) == 0 {
		return vs
	}

	// Sort the map keys.
	sort.Slice(vs, func(i, j int) bool { return isLess(vs[i], vs[j]) })

	// Deduplicate keys (fails for NaNs).
	vs2 := vs[:1]
	for _, v := range vs[1:] {
		if isLess(vs2[len(vs2)-1], v) {
			vs2 = append(vs2, v)
		}
	}
	return vs2
}

// isLess is a generic function for sorting arbitrary map keys.
// The inputs must be of the same type and must be comparable.
func isLess(x, y reflect.Value) bool {
	switch x.Type().Kind() {
	case reflect.Bool:
		return !x.Bool() && y.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return x.Int() < y.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return x.Uint() < y.Uint()
	case reflect.Float32, reflect.Float64:
		fx, fy := x.Float(), y.Float()
		return fx < fy || math.IsNaN(fx) && !math.IsNaN(fy)
	case reflect.Complex64, reflect.Complex128:
		cx, cy := x.Complex(), y.Complex()
		rx, ix, ry, iy := real(cx), imag(cx), real(cy), imag(cy)
		if rx == ry || (math.IsNaN(rx) && math.IsNaN(ry)) {
			return ix < iy || math.IsNaN(ix) && !math.IsNaN(iy)
		}
		return rx < ry || math.IsNaN(rx) && !math.IsNaN(ry)
	case reflect.Ptr, reflect.UnsafePointer, reflect.Chan:
		return x.Pointer() < y.Pointer()
	case reflect.String:
		return x.String() < y.String()
	case reflect.Array:
		for i := 0; i < x.Len(); i++ {
			if isLess(x.Index(i), y.Index(i)) {
				return true
			}
			if isLess(y.Index(i), x.Index(i)) {
				return false
			}
		}
		return false
	case reflect.Struct:
		for i := 0; i < x.NumField(); i++ {
			if isLess(x.Field(i), y.Field(i)) {
				return true
			}
			if isLess(y.Field(i), x.Field(i)) {
				return false
			}
		}
		return false
	case reflect.Interface:
		vx, vy := x.Elem(), y.Elem()
		if !vx.IsValid() || !vy.IsValid() {
			return !vx.IsValid() && vy.IsValid()
		}
		tx, ty := vx.Type(), vy.Type()
		if tx == ty {
			return isLess(x.Elem(), y.Elem())
		}
		if tx.Kind() != ty.Kind() {
			return vx.Kind() < vy.Kind()
		}
		if tx.String() != ty.String() {
			return tx.String() < ty.String()
		}
		if tx.PkgPath() != ty.PkgPath() {
			return tx.PkgPath() < ty.PkgPath()
		}
		// This can happen in rare situations, so we fallback to just comparing
		// the unique pointer for a reflect.Type. This guarantees deterministic
		// ordering within a program, but it is obviously not stable.
		return reflect.ValueOf(vx.Type()).Pointer() < reflect.ValueOf(vy.Type()).Pointer()
	default:
		// Must be Func, Map, or Slice; which are not comparable.
		panic(fmt.Sprintf("%T is not comparable", x.Type()))
	}
}
