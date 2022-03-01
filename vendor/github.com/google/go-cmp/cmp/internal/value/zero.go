// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package value

import (
	"math"
	"reflect"
)

// IsZero reports whether v is the zero value.
// This does not rely on Interface and so can be used on unexported fields.
func IsZero(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Bool:
		return v.Bool() == false
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return math.Float64bits(v.Float()) == 0
	case reflect.Complex64, reflect.Complex128:
		return math.Float64bits(real(v.Complex())) == 0 && math.Float64bits(imag(v.Complex())) == 0
	case reflect.String:
		return v.String() == ""
	case reflect.UnsafePointer:
		return v.Pointer() == 0
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Ptr, reflect.Map, reflect.Slice:
		return v.IsNil()
	case reflect.Array:
		for i := 0; i < v.Len(); i++ {
			if !IsZero(v.Index(i)) {
				return false
			}
		}
		return true
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if !IsZero(v.Field(i)) {
				return false
			}
		}
		return true
	}
	return false
}
