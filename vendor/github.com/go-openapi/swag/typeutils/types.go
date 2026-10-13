// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package typeutils

import "reflect"

type zeroable interface {
	IsZero() bool
}

// IsZero returns true when the value passed into the function is a zero value.
// This allows for safer checking of interface values.
func IsZero(data any) bool {
	v := reflect.ValueOf(data)
	// check for nil data
	switch v.Kind() { //nolint:exhaustive
	case
		reflect.Interface,
		reflect.Func,
		reflect.Chan,
		reflect.Pointer,
		reflect.UnsafePointer,
		reflect.Map,
		reflect.Slice:
		if v.IsNil() {
			return true
		}
	}

	// check for things that have an IsZero method instead
	if vv, ok := data.(zeroable); ok {
		return vv.IsZero()
	}

	// continue with slightly more complex reflection
	switch v.Kind() { //nolint:exhaustive
	case reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Struct, reflect.Array:
		return reflect.DeepEqual(data, reflect.Zero(v.Type()).Interface())
	case reflect.Invalid:
		return true
	default:
		return false
	}
}

// IsNil checks if input is nil.
//
// For types chan, func, interface, map, pointer, or slice it returns true if its argument is nil.
//
// See [reflect.Value.IsNil].
func IsNil(input any) bool {
	if input == nil {
		return true
	}

	kind := reflect.TypeOf(input).Kind()
	switch kind { //nolint:exhaustive
	case reflect.Pointer,
		reflect.UnsafePointer,
		reflect.Map,
		reflect.Slice,
		reflect.Chan,
		reflect.Interface,
		reflect.Func:
		return reflect.ValueOf(input).IsNil()
	default:
		return false
	}
}
