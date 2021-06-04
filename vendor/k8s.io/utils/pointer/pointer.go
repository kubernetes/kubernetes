/*
Copyright 2018 The Kubernetes Authors.

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

package pointer

import (
	"fmt"
	"reflect"
)

// AllPtrFieldsNil tests whether all pointer fields in a struct are nil.  This is useful when,
// for example, an API struct is handled by plugins which need to distinguish
// "no plugin accepted this spec" from "this spec is empty".
//
// This function is only valid for structs and pointers to structs.  Any other
// type will cause a panic.  Passing a typed nil pointer will return true.
func AllPtrFieldsNil(obj interface{}) bool {
	v := reflect.ValueOf(obj)
	if !v.IsValid() {
		panic(fmt.Sprintf("reflect.ValueOf() produced a non-valid Value for %#v", obj))
	}
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return true
		}
		v = v.Elem()
	}
	for i := 0; i < v.NumField(); i++ {
		if v.Field(i).Kind() == reflect.Ptr && !v.Field(i).IsNil() {
			return false
		}
	}
	return true
}

// Int32 returns a pointer to an int32.
func Int32(i int32) *int32 {
	return &i
}

var Int32Ptr = Int32 // for back-compat

// Int32Deref dereferences the int32 ptr and returns it if not nil, or else
// returns def.
func Int32Deref(ptr *int32, def int32) int32 {
	if ptr != nil {
		return *ptr
	}
	return def
}

var Int32PtrDerefOr = Int32Deref // for back-compat

// Int32Equal returns true if both arguments are nil or both arguments
// dereference to the same value.
func Int32Equal(a, b *int32) bool {
	if (a == nil) != (b == nil) {
		return false
	}
	if a == nil {
		return true
	}
	return *a == *b
}

// Int64 returns a pointer to an int64.
func Int64(i int64) *int64 {
	return &i
}

var Int64Ptr = Int64 // for back-compat

// Int64Deref dereferences the int64 ptr and returns it if not nil, or else
// returns def.
func Int64Deref(ptr *int64, def int64) int64 {
	if ptr != nil {
		return *ptr
	}
	return def
}

var Int64PtrDerefOr = Int64Deref // for back-compat

// Int64Equal returns true if both arguments are nil or both arguments
// dereference to the same value.
func Int64Equal(a, b *int64) bool {
	if (a == nil) != (b == nil) {
		return false
	}
	if a == nil {
		return true
	}
	return *a == *b
}

// Bool returns a pointer to a bool.
func Bool(b bool) *bool {
	return &b
}

var BoolPtr = Bool // for back-compat

// BoolDeref dereferences the bool ptr and returns it if not nil, or else
// returns def.
func BoolDeref(ptr *bool, def bool) bool {
	if ptr != nil {
		return *ptr
	}
	return def
}

var BoolPtrDerefOr = BoolDeref // for back-compat

// BoolEqual returns true if both arguments are nil or both arguments
// dereference to the same value.
func BoolEqual(a, b *bool) bool {
	if (a == nil) != (b == nil) {
		return false
	}
	if a == nil {
		return true
	}
	return *a == *b
}

// String returns a pointer to a string.
func String(s string) *string {
	return &s
}

var StringPtr = String // for back-compat

// StringDeref dereferences the string ptr and returns it if not nil, or else
// returns def.
func StringDeref(ptr *string, def string) string {
	if ptr != nil {
		return *ptr
	}
	return def
}

var StringPtrDerefOr = StringDeref // for back-compat

// StringEqual returns true if both arguments are nil or both arguments
// dereference to the same value.
func StringEqual(a, b *string) bool {
	if (a == nil) != (b == nil) {
		return false
	}
	if a == nil {
		return true
	}
	return *a == *b
}

// Float32 returns a pointer to the a float32.
func Float32(i float32) *float32 {
	return &i
}

var Float32Ptr = Float32

// Float32Deref dereferences the float32 ptr and returns it if not nil, or else
// returns def.
func Float32Deref(ptr *float32, def float32) float32 {
	if ptr != nil {
		return *ptr
	}
	return def
}

var Float32PtrDerefOr = Float32Deref // for back-compat

// Float32Equal returns true if both arguments are nil or both arguments
// dereference to the same value.
func Float32Equal(a, b *float32) bool {
	if (a == nil) != (b == nil) {
		return false
	}
	if a == nil {
		return true
	}
	return *a == *b
}

// Float64 returns a pointer to the a float64.
func Float64(i float64) *float64 {
	return &i
}

var Float64Ptr = Float64

// Float64Deref dereferences the float64 ptr and returns it if not nil, or else
// returns def.
func Float64Deref(ptr *float64, def float64) float64 {
	if ptr != nil {
		return *ptr
	}
	return def
}

var Float64PtrDerefOr = Float64Deref // for back-compat

// Float64Equal returns true if both arguments are nil or both arguments
// dereference to the same value.
func Float64Equal(a, b *float64) bool {
	if (a == nil) != (b == nil) {
		return false
	}
	if a == nil {
		return true
	}
	return *a == *b
}
