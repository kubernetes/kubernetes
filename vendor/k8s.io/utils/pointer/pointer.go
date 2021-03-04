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

// Int32Ptr returns a pointer to an int32
func Int32Ptr(i int32) *int32 {
	return &i
}

// Int32PtrDerefOr dereference the int32 ptr and returns it if not nil,
// else returns def.
func Int32PtrDerefOr(ptr *int32, def int32) int32 {
	if ptr != nil {
		return *ptr
	}
	return def
}

// Int64Ptr returns a pointer to an int64
func Int64Ptr(i int64) *int64 {
	return &i
}

// Int64PtrDerefOr dereference the int64 ptr and returns it if not nil,
// else returns def.
func Int64PtrDerefOr(ptr *int64, def int64) int64 {
	if ptr != nil {
		return *ptr
	}
	return def
}

// BoolPtr returns a pointer to a bool
func BoolPtr(b bool) *bool {
	return &b
}

// BoolPtrDerefOr dereference the bool ptr and returns it if not nil,
// else returns def.
func BoolPtrDerefOr(ptr *bool, def bool) bool {
	if ptr != nil {
		return *ptr
	}
	return def
}

// StringPtr returns a pointer to the passed string.
func StringPtr(s string) *string {
	return &s
}

// StringPtrDerefOr dereference the string ptr and returns it if not nil,
// else returns def.
func StringPtrDerefOr(ptr *string, def string) string {
	if ptr != nil {
		return *ptr
	}
	return def
}

// Float32Ptr returns a pointer to the passed float32.
func Float32Ptr(i float32) *float32 {
	return &i
}

// Float32PtrDerefOr dereference the float32 ptr and returns it if not nil,
// else returns def.
func Float32PtrDerefOr(ptr *float32, def float32) float32 {
	if ptr != nil {
		return *ptr
	}
	return def
}

// Float64Ptr returns a pointer to the passed float64.
func Float64Ptr(i float64) *float64 {
	return &i
}

// Float64PtrDerefOr dereference the float64 ptr and returns it if not nil,
// else returns def.
func Float64PtrDerefOr(ptr *float64, def float64) float64 {
	if ptr != nil {
		return *ptr
	}
	return def
}
