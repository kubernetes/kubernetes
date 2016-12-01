// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package reflect is a fork of go's standard library reflection package, which
// allows for deep equal with overrides
package reflect

import (
	"fmt"
	"reflect"
	"strings"
)

// During deepValueEqual, must keep track of checks that are
// in progress.  The comparison algorithm assumes that all
// checks in progress are true when it reencounters them.
// Visited comparisons are stored in a map indexed by visit.
type visit struct {
	a1  uintptr
	a2  uintptr
	typ reflect.Type
}

// unexportedTypePanic is thrown when you use this DeepEqual on something that has an
// unexported type. It indicates a programmer error, so should not occur at runtime,
// which is why it's not public and thus impossible to catch.
type unexportedTypePanic []reflect.Type

func (u unexportedTypePanic) Error() string { return u.String() }
func (u unexportedTypePanic) String() string {
	strs := make([]string, len(u))
	for i, t := range u {
		strs[i] = fmt.Sprintf("%v", t)
	}
	return "an unexported field was encountered, nested like this: " + strings.Join(strs, " -> ")
}

func makeUsefulPanic(v reflect.Value) {
	if x := recover(); x != nil {
		if u, ok := x.(unexportedTypePanic); ok {
			u = append(unexportedTypePanic{v.Type()}, u...)
			x = u
		}
		panic(x)
	}
}

// Tests for deep equality using reflected types. The map argument tracks
// comparisons that have already been seen, which allows short circuiting on
// recursive types.
func deepValueEqual(v1, v2 reflect.Value, visited map[visit]bool, depth int) bool {
	defer makeUsefulPanic(v1)

	if !v1.IsValid() || !v2.IsValid() {
		return v1.IsValid() == v2.IsValid()
	}
	if v1.Type() != v2.Type() {
		return false
	}
	if m, ok := v1.Type().MethodByName("SemanticDeepEqual"); ok {
		if v2.Kind() == reflect.Ptr && m.Type.In(1).Kind() != reflect.Ptr {
			if v2.IsNil() {
				return v1.IsNil()
			}
			return m.Func.Call([]reflect.Value{v1, v2.Elem()})[0].Bool()
		} else {
			return m.Func.Call([]reflect.Value{v1, v2})[0].Bool()
		}
	}

	hard := func(k reflect.Kind) bool {
		switch k {
		case reflect.Array, reflect.Map, reflect.Slice, reflect.Struct:
			return true
		}
		return false
	}

	if v1.CanAddr() && v2.CanAddr() && hard(v1.Kind()) {
		addr1 := v1.UnsafeAddr()
		addr2 := v2.UnsafeAddr()
		if addr1 > addr2 {
			// Canonicalize order to reduce number of entries in visited.
			addr1, addr2 = addr2, addr1
		}

		// Short circuit if references are identical ...
		if addr1 == addr2 {
			return true
		}

		// ... or already seen
		typ := v1.Type()
		v := visit{addr1, addr2, typ}
		if visited[v] {
			return true
		}

		// Remember for later.
		visited[v] = true
	}

	switch v1.Kind() {
	case reflect.Array:
		// We don't need to check length here because length is part of
		// an array's type, which has already been filtered for.
		for i := 0; i < v1.Len(); i++ {
			if !deepValueEqual(v1.Index(i), v2.Index(i), visited, depth+1) {
				return false
			}
		}
		return true
	case reflect.Slice:
		if (v1.IsNil() || v1.Len() == 0) != (v2.IsNil() || v2.Len() == 0) {
			return false
		}
		if v1.IsNil() || v1.Len() == 0 {
			return true
		}
		if v1.Len() != v2.Len() {
			return false
		}
		if v1.Pointer() == v2.Pointer() {
			return true
		}
		for i := 0; i < v1.Len(); i++ {
			if !deepValueEqual(v1.Index(i), v2.Index(i), visited, depth+1) {
				return false
			}
		}
		return true
	case reflect.Interface:
		if v1.IsNil() || v2.IsNil() {
			return v1.IsNil() == v2.IsNil()
		}
		return deepValueEqual(v1.Elem(), v2.Elem(), visited, depth+1)
	case reflect.Ptr:
		return deepValueEqual(v1.Elem(), v2.Elem(), visited, depth+1)
	case reflect.Struct:
		for i, n := 0, v1.NumField(); i < n; i++ {
			if !deepValueEqual(v1.Field(i), v2.Field(i), visited, depth+1) {
				return false
			}
		}
		return true
	case reflect.Map:
		if (v1.IsNil() || v1.Len() == 0) != (v2.IsNil() || v2.Len() == 0) {
			return false
		}
		if v1.IsNil() || v1.Len() == 0 {
			return true
		}
		if v1.Len() != v2.Len() {
			return false
		}
		if v1.Pointer() == v2.Pointer() {
			return true
		}
		for _, k := range v1.MapKeys() {
			if !deepValueEqual(v1.MapIndex(k), v2.MapIndex(k), visited, depth+1) {
				return false
			}
		}
		return true
	case reflect.Func:
		if v1.IsNil() && v2.IsNil() {
			return true
		}
		// Can't do better than this:
		return false
	default:
		// Normal equality suffices
		if !v1.CanInterface() || !v2.CanInterface() {
			panic(unexportedTypePanic{})
		}
		return v1.Interface() == v2.Interface()
	}
}

// DeepEqual is like reflect.DeepEqual, but focused on semantic equality
// instead of memory equality.
//
// It will use e's equality functions if it finds types that match.
//
// An empty slice *is* equal to a nil slice for our purposes; same for maps.
//
// Unexported field members cannot be compared and will cause an imformative panic; you must add an Equality
// function for these types.
func DeepEqual(a1, a2 interface{}) bool {
	if a1 == nil || a2 == nil {
		return a1 == a2
	}
	v1 := reflect.ValueOf(a1)
	v2 := reflect.ValueOf(a2)
	if v1.Type() != v2.Type() {
		return false
	}
	return deepValueEqual(v1, v2, make(map[visit]bool), 0)
}

func deepValueDerive(v1, v2 reflect.Value, visited map[visit]bool, depth int) bool {
	defer makeUsefulPanic(v1)

	if !v1.IsValid() || !v2.IsValid() {
		return v1.IsValid() == v2.IsValid()
	}
	if v1.Type() != v2.Type() {
		return false
	}
	if m, ok := v1.Type().MethodByName("SemanticDeepEqual"); ok {
		if v2.Kind() == reflect.Ptr && m.Type.In(1).Kind() != reflect.Ptr {
			if v2.IsNil() {
				return v1.IsNil()
			}
			return m.Func.Call([]reflect.Value{v1, v2.Elem()})[0].Bool()
		} else {
			return m.Func.Call([]reflect.Value{v1, v2})[0].Bool()
		}
	}

	hard := func(k reflect.Kind) bool {
		switch k {
		case reflect.Array, reflect.Map, reflect.Slice, reflect.Struct:
			return true
		}
		return false
	}

	if v1.CanAddr() && v2.CanAddr() && hard(v1.Kind()) {
		addr1 := v1.UnsafeAddr()
		addr2 := v2.UnsafeAddr()
		if addr1 > addr2 {
			// Canonicalize order to reduce number of entries in visited.
			addr1, addr2 = addr2, addr1
		}

		// Short circuit if references are identical ...
		if addr1 == addr2 {
			return true
		}

		// ... or already seen
		typ := v1.Type()
		v := visit{addr1, addr2, typ}
		if visited[v] {
			return true
		}

		// Remember for later.
		visited[v] = true
	}

	switch v1.Kind() {
	case reflect.Array:
		// We don't need to check length here because length is part of
		// an array's type, which has already been filtered for.
		for i := 0; i < v1.Len(); i++ {
			if !deepValueDerive(v1.Index(i), v2.Index(i), visited, depth+1) {
				return false
			}
		}
		return true
	case reflect.Slice:
		if v1.IsNil() || v1.Len() == 0 {
			return true
		}
		if v1.Len() > v2.Len() {
			return false
		}
		if v1.Pointer() == v2.Pointer() {
			return true
		}
		for i := 0; i < v1.Len(); i++ {
			if !deepValueDerive(v1.Index(i), v2.Index(i), visited, depth+1) {
				return false
			}
		}
		return true
	case reflect.String:
		if v1.Len() == 0 {
			return true
		}
		if v1.Len() > v2.Len() {
			return false
		}
		return v1.String() == v2.String()
	case reflect.Interface:
		if v1.IsNil() {
			return true
		}
		return deepValueDerive(v1.Elem(), v2.Elem(), visited, depth+1)
	case reflect.Ptr:
		if v1.IsNil() {
			return true
		}
		return deepValueDerive(v1.Elem(), v2.Elem(), visited, depth+1)
	case reflect.Struct:
		for i, n := 0, v1.NumField(); i < n; i++ {
			if !deepValueDerive(v1.Field(i), v2.Field(i), visited, depth+1) {
				return false
			}
		}
		return true
	case reflect.Map:
		if v1.IsNil() || v1.Len() == 0 {
			return true
		}
		if v1.Len() > v2.Len() {
			return false
		}
		if v1.Pointer() == v2.Pointer() {
			return true
		}
		for _, k := range v1.MapKeys() {
			if !deepValueDerive(v1.MapIndex(k), v2.MapIndex(k), visited, depth+1) {
				return false
			}
		}
		return true
	case reflect.Func:
		if v1.IsNil() && v2.IsNil() {
			return true
		}
		// Can't do better than this:
		return false
	default:
		// Normal equality suffices
		if !v1.CanInterface() || !v2.CanInterface() {
			panic(unexportedTypePanic{})
		}
		return v1.Interface() == v2.Interface()
	}
}

// DeepDerivative is similar to DeepEqual except that unset fields in a1 are
// ignored (not compared). This allows us to focus on the fields that matter to
// the semantic comparison.
//
// The unset fields include a nil pointer and an empty string.
func DeepDerivative(a1, a2 interface{}) bool {
	if a1 == nil {
		return true
	}
	v1 := reflect.ValueOf(a1)
	v2 := reflect.ValueOf(a2)
	if v1.Type() != v2.Type() {
		return false
	}
	return deepValueDerive(v1, v2, make(map[visit]bool), 0)
}
