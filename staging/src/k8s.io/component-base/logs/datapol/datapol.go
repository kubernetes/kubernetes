/*
Copyright 2020 The Kubernetes Authors.

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

// Package datapol contains functions to determine if objects contain sensitive
// data to e.g. make decisions on whether to log them or not, and to redact
// datapolicy-tagged fields in place before such objects are logged or served
// (see Redact).
package datapol

import (
	"fmt"
	"reflect"
	"strings"

	"k8s.io/klog/v2"
)

// redacted is the placeholder value written over every field tagged with
// `datapolicy`. It matches the flagz/zpages "CLASSIFIED" convention.
const redacted = "CLASSIFIED"

// Redact walks obj via reflection and replaces the value of every field tagged
// with `datapolicy` (regardless of the tag's value) with the string
// "CLASSIFIED". Redaction happens in place, so callers MUST pass a pointer to a
// deep copy of any object they do not want mutated. Fields without a datapolicy
// tag are left untouched, but nested structs, slices, and maps are traversed so
// that tagged fields nested arbitrarily deep are still redacted.
func Redact(obj interface{}) (retErr error) {
	defer func() {
		if r := recover(); r != nil {
			retErr = fmt.Errorf("panic while redacting sensitive data: %v", r)
		}
	}()
	v := reflect.ValueOf(obj)
	if v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface {
		if v.IsNil() {
			return nil
		}
		v = v.Elem()
	}
	redactWalk(v, map[visitKey]struct{}{})
	return nil
}

// visitKey identifies a value already visited during a walk. ptr is the value's
// address via reflect.Value.Pointer(); length disambiguates slices that share a
// backing array but span different ranges (e.g. s and s[:1]) so that aliased
// sub-slices are still fully walked while a genuine self-referential slice is
// still caught. For pointers and maps length is always 0.
type visitKey struct {
	ptr    uintptr
	length int
}

// seen records the identity of pointers, maps, and slices already visited during
// a walk so that a cyclic object graph short-circuits instead of recursing
// forever. Without this guard a self-referential input (including a slice that
// contains itself through an interface) overflows the stack, which is a fatal
// error that Redact's recover() cannot catch. Returns true if v was already
// visited (and records it otherwise). Only pointer, map, and slice kinds carry a
// meaningful identity via Pointer(); other kinds return false.
func seen(v reflect.Value, visited map[visitKey]struct{}) bool {
	var key visitKey
	switch v.Kind() {
	case reflect.Pointer, reflect.Map:
		if v.IsNil() {
			return false
		}
		key = visitKey{ptr: v.Pointer()}
	case reflect.Slice:
		if v.IsNil() {
			return false
		}
		// A slice's identity is its backing-array address plus its length: a
		// self-referential slice repeats both, while a distinct sub-slice of a
		// shared backing array differs in length and must still be walked.
		key = visitKey{ptr: v.Pointer(), length: v.Len()}
	default:
		return false
	}
	if _, ok := visited[key]; ok {
		return true
	}
	visited[key] = struct{}{}
	return false
}

// redactWalk traverses untagged values looking for struct fields carrying a
// datapolicy tag. When it finds one it hands the field to redactValue.
func redactWalk(v reflect.Value, visited map[visitKey]struct{}) {
	if seen(v, visited) {
		return
	}
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return
		}
		redactWalk(v.Elem(), visited)
	case reflect.Interface:
		if v.IsNil() {
			return
		}
		// An interface's Elem() is not addressable, so walking it directly would
		// leave any nested tagged fields unset (CanSet is false). Walk a settable
		// copy and write it back so tagged fields reached through a by-value
		// interface are still redacted.
		ev := v.Elem()
		cp := reflect.New(ev.Type()).Elem()
		cp.Set(ev)
		redactWalk(cp, visited)
		if v.CanSet() {
			v.Set(cp)
		}
	case reflect.Struct:
		t := v.Type()
		for i := 0; i < t.NumField(); i++ {
			fv := v.Field(i)
			// Unexported fields cannot be set via reflection; skip them.
			if !fv.CanSet() {
				continue
			}
			if _, ok := t.Field(i).Tag.Lookup("datapolicy"); ok {
				// Use a fresh visited set for the value redaction rather than
				// sharing redactWalk's. The walk records the addresses of maps
				// and pointers it merely traverses (without redacting); if a
				// tagged field aliases one of those already-walked addresses,
				// sharing the set would make seen() short-circuit and skip
				// redaction entirely — a fail-open leak whose occurrence depends
				// on struct field order. redactValue never re-enters redactWalk
				// and zeroes structs instead of recursing into them, so its own
				// cycle detection is self-contained within the fresh set.
				redactValue(fv, map[visitKey]struct{}{})
				continue
			}
			redactWalk(fv, visited)
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			redactWalk(v.Index(i), visited)
		}
	case reflect.Map:
		iter := v.MapRange()
		for iter.Next() {
			// Map values are not addressable, so operate on a settable copy and
			// write it back.
			mv := iter.Value()
			cp := reflect.New(mv.Type()).Elem()
			cp.Set(mv)
			redactWalk(cp, visited)
			v.SetMapIndex(iter.Key(), cp)
		}
	}
}

// redactValue overwrites the value of a datapolicy-tagged field. Strings,
// byte slices, and string slices are replaced with the "CLASSIFIED" sentinel;
// maps preserve their keys but have their leaf values redacted; pointers and
// interfaces are dereferenced; any other scalar is zeroed.
func redactValue(v reflect.Value, visited map[visitKey]struct{}) {
	switch v.Kind() {
	case reflect.String:
		v.SetString(redacted)
	case reflect.Pointer:
		// A nil pointer holds no value to redact; leave it nil rather than
		// materializing an empty object that was not present in the input.
		if v.IsNil() {
			return
		}
		// Cycle guard: a pointer that (transitively) points back to itself must
		// not recurse forever. Redaction of the pointee mutates shared storage
		// in place, so skipping an already-visited pointer loses nothing.
		if seen(v, visited) {
			return
		}
		redactValue(v.Elem(), visited)
	case reflect.Interface:
		if v.IsNil() {
			return
		}
		ev := v.Elem()
		cp := reflect.New(ev.Type()).Elem()
		cp.Set(ev)
		redactValue(cp, visited)
		v.Set(cp)
	case reflect.Slice:
		switch v.Type().Elem().Kind() {
		case reflect.Uint8:
			// []byte — terminal replacement. This overwrites the slice header
			// rather than mutating the shared backing array, so it must NOT be
			// short-circuited by the seen() cycle guard: two tagged fields (or
			// two map values) that alias the same slice each need their own
			// replacement, otherwise the second alias leaks its original value.
			v.SetBytes([]byte(redacted))
		case reflect.String:
			// []string (or a named type with string elements) — terminal
			// replacement; same rationale as []byte above, so no seen() guard.
			s := reflect.MakeSlice(v.Type(), 1, 1)
			s.Index(0).SetString(redacted)
			v.Set(s)
		default:
			// Recursive slice: redaction mutates each element in place through
			// the shared backing array, so a cycle guard is both safe (aliases
			// already share the mutation) and necessary (a self-referential
			// slice would otherwise recurse forever).
			if seen(v, visited) {
				return
			}
			for i := 0; i < v.Len(); i++ {
				redactValue(v.Index(i), visited)
			}
		}
	case reflect.Array:
		for i := 0; i < v.Len(); i++ {
			redactValue(v.Index(i), visited)
		}
	case reflect.Map:
		// Cycle guard: a map reachable from its own values must not recurse
		// forever. Values are rewritten via SetMapIndex on the shared map, so
		// skipping an already-visited map loses nothing.
		if seen(v, visited) {
			return
		}
		iter := v.MapRange()
		for iter.Next() {
			mv := iter.Value()
			cp := reflect.New(mv.Type()).Elem()
			cp.Set(mv)
			redactValue(cp, visited)
			v.SetMapIndex(iter.Key(), cp)
		}
	default:
		// Numbers, bools, and other scalars: clear the value.
		v.Set(reflect.Zero(v.Type()))
	}
}

// Verify returns a list of the datatypes contained in the argument that can be
// considered sensitive w.r.t. to logging
func Verify(logger klog.Logger, value interface{}) []string {
	defer func() {
		if r := recover(); r != nil {
			logger.Error(nil, "Error while inspecting arguments for sensitive data", "panic", r)
		}
	}()
	t := reflect.ValueOf(value)
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return datatypes(t)
}

func datatypes(v reflect.Value) []string {
	if types := byType(v.Type()); len(types) > 0 {
		// Slices, and maps can be nil or empty, only the nil case is zero
		switch v.Kind() {
		case reflect.Slice, reflect.Map:
			if !v.IsZero() && v.Len() > 0 {
				return types
			}
		default:
			if !v.IsZero() {
				return types
			}
		}
	}
	switch v.Kind() {
	case reflect.Interface:
		return datatypes(v.Elem())
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			if types := datatypes(v.Index(i)); len(types) > 0 {
				return types
			}
		}
	case reflect.Map:
		mapIter := v.MapRange()
		for mapIter.Next() {
			k := mapIter.Key()
			v := mapIter.Value()
			if types := datatypes(k); len(types) > 0 {
				return types
			}
			if types := datatypes(v); len(types) > 0 {
				return types
			}
		}
	case reflect.Struct:
		t := v.Type()
		numField := t.NumField()

		for i := 0; i < numField; i++ {
			f := t.Field(i)
			if f.Type.Kind() == reflect.Pointer {
				continue
			}
			if reason, ok := f.Tag.Lookup("datapolicy"); ok {
				if !v.Field(i).IsZero() {
					return strings.Split(reason, ",")
				}
			}
			if types := datatypes(v.Field(i)); len(types) > 0 {
				return types
			}
		}
	}
	return nil
}
