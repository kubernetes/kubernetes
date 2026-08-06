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
// (see Redact). This package is not intended for hot paths: it uses reflection
// to traverse arbitrary object graphs.
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
	if !v.IsValid() {
		return nil // nil interface
	}
	if v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface {
		if v.IsNil() {
			return nil
		}
		v = v.Elem()
	}
	// A non-pointer value cannot be mutated in place; silently return rather
	// than erroring, because the caller cannot fix the problem at runtime.
	if !v.CanSet() {
		return nil
	}
	return redactWalk(v, map[visitKey]struct{}{})
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
// datapolicy tag. When it finds one it hands the field to redactValue. It
// returns an error if it encounters a value of a kind it does not know how to
// traverse, so an unexpected shape fails closed rather than silently passing a
// value that could hide an unredacted datapolicy-tagged field.
func redactWalk(v reflect.Value, visited map[visitKey]struct{}) error {
	if seen(v, visited) {
		return nil
	}
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return nil
		}
		return redactWalk(v.Elem(), visited)
	case reflect.Interface:
		if v.IsNil() {
			return nil
		}
		// An interface's Elem() is not addressable, so walking it directly would
		// leave any nested tagged fields unset (CanSet is false). Walk a settable
		// copy and write it back so tagged fields reached through a by-value
		// interface are still redacted.
		ev := v.Elem()
		cp := reflect.New(ev.Type()).Elem()
		cp.Set(ev)
		if err := redactWalk(cp, visited); err != nil {
			return err
		}
		if v.CanSet() {
			v.Set(cp)
		}
		return nil
	case reflect.Struct:
		t := v.Type()
		for i := 0; i < t.NumField(); i++ {
			fv := v.Field(i)
			sf := t.Field(i)
			if _, ok := sf.Tag.Lookup("datapolicy"); ok {
				// A tagged field we cannot set (unexported) would leak its value
				// unredacted. Fail closed rather than silently pass it through.
				if !fv.CanSet() {
					return fmt.Errorf("cannot redact datapolicy-tagged field %s.%s: field is unexported and cannot be set via reflection", t, sf.Name)
				}
				// Use a fresh visited set for the value redaction rather than
				// sharing redactWalk's. The walk records the addresses of maps
				// and pointers it merely traverses (without redacting); if a
				// tagged field aliases one of those already-walked addresses,
				// sharing the set would make seen() short-circuit and skip
				// redaction entirely — a fail-open leak whose occurrence depends
				// on struct field order. redactValue never re-enters redactWalk
				// and zeroes structs instead of recursing into them, so its own
				// cycle detection is self-contained within the fresh set.
				if err := redactValue(fv, map[visitKey]struct{}{}); err != nil {
					return err
				}
				continue
			}
			if !fv.CanSet() {
				// Unexported field: reflection cannot overwrite anything beneath
				// it, so we cannot recurse with redactWalk (which mutates). But
				// skipping it blindly would silently leak a datapolicy-tagged
				// field nested below. Inspect the subtree read-only and fail
				// closed if it hides a tag we would be unable to redact.
				if path := findDatapolicyTag(fv, sf.Name, map[reflect.Type]struct{}{}); path != "" {
					return fmt.Errorf("cannot redact datapolicy-tagged field reached through unexported field %s.%s: found tag at %s", t, sf.Name, path)
				}
				continue
			}
			if err := redactWalk(fv, visited); err != nil {
				return err
			}
		}
		return nil
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			if err := redactWalk(v.Index(i), visited); err != nil {
				return err
			}
		}
		return nil
	case reflect.Map:
		iter := v.MapRange()
		for iter.Next() {
			// Map values are not addressable, so operate on a settable copy and
			// write it back.
			mv := iter.Value()
			cp := reflect.New(mv.Type()).Elem()
			cp.Set(mv)
			if err := redactWalk(cp, visited); err != nil {
				return err
			}
			v.SetMapIndex(iter.Key(), cp)
		}
		return nil
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64,
		reflect.Complex64, reflect.Complex128,
		reflect.String:
		// Scalar leaf: it carries no datapolicy-tagged fields beneath it, so
		// there is nothing to walk. A scalar reached here is untagged (tagged
		// fields are handed to redactValue by the struct case above).
		return nil
	default:
		// Chan, Func, UnsafePointer, Invalid, and any future kind we do not know
		// how to traverse. Fail closed: error rather than silently passing a
		// value that could hide an unredacted datapolicy-tagged field.
		return fmt.Errorf("cannot redact value of unexpected kind %s", v.Kind())
	}
}

// findDatapolicyTag scans v read-only for a struct field carrying a datapolicy
// tag, descending through pointers, interfaces, structs, slices, arrays, and
// maps. It exists to inspect subtrees that redactWalk cannot mutate — namely
// those reached through an unexported field — so redaction can fail closed when
// such a subtree hides a tagged field that should have been redacted. It only
// reads values (it never calls Set or Interface), so it is safe on the
// read-only reflect.Values that unexported fields yield, where a mutating walk
// would panic. seenTypes bounds recursion on self-referential types: detection
// is structural, so once a type has been fully scanned without a tag, revisiting
// it can add nothing. It returns a dotted field path to the first tag found, or
// "" if the subtree carries none.
func findDatapolicyTag(v reflect.Value, path string, seenTypes map[reflect.Type]struct{}) string {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if v.IsNil() {
			return ""
		}
		return findDatapolicyTag(v.Elem(), path, seenTypes)
	case reflect.Struct:
		t := v.Type()
		if _, ok := seenTypes[t]; ok {
			return ""
		}
		seenTypes[t] = struct{}{}
		for i := 0; i < t.NumField(); i++ {
			sf := t.Field(i)
			fieldPath := path + "." + sf.Name
			if _, ok := sf.Tag.Lookup("datapolicy"); ok {
				return fieldPath
			}
			if p := findDatapolicyTag(v.Field(i), fieldPath, seenTypes); p != "" {
				return p
			}
		}
		return ""
	case reflect.Slice, reflect.Array:
		// Walk every element: a slice/array of interfaces can hold differing
		// dynamic types, so no single element is representative. An empty
		// container holds no value to redact, so finding no tag is correct.
		for i := 0; i < v.Len(); i++ {
			if p := findDatapolicyTag(v.Index(i), path, seenTypes); p != "" {
				return p
			}
		}
		return ""
	case reflect.Map:
		iter := v.MapRange()
		for iter.Next() {
			// Match redactWalk, which only redacts map values, not keys.
			if p := findDatapolicyTag(iter.Value(), path, seenTypes); p != "" {
				return p
			}
		}
		return ""
	default:
		// Scalars and other leaf kinds carry no nested tags.
		return ""
	}
}

// redactValue overwrites the value of a datapolicy-tagged field. Strings,
// byte slices, and string slices are replaced with the "CLASSIFIED" sentinel;
// maps preserve their keys but have their leaf values redacted; pointers and
// interfaces are dereferenced; any other scalar is zeroed.
//
// It returns an error if it cannot actually overwrite a value it was asked to
// redact — for example a leaf that reflection reports as not settable. Every
// terminal write is guarded so that an un-redactable value fails closed with a
// descriptive error instead of silently leaking (or only surfacing as a
// recovered panic).
func redactValue(v reflect.Value, visited map[visitKey]struct{}) error {
	switch v.Kind() {
	case reflect.String:
		if !v.CanSet() {
			return notSettableErr(v)
		}
		v.SetString(redacted)
		return nil
	case reflect.Pointer:
		// A nil pointer holds no value to redact; leave it nil rather than
		// materializing an empty object that was not present in the input.
		if v.IsNil() {
			return nil
		}
		// Cycle guard: a pointer that (transitively) points back to itself must
		// not recurse forever. Redaction of the pointee mutates shared storage
		// in place, so skipping an already-visited pointer loses nothing.
		if seen(v, visited) {
			return nil
		}
		return redactValue(v.Elem(), visited)
	case reflect.Interface:
		if v.IsNil() {
			return nil
		}
		ev := v.Elem()
		cp := reflect.New(ev.Type()).Elem()
		cp.Set(ev)
		if err := redactValue(cp, visited); err != nil {
			return err
		}
		if !v.CanSet() {
			return notSettableErr(v)
		}
		v.Set(cp)
		return nil
	case reflect.Slice:
		switch v.Type().Elem().Kind() {
		case reflect.Uint8:
			// []byte — terminal replacement. This overwrites the slice header
			// rather than mutating the shared backing array, so it must NOT be
			// short-circuited by the seen() cycle guard: two tagged fields (or
			// two map values) that alias the same slice each need their own
			// replacement, otherwise the second alias leaks its original value.
			if !v.CanSet() {
				return notSettableErr(v)
			}
			v.SetBytes([]byte(redacted))
			return nil
		case reflect.String:
			// []string (or a named type with string elements) — terminal
			// replacement; same rationale as []byte above, so no seen() guard.
			if !v.CanSet() {
				return notSettableErr(v)
			}
			s := reflect.MakeSlice(v.Type(), 1, 1)
			s.Index(0).SetString(redacted)
			v.Set(s)
			return nil
		default:
			// Recursive slice: redaction mutates each element in place through
			// the shared backing array, so a cycle guard is both safe (aliases
			// already share the mutation) and necessary (a self-referential
			// slice would otherwise recurse forever).
			if seen(v, visited) {
				return nil
			}
			for i := 0; i < v.Len(); i++ {
				if err := redactValue(v.Index(i), visited); err != nil {
					return err
				}
			}
			return nil
		}
	case reflect.Array:
		for i := 0; i < v.Len(); i++ {
			if err := redactValue(v.Index(i), visited); err != nil {
				return err
			}
		}
		return nil
	case reflect.Map:
		// Cycle guard: a map reachable from its own values must not recurse
		// forever. Values are rewritten via SetMapIndex on the shared map, so
		// skipping an already-visited map loses nothing.
		if seen(v, visited) {
			return nil
		}
		iter := v.MapRange()
		for iter.Next() {
			mv := iter.Value()
			cp := reflect.New(mv.Type()).Elem()
			cp.Set(mv)
			if err := redactValue(cp, visited); err != nil {
				return err
			}
			v.SetMapIndex(iter.Key(), cp)
		}
		return nil
	default:
		// Numbers, bools, and other scalars: clear the value.
		if !v.CanSet() {
			return notSettableErr(v)
		}
		v.Set(reflect.Zero(v.Type()))
		return nil
	}
}

// notSettableErr reports that reflection cannot overwrite v, so the
// datapolicy-tagged value it holds cannot be redacted and redaction must fail
// closed rather than leak the value.
func notSettableErr(v reflect.Value) error {
	return fmt.Errorf("cannot redact datapolicy-tagged value of kind %s: reflection reports it is not settable", v.Kind())
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
