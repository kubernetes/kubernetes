/*
Copyright 2025 The Kubernetes Authors.

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

package util

import (
	"k8s.io/gengo/v2/parser/tags"
	"k8s.io/gengo/v2/types"
)

// GetMemberByJSON returns the child member of the type that has the given JSON
// name. It returns nil if no such member exists.
func GetMemberByJSON(t *types.Type, jsonName string) *types.Member {
	for i := range t.Members {
		if jsonTag, ok := tags.LookupJSON(t.Members[i]); ok {
			if jsonTag.Name == jsonName {
				return &t.Members[i]
			}
		}
	}
	return nil
}

// IsNilableType returns true if the argument type can be compared to nil.
func IsNilableType(t *types.Type) bool {
	t = NativeType(t)

	switch t.Kind {
	case types.Pointer, types.Map, types.Slice, types.Interface: // Note: Arrays are not nilable
		return true
	}
	return false
}

// NativeType returns the Go native type of the argument type, with any
// intermediate typedefs removed. Go itself already flattens typedefs, but this
// handles it in the unlikely event that we ever fix that.
//
// Examples:
// * Trivial:
//   - given `int`, returns `int`
//   - given `*int`, returns `*int`
//   - given `[]int`, returns `[]int`
//
// * Typedefs
//   - given `type X int; X`, returns `int`
//   - given `type X int; []X`, returns `[]X`
//
// * Typedefs and pointers:
//   - given `type X int; *X`, returns `*int`
//   - given `type X *int; *X`, returns `**int`
//   - given `type X []int; X`, returns `[]int`
//   - given `type X []int; *X`, returns `*[]int`
func NativeType(t *types.Type) *types.Type {
	ptrs := 0
	conditionMet := false
	for !conditionMet {
		switch t.Kind {
		case types.Alias:
			t = t.Underlying
		case types.Pointer:
			ptrs++
			t = t.Elem
		default:
			conditionMet = true
		}
	}
	for range ptrs {
		t = types.PointerTo(t)
	}
	return t
}

// NonPointer returns the value-type of a possibly pointer type. If type is not
// a pointer, it returns the input type.
func NonPointer(t *types.Type) *types.Type {
	for t.Kind == types.Pointer {
		t = t.Elem
	}
	return t
}

// IsDirectComparable returns true if the type is safe to compare using "==".
// It is similar to gengo.IsComparable, but it doesn't consider Pointers to be
// comparable (we don't want shallow compare).
func IsDirectComparable(t *types.Type) bool {
	switch t.Kind {
	case types.Builtin:
		return true
	case types.Struct:
		for _, f := range t.Members {
			if !IsDirectComparable(f.Type) {
				return false
			}
		}
		return true
	case types.Array:
		return IsDirectComparable(t.Elem)
	case types.Alias:
		return IsDirectComparable(t.Underlying)
	}
	return false
}
