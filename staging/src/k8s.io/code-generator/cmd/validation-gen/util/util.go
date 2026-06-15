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
	"fmt"
	"math"
	"strconv"

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

// ParseInt strictly parses an int from a string input,
// ensuring that when converted back to a string, the resulting
// int and the input string have the exact same representation.
// This prevents scenarios where an input like "0100" parses
// as 100 and would be re-stringed as "100".
func ParseInt(val string) (int, error) {
	intVal, err := strconv.Atoi(val)
	if err != nil {
		return 0, fmt.Errorf("parsing %q as int: %w", val, err)
	}

	strVal := strconv.Itoa(intVal)
	if strVal != val {
		return 0, fmt.Errorf("%q is not a valid int value", val)
	}

	return intVal, nil
}

// ParseSignedInt strictly parses a signed integer from a string input and
// validates that the result fits within the specified bit size. The bitSize
// parameter should be 8, 16, 32, or 64, corresponding to the target Go type.
// Values outside the representable range for the target type are rejected at
// parse time with a descriptive error.
func ParseSignedInt(val string, bitSize int) (int64, error) {
	intVal, err := strconv.ParseInt(val, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing %q as int: %w", val, err)
	}

	// Verify canonical form: reject leading zeros, unary plus, etc.
	strVal := strconv.FormatInt(intVal, 10)
	if strVal != val {
		return 0, fmt.Errorf("%q is not a valid int value", val)
	}

	// Validate the parsed value fits in the target type's range.
	var minVal, maxVal int64
	switch bitSize {
	case 8:
		minVal, maxVal = math.MinInt8, math.MaxInt8
	case 16:
		minVal, maxVal = math.MinInt16, math.MaxInt16
	case 32:
		minVal, maxVal = math.MinInt32, math.MaxInt32
	case 64:
		minVal, maxVal = math.MinInt64, math.MaxInt64
	default:
		return 0, fmt.Errorf("unsupported bitSize %d; must be 8, 16, 32, or 64", bitSize)
	}
	if intVal < minVal || intVal > maxVal {
		return 0, fmt.Errorf("value %d does not fit in int%d (range [%d, %d])", intVal, bitSize, minVal, maxVal)
	}

	return intVal, nil
}

// ParseUnsignedInt strictly parses an unsigned integer from a string input and
// validates that the result fits within the specified bit size. The bitSize
// parameter should be 8, 16, 32, or 64, corresponding to the target Go type.
func ParseUnsignedInt(val string, bitSize int) (uint64, error) {
	uintVal, err := strconv.ParseUint(val, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing %q as uint: %w", val, err)
	}

	// Verify canonical form: reject leading zeros, unary plus, etc.
	strVal := strconv.FormatUint(uintVal, 10)
	if strVal != val {
		return 0, fmt.Errorf("%q is not a valid uint value", val)
	}

	// Validate the parsed value fits in the target type's range.
	var maxVal uint64
	switch bitSize {
	case 8:
		maxVal = math.MaxUint8
	case 16:
		maxVal = math.MaxUint16
	case 32:
		maxVal = math.MaxUint32
	case 64:
		maxVal = math.MaxUint64
	default:
		return 0, fmt.Errorf("unsupported bitSize %d; must be 8, 16, 32, or 64", bitSize)
	}
	if uintVal > maxVal {
		return 0, fmt.Errorf("value %d does not fit in uint%d (range [0, %d])", uintVal, bitSize, maxVal)
	}

	return uintVal, nil
}

// ParseBool strictly parses a bool from a string input,
// ensuring that when converted back to a string, the resulting
// bool and the input string have the exact same representation.
// This prevents scenarios where an input like "TRUE" parses
// as true and would be re-stringed as "true".
func ParseBool(val string) (bool, error) {
	switch val {
	case "true":
		return true, nil
	case "false":
		return false, nil
	}
	return false, fmt.Errorf("%q is not a valid bool value", val)
}
