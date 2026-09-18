/*
Copyright 2024 The Kubernetes Authors.

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

package minimum

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestBasicStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Test that zero values are rejected because they are below the minimum of 1.
	st.Value(&BasicStruct{
		// all zero values
		IntPtrField:     new(0),
		UintPtrField:    new(uint(0)),
		TypedefPtrField: new(IntType(0)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("intPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("int16Field"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("int32Field"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("int64Field"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("uintField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("uintPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("uint16Field"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("uint32Field"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("uint64Field"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("typedefField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("typedefPtrField"), nil, "").WithOrigin("minimum"),
	})

	// Test validation ratcheting: unchanged invalid data is allowed.
	st.Value(&BasicStruct{
		IntPtrField:     new(0),
		UintPtrField:    new(uint(0)),
		TypedefPtrField: new(IntType(0)),
	}).OldValue(&BasicStruct{
		IntPtrField:     new(0),
		UintPtrField:    new(uint(0)),
		TypedefPtrField: new(IntType(0)),
	}).ExpectValid()

	// Changed invalid data is still rejected.
	st.Value(&BasicStruct{
		IntField:        -1,
		IntPtrField:     new(-1),
		Int16Field:      -1,
		Int32Field:      -1,
		Int64Field:      -1,
		TypedefField:    IntType(-1),
		TypedefPtrField: new(IntType(-1)),
	}).OldValue(&BasicStruct{
		IntField:        0,
		IntPtrField:     new(0),
		Int16Field:      0,
		Int32Field:      0,
		Int64Field:      0,
		TypedefField:    IntType(0),
		TypedefPtrField: new(IntType(0)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("intPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("int16Field"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("int32Field"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("int64Field"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("typedefField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("typedefPtrField"), nil, "").WithOrigin("minimum"),
	})

	// Test that values meeting the minimum of 1 are valid.
	st.Value(&BasicStruct{
		IntField:        1,
		IntPtrField:     new(1),
		Int16Field:      1,
		Int32Field:      1,
		Int64Field:      1,
		UintField:       1,
		Uint16Field:     1,
		Uint32Field:     1,
		Uint64Field:     1,
		UintPtrField:    new(uint(1)),
		TypedefField:    IntType(1),
		TypedefPtrField: new(IntType(1)),
	}).ExpectValid()
}

func TestOptionalStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Test that explicitly provided zero values for optional pointers are still validated against minimum.
	st.Value(&OptionalStruct{
		// zero values
		OptionalIntPtrField:     new(0),
		OptionalTypedefPtrField: new(IntType(0)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("optionalIntPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalTypedefPtrField"), nil, "").WithOrigin("minimum"),
	})

	// Test that omitting an optional pointer field (nil) short-circuits minimum validation.
	st.Value(&OptionalStruct{
		// OptionalIntField is zero and OptionalIntPtrField is nil, so optional
		// short-circuits before minimum runs.
	}).ExpectValid()

	// Test validation ratcheting: unchanged invalid data is allowed.
	st.Value(&OptionalStruct{
		OptionalIntField:        -1,
		OptionalIntPtrField:     new(-1),
		OptionalTypedefField:    IntType(-1),
		OptionalTypedefPtrField: new(IntType(-1)),
	}).OldValue(&OptionalStruct{
		OptionalIntField:        -1,
		OptionalIntPtrField:     new(-1),
		OptionalTypedefField:    IntType(-1),
		OptionalTypedefPtrField: new(IntType(-1)),
	}).ExpectValid()

	// Changed invalid data is still rejected.
	st.Value(&OptionalStruct{
		OptionalIntField:        -2,
		OptionalIntPtrField:     new(-2),
		OptionalTypedefField:    IntType(-2),
		OptionalTypedefPtrField: new(IntType(-2)),
	}).OldValue(&OptionalStruct{
		OptionalIntField:        -1,
		OptionalIntPtrField:     new(-1),
		OptionalTypedefField:    IntType(-1),
		OptionalTypedefPtrField: new(IntType(-1)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("optionalIntField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalIntPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalTypedefField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalTypedefPtrField"), nil, "").WithOrigin("minimum"),
	})

	// Test that values meeting the minimum of 1 are valid.
	st.Value(&OptionalStruct{
		OptionalIntField:        1,
		OptionalIntPtrField:     new(1),
		OptionalTypedefField:    IntType(1),
		OptionalTypedefPtrField: new(IntType(1)),
	}).ExpectValid()

	// Test that invalid values below the minimum are rejected.
	st.Value(&OptionalStruct{
		OptionalIntField:        -1,
		OptionalIntPtrField:     new(-1),
		OptionalTypedefField:    IntType(-1),
		OptionalTypedefPtrField: new(IntType(-1)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("optionalIntField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalIntPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalTypedefField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalTypedefPtrField"), nil, "").WithOrigin("minimum"),
	})
}

func TestRequiredStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Test that explicitly provided zero values for pointers are validated against minimum,
	// while zero values for non-pointers trigger required validation.
	st.Value(&RequiredStruct{
		// zero values
		RequiredIntField:        0,
		RequiredIntPtrField:     new(0),
		RequiredTypedefField:    IntType(0),
		RequiredTypedefPtrField: new(IntType(0)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Required(field.NewPath("requiredIntField"), ""),
		field.Invalid(field.NewPath("requiredIntPtrField"), nil, "").WithOrigin("minimum"),
		field.Required(field.NewPath("requiredTypedefField"), ""),
		field.Invalid(field.NewPath("requiredTypedefPtrField"), nil, "").WithOrigin("minimum"),
	})

	// Test that omitted required pointer fields (nil) emit field.Required and short-circuit minimum validation.
	st.Value(&RequiredStruct{
		// RequiredIntField is zero and RequiredIntPtrField is nil.
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Required(field.NewPath("requiredIntField"), ""),
		field.Required(field.NewPath("requiredIntPtrField"), ""),
		field.Required(field.NewPath("requiredTypedefField"), ""),
		field.Required(field.NewPath("requiredTypedefPtrField"), ""),
	})

	// Test validation ratcheting: unchanged invalid data is allowed.
	st.Value(&RequiredStruct{
		RequiredIntField:        0,
		RequiredIntPtrField:     new(0),
		RequiredTypedefField:    IntType(0),
		RequiredTypedefPtrField: new(IntType(0)),
	}).OldValue(&RequiredStruct{
		RequiredIntField:        0,
		RequiredIntPtrField:     new(0),
		RequiredTypedefField:    IntType(0),
		RequiredTypedefPtrField: new(IntType(0)),
	}).ExpectValid()

	// Changed invalid data is still rejected.
	st.Value(&RequiredStruct{
		RequiredIntField:        -1,
		RequiredIntPtrField:     new(-1),
		RequiredTypedefField:    IntType(-1),
		RequiredTypedefPtrField: new(IntType(-1)),
	}).OldValue(&RequiredStruct{
		RequiredIntField:        0,
		RequiredIntPtrField:     new(0),
		RequiredTypedefField:    IntType(0),
		RequiredTypedefPtrField: new(IntType(0)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("requiredIntField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("requiredIntPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("requiredTypedefField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("requiredTypedefPtrField"), nil, "").WithOrigin("minimum"),
	})

	// Test that values meeting the minimum of 1 are valid.
	st.Value(&RequiredStruct{
		RequiredIntField:        1,
		RequiredIntPtrField:     new(1),
		RequiredTypedefField:    IntType(1),
		RequiredTypedefPtrField: new(IntType(1)),
	}).ExpectValid()

	// Test that invalid values below the minimum are rejected.
	st.Value(&RequiredStruct{
		RequiredIntField:        -1,
		RequiredIntPtrField:     new(-1),
		RequiredTypedefField:    IntType(-1),
		RequiredTypedefPtrField: new(IntType(-1)),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("requiredIntField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("requiredIntPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("requiredTypedefField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("requiredTypedefPtrField"), nil, "").WithOrigin("minimum"),
	})
}

func TestNegativeMinimumStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Test that zero values for required pointers are validated against the negative minimum,
	// while zero values for non-pointers trigger required validation.
	st.Value(&NegativeMinimumStruct{
		// zero values (valid for -10)
		NegativeMinimumPtrField:         new(0),
		OptionalNegativeMinimumPtrField: new(0),
		RequiredNegativeMinimumField:    0,
		RequiredNegativeMinimumPtrField: new(0),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Required(field.NewPath("requiredNegativeMinimumField"), ""),
	})

	// Test that valid values at the negative minimum boundary (-10) are accepted.
	st.Value(&NegativeMinimumStruct{
		NegativeMinimumField:            -10,
		NegativeMinimumPtrField:         new(-10),
		OptionalNegativeMinimumField:    -10,
		OptionalNegativeMinimumPtrField: new(-10),
		RequiredNegativeMinimumField:    -10,
		RequiredNegativeMinimumPtrField: new(-10),
	}).ExpectValid()

	// Test validation ratcheting: unchanged invalid data is allowed.
	st.Value(&NegativeMinimumStruct{
		NegativeMinimumField:            -11,
		NegativeMinimumPtrField:         new(-11),
		OptionalNegativeMinimumField:    -11,
		OptionalNegativeMinimumPtrField: new(-11),
		RequiredNegativeMinimumField:    -11,
		RequiredNegativeMinimumPtrField: new(-11),
	}).OldValue(&NegativeMinimumStruct{
		NegativeMinimumField:            -11,
		NegativeMinimumPtrField:         new(-11),
		OptionalNegativeMinimumField:    -11,
		OptionalNegativeMinimumPtrField: new(-11),
		RequiredNegativeMinimumField:    -11,
		RequiredNegativeMinimumPtrField: new(-11),
	}).ExpectValid()

	// Changed invalid data is still rejected.
	st.Value(&NegativeMinimumStruct{
		NegativeMinimumField:            -12,
		NegativeMinimumPtrField:         new(-12),
		OptionalNegativeMinimumField:    -12,
		OptionalNegativeMinimumPtrField: new(-12),
		RequiredNegativeMinimumField:    -12,
		RequiredNegativeMinimumPtrField: new(-12),
	}).OldValue(&NegativeMinimumStruct{
		NegativeMinimumField:            -11,
		NegativeMinimumPtrField:         new(-11),
		OptionalNegativeMinimumField:    -11,
		OptionalNegativeMinimumPtrField: new(-11),
		RequiredNegativeMinimumField:    -11,
		RequiredNegativeMinimumPtrField: new(-11),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("negativeMinimumField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("negativeMinimumPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalNegativeMinimumField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalNegativeMinimumPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("requiredNegativeMinimumField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("requiredNegativeMinimumPtrField"), nil, "").WithOrigin("minimum"),
	})

	// Test that invalid values below the negative minimum (-10) are rejected.
	st.Value(&NegativeMinimumStruct{
		NegativeMinimumField:            -11,
		NegativeMinimumPtrField:         new(-11),
		OptionalNegativeMinimumField:    -11,
		OptionalNegativeMinimumPtrField: new(-11),
		RequiredNegativeMinimumField:    -11,
		RequiredNegativeMinimumPtrField: new(-11),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("negativeMinimumField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("negativeMinimumPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalNegativeMinimumField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("optionalNegativeMinimumPtrField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("requiredNegativeMinimumField"), nil, "").WithOrigin("minimum"),
		field.Invalid(field.NewPath("requiredNegativeMinimumPtrField"), nil, "").WithOrigin("minimum"),
	})
}
