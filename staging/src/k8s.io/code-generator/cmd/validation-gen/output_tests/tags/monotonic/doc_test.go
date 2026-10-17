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

package monotonic

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	structOrig := Struct{
		IntField:          10,
		Int64Field:        20,
		Uint64Field:       30,
		IntPtrField:       new(40),
		MonotonicField:    50,
		MonotonicPtrField: new(MonotonicType(60)),
		OptionalInt:       10,
		RequiredInt:       10,
		OptionalIntPtr:    new(10),
		RequiredIntPtr:    new(10),
		NegativeInt:       0,
	}

	structIncrease := Struct{
		IntField:          11,
		Int64Field:        21,
		Uint64Field:       31,
		IntPtrField:       new(41),
		MonotonicField:    51,
		MonotonicPtrField: new(MonotonicType(61)),
		OptionalInt:       11,
		RequiredInt:       11,
		OptionalIntPtr:    new(11),
		RequiredIntPtr:    new(11),
		NegativeInt:       5,
	}

	structDecrease := Struct{
		IntField:          9,
		Int64Field:        19,
		Uint64Field:       29,
		IntPtrField:       new(39),
		MonotonicField:    49,
		MonotonicPtrField: new(MonotonicType(59)),
		OptionalInt:       9,
		RequiredInt:       9,
		OptionalIntPtr:    new(9),
		RequiredIntPtr:    new(9),
		NegativeInt:       0,
	}

	// Valid updates
	st.Value(&structOrig).OldValue(&structOrig).ExpectValid()
	st.Value(&structIncrease).OldValue(&structOrig).ExpectValid()

	// Invalid updates (decreases)
	st.Value(&structDecrease).OldValue(&structOrig).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), nil, "").WithOrigin("monotonic"),
		field.Invalid(field.NewPath("int64Field"), nil, "").WithOrigin("monotonic"),
		field.Invalid(field.NewPath("uint64Field"), nil, "").WithOrigin("monotonic"),
		field.Invalid(field.NewPath("intPtrField"), nil, "").WithOrigin("monotonic"),
		field.Invalid(field.NewPath("monotonicField"), nil, "").WithOrigin("monotonic"),
		field.Invalid(field.NewPath("monotonicPtrField"), nil, "").WithOrigin("monotonic"),
		field.Invalid(field.NewPath("optionalInt"), nil, "").WithOrigin("monotonic"),
		field.Invalid(field.NewPath("requiredInt"), nil, "").WithOrigin("monotonic"),
		field.Invalid(field.NewPath("optionalIntPtr"), nil, "").WithOrigin("monotonic"),
		field.Invalid(field.NewPath("requiredIntPtr"), nil, "").WithOrigin("monotonic"),
	})

	// Test special cases: zero and nil
	structZero := structOrig
	structZero.OptionalInt = 0
	structZero.RequiredInt = 0
	structZero.OptionalIntPtr = new(0)
	structZero.RequiredIntPtr = new(0)

	// OptionalInt (non-pointer) -> 0 is INVALID (Fails because +k8s:update=NoUnset is present).
	// RequiredInt -> 0 is INVALID (Fails because of +k8s:required).
	// OptionalIntPtr -> 0 is INVALID (Monotonic check detects decreased value).
	// RequiredIntPtr -> 0 is INVALID (Monotonic check detects decreased value).
	st.Value(&structZero).OldValue(&structOrig).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("optionalInt"), 0, "").WithOrigin("update"),
		field.Required(field.NewPath("requiredInt"), ""),
		field.Invalid(field.NewPath("optionalIntPtr"), 0, "").WithOrigin("monotonic"),
		field.Invalid(field.NewPath("requiredIntPtr"), 0, "").WithOrigin("monotonic"),
	})

	// OptionalIntPtr -> nil is INVALID because of +k8s:update=NoUnset
	structNil := structOrig
	structNil.OptionalIntPtr = nil
	st.Value(&structNil).OldValue(&structOrig).ExpectMatches(field.ErrorMatcher{}.ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("optionalIntPtr"), nil, "").WithOrigin("update"),
	})

	// Create should be valid (monotonic validation is update-only)
	st.Value(&structDecrease).ExpectValid()

	// OptionalIntPtr unset -> set should be valid (transition from nil to value)
	structSet := structOrig
	structSet.OptionalIntPtr = new(10)
	oldObjectSet := structSet
	oldObjectSet.OptionalIntPtr = nil
	st.Value(&structSet).OldValue(&oldObjectSet).ExpectValid()

	// OptionalIntPtr set -> unset is now INVALID due to +k8s:update=NoUnset
	structUnset := structOrig
	structUnset.OptionalIntPtr = nil
	st.Value(&structUnset).OldValue(&structOrig).ExpectMatches(field.ErrorMatcher{}.ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("optionalIntPtr"), nil, "").WithOrigin("update"),
	})

	// Invalid because it violates the minimum constraint (origin = "minimum")
	structBad := Struct{RequiredInt: 1, RequiredIntPtr: new(int)}
	structBad.NegativeInt = -11 // below the declared -10

	st.Value(&structBad).ExpectMatches(
		field.ErrorMatcher{}.ByOrigin(),
		field.ErrorList{
			field.Invalid(field.NewPath("negativeInt"), -11, "").
				WithOrigin("minimum"),
		})
}
