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

package update

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestUpdateTags(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	baseStruct := UpdateTestStruct{}

	// String NoSet
	old := baseStruct
	old.StringNoSet = "" // unset

	new := baseStruct
	new.StringNoSet = "value"

	st.Value(&new).OldValue(&old).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("stringNoSet"), nil, "field cannot be set once created").WithOrigin("update"),
	})

	st.Value(&old).OldValue(&old).ExpectValid()

	// String NoUnset
	oldWithValue := baseStruct
	oldWithValue.StringNoUnset = "value"

	newUnset := baseStruct
	newUnset.StringNoUnset = ""

	// Can set initially (empty to non-empty)
	st.Value(&oldWithValue).OldValue(&baseStruct).ExpectValid()

	// Cannot unset (non-empty to empty)
	st.Value(&newUnset).OldValue(&oldWithValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("stringNoUnset"), nil, "field cannot be cleared once set").WithOrigin("update"),
	})

	// String NoModify
	oldEmpty := baseStruct
	withValue := baseStruct
	withValue.StringNoModify = "value"
	modified := baseStruct
	modified.StringNoModify = "different"

	// Can set initially (empty to non-empty)
	st.Value(&withValue).OldValue(&oldEmpty).ExpectValid()

	// Can unset (non-empty to empty)
	st.Value(&oldEmpty).OldValue(&withValue).ExpectValid()

	// Cannot modify (non-empty to different non-empty)
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("stringNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// String Fully Restricted
	oldEmpty = baseStruct
	withValue = baseStruct
	withValue.StringFullyRestricted = "value"
	modified = baseStruct
	modified.StringFullyRestricted = "different"

	// Cannot set (NoSet)
	st.Value(&withValue).OldValue(&oldEmpty).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("stringFullyRestricted"), nil, "field cannot be set once created").WithOrigin("update"),
	})

	// Cannot unset (NoUnset)
	st.Value(&oldEmpty).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("stringFullyRestricted"), nil, "field cannot be cleared once set").WithOrigin("update"),
	})

	// Cannot modify (NoModify)
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("stringFullyRestricted"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// String Set-Once Pattern
	oldEmpty = baseStruct
	withValue = baseStruct
	withValue.StringSetOnce = "value"
	modified = baseStruct
	modified.StringSetOnce = "different"

	// Can set once (empty to non-empty)
	st.Value(&withValue).OldValue(&oldEmpty).ExpectValid()

	// Cannot unset (NoUnset)
	st.Value(&oldEmpty).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("stringSetOnce"), nil, "field cannot be cleared once set").WithOrigin("update"),
	})

	// Cannot modify (NoModify)
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("stringSetOnce"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Int NoModify
	oldZero := baseStruct
	withValue = baseStruct
	withValue.IntNoModify = 10

	// Can transition from 0 to 10 (unset to set is allowed)
	st.Value(&withValue).OldValue(&oldZero).ExpectValid()

	// Cannot modify from one non-zero to another
	modified = baseStruct
	modified.IntNoModify = 20
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("intNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Int32 NoModify
	old = baseStruct
	withValue = baseStruct
	withValue.Int32NoModify = 42

	// Can set initially
	st.Value(&withValue).OldValue(&old).ExpectValid()

	// Cannot modify
	modified = baseStruct
	modified.Int32NoModify = 100
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("int32NoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Uint NoModify
	old = baseStruct
	withValue = baseStruct
	withValue.UintNoModify = 42

	// Can set initially
	st.Value(&withValue).OldValue(&old).ExpectValid()

	// Cannot modify
	modified = baseStruct
	modified.UintNoModify = 100
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("uintNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Bool NoModify
	oldFalse := baseStruct
	withTrue := baseStruct
	withTrue.BoolNoModify = true

	// Can transition from false to true (unset to set)
	st.Value(&withTrue).OldValue(&oldFalse).ExpectValid()

	// Cannot modify back to false
	st.Value(&oldFalse).OldValue(&withTrue).ExpectValid() // This is allowed as it's set->unset

	// Float32 NoModify
	old = baseStruct
	withValue = baseStruct
	withValue.Float32NoModify = 3.14

	// Can set initially
	st.Value(&withValue).OldValue(&old).ExpectValid()

	// Cannot modify
	modified = baseStruct
	modified.Float32NoModify = 2.71
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("float32NoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Float64 NoModify
	oldZero = baseStruct
	withValue = baseStruct
	withValue.Float64NoModify = 3.14

	// Can transition from 0.0 to 3.14
	st.Value(&withValue).OldValue(&oldZero).ExpectValid()

	// Cannot modify to different value
	modified = baseStruct
	modified.Float64NoModify = 2.71
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("float64NoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Byte NoModify
	old = baseStruct
	withValue = baseStruct
	withValue.ByteNoModify = 255

	// Can set initially
	st.Value(&withValue).OldValue(&old).ExpectValid()

	// Cannot modify
	modified = baseStruct
	modified.ByteNoModify = 128
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("byteNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Struct NoModify
	withStruct := baseStruct
	withStruct.StructNoModify = TestStruct{StringField: "value", IntField: 42}

	modifiedStruct := baseStruct
	modifiedStruct.StructNoModify = TestStruct{StringField: "different", IntField: 100}

	// Cannot modify (struct fields are always set, never unset)
	st.Value(&modifiedStruct).OldValue(&withStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("structNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// NonComparable Struct NoModify - uses reflection
	// For non-pointer structs, even zero value is considered "set"
	// So any change is a modification
	old = baseStruct
	old.NonComparableStructNoModify = NonComparableStruct{} // zero value

	withValue = baseStruct
	withValue.NonComparableStructNoModify = NonComparableStruct{
		SliceField: []string{"a", "b"},
		IntField:   42,
	}

	// Cannot change from zero value to non-zero (both are "set")
	st.Value(&withValue).OldValue(&old).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("nonComparableStructNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Also cannot modify between two non-zero values
	modified = baseStruct
	modified.NonComparableStructNoModify = NonComparableStruct{
		SliceField: []string{"c", "d"},
		IntField:   100,
	}
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("nonComparableStructNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Can keep the same value
	st.Value(&withValue).OldValue(&withValue).ExpectValid()

	// Pointer NoSet
	withSet := baseStruct
	withSet.PointerNoSet = ptr.To("value")

	// Cannot set after creation (nil to non-nil)
	st.Value(&withSet).OldValue(&baseStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("pointerNoSet"), nil, "field cannot be set once created").WithOrigin("update"),
	})

	// Pointer NoUnset
	withPointer := baseStruct
	withPointer.PointerNoUnset = ptr.To("value")

	// Can set initially
	st.Value(&withPointer).OldValue(&baseStruct).ExpectValid()

	// Cannot unset (non-nil to nil)
	st.Value(&baseStruct).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("pointerNoUnset"), nil, "field cannot be cleared once set").WithOrigin("update"),
	})

	// Pointer NoModify
	withPointer = baseStruct
	withPointer.PointerNoModify = ptr.To("value")

	modifiedPointer := baseStruct
	modifiedPointer.PointerNoModify = ptr.To("different")

	// Can set initially
	st.Value(&withPointer).OldValue(&baseStruct).ExpectValid()

	// Can unset (NoModify allows set/unset transitions)
	st.Value(&baseStruct).OldValue(&withPointer).ExpectValid()

	// Cannot modify content
	st.Value(&modifiedPointer).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("pointerNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Pointer Fully Restricted
	withPointer = baseStruct
	withPointer.PointerFullyRestricted = ptr.To("value")

	modifiedPointer = baseStruct
	modifiedPointer.PointerFullyRestricted = ptr.To("different")

	// Cannot set (NoSet)
	st.Value(&withPointer).OldValue(&baseStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("pointerFullyRestricted"), nil, "field cannot be set once created").WithOrigin("update"),
	})

	// Cannot unset (NoUnset)
	st.Value(&baseStruct).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("pointerFullyRestricted"), nil, "field cannot be cleared once set").WithOrigin("update"),
	})

	// Cannot modify (NoModify)
	st.Value(&modifiedPointer).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("pointerFullyRestricted"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Int Pointer NoModify
	withPointer = baseStruct
	withPointer.IntPointerNoModify = ptr.To(42)

	modifiedPointer = baseStruct
	modifiedPointer.IntPointerNoModify = ptr.To(100)

	// Can set initially
	st.Value(&withPointer).OldValue(&baseStruct).ExpectValid()

	// Can unset
	st.Value(&baseStruct).OldValue(&withPointer).ExpectValid()

	// Cannot modify content
	st.Value(&modifiedPointer).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("intPointerNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Bool Pointer NoModify
	falseVal := false
	trueVal := true

	withFalse := baseStruct
	withFalse.BoolPointerNoModify = &falseVal

	withTrue = baseStruct
	withTrue.BoolPointerNoModify = &trueVal

	// Can set initially
	st.Value(&withFalse).OldValue(&baseStruct).ExpectValid()

	// Cannot modify content
	st.Value(&withTrue).OldValue(&withFalse).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("boolPointerNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Struct Pointer NoModify
	withPointer = baseStruct
	withPointer.StructPointerNoModify = &TestStruct{StringField: "value", IntField: 42}

	modifiedPointer = baseStruct
	modifiedPointer.StructPointerNoModify = &TestStruct{StringField: "different", IntField: 100}

	// Can set initially
	st.Value(&withPointer).OldValue(&baseStruct).ExpectValid()

	// Can unset
	st.Value(&baseStruct).OldValue(&withPointer).ExpectValid()

	// Cannot modify content
	st.Value(&modifiedPointer).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("structPointerNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Custom Type NoModify
	old = baseStruct
	withValue = baseStruct
	withValue.CustomTypeNoModify = "custom-value"

	// Can set initially
	st.Value(&withValue).OldValue(&old).ExpectValid()

	// Cannot modify
	modified = baseStruct
	modified.CustomTypeNoModify = "different-value"
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("customTypeNoModify"), nil, "field cannot be modified once set").WithOrigin("update"),
	})

	// Custom Type NoSet
	old = baseStruct
	withValue = baseStruct
	withValue.CustomTypeNoSet = 42

	// Cannot set
	st.Value(&withValue).OldValue(&old).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("customTypeNoSet"), nil, "field cannot be set once created").WithOrigin("update"),
	})
}
