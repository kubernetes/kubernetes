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

	st.Value(&new).OldValue(&old).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("stringNoSet"), "field cannot be set once created"),
	})

	st.Value(&old).OldValue(&old).ExpectValid()

	/// String NoUnset
	oldWithValue := baseStruct
	oldWithValue.StringNoUnset = "value"

	newUnset := baseStruct
	newUnset.StringNoUnset = ""

	// Can set initially (empty to non-empty)
	st.Value(&oldWithValue).OldValue(&baseStruct).ExpectValid()

	// Cannot unset (non-empty to empty)
	st.Value(&newUnset).OldValue(&oldWithValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("stringNoUnset"), "field cannot be cleared once set"),
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
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("stringNoModify"), "field cannot be modified once set"),
	})

	// String Fully Restricted
	oldEmpty = baseStruct
	withValue = baseStruct
	withValue.StringFullyRestricted = "value"
	modified = baseStruct
	modified.StringFullyRestricted = "different"

	// Cannot set (NoSet)
	st.Value(&withValue).OldValue(&oldEmpty).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("stringFullyRestricted"), "field cannot be set once created"),
	})

	// Cannot unset (NoUnset)
	st.Value(&oldEmpty).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("stringFullyRestricted"), "field cannot be cleared once set"),
	})

	// Cannot modify (NoModify)
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("stringFullyRestricted"), "field cannot be modified once set"),
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
	st.Value(&oldEmpty).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("stringSetOnce"), "field cannot be cleared once set"),
	})

	// Cannot modify (NoModify)
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("stringSetOnce"), "field cannot be modified once set"),
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
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("intNoModify"), "field cannot be modified once set"),
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
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("int32NoModify"), "field cannot be modified once set"),
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
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("uintNoModify"), "field cannot be modified once set"),
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
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("float32NoModify"), "field cannot be modified once set"),
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
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("float64NoModify"), "field cannot be modified once set"),
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
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("byteNoModify"), "field cannot be modified once set"),
	})

	// Struct NoModify
	withStruct := baseStruct
	withStruct.StructNoModify = TestStruct{StringField: "value", IntField: 42}

	modifiedStruct := baseStruct
	modifiedStruct.StructNoModify = TestStruct{StringField: "different", IntField: 100}

	// Cannot modify (struct fields are always set, never unset)
	st.Value(&modifiedStruct).OldValue(&withStruct).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("structNoModify"), "field cannot be modified once set"),
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
	st.Value(&withValue).OldValue(&old).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("nonComparableStructNoModify"), "field cannot be modified once set"),
	})

	// Also cannot modify between two non-zero values
	modified = baseStruct
	modified.NonComparableStructNoModify = NonComparableStruct{
		SliceField: []string{"c", "d"},
		IntField:   100,
	}
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("nonComparableStructNoModify"), "field cannot be modified once set"),
	})

	// Can keep the same value
	st.Value(&withValue).OldValue(&withValue).ExpectValid()

	// Pointer NoSet
	withSet := baseStruct
	withSet.PointerNoSet = ptr.To("value")

	// Cannot set after creation (nil to non-nil)
	st.Value(&withSet).OldValue(&baseStruct).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("pointerNoSet"), "field cannot be set once created"),
	})

	// Pointer NoUnset
	withPointer := baseStruct
	withPointer.PointerNoUnset = ptr.To("value")

	// Can set initially
	st.Value(&withPointer).OldValue(&baseStruct).ExpectValid()

	// Cannot unset (non-nil to nil)
	st.Value(&baseStruct).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("pointerNoUnset"), "field cannot be cleared once set"),
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
	st.Value(&modifiedPointer).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("pointerNoModify"), "field cannot be modified once set"),
	})

	// Pointer Fully Restricted
	withPointer = baseStruct
	withPointer.PointerFullyRestricted = ptr.To("value")

	modifiedPointer = baseStruct
	modifiedPointer.PointerFullyRestricted = ptr.To("different")

	// Cannot set (NoSet)
	st.Value(&withPointer).OldValue(&baseStruct).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("pointerFullyRestricted"), "field cannot be set once created"),
	})

	// Cannot unset (NoUnset)
	st.Value(&baseStruct).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("pointerFullyRestricted"), "field cannot be cleared once set"),
	})

	// Cannot modify (NoModify)
	st.Value(&modifiedPointer).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("pointerFullyRestricted"), "field cannot be modified once set"),
	})

	// Int Pointer NoModify"
	withPointer = baseStruct
	withPointer.IntPointerNoModify = ptr.To(42)

	modifiedPointer = baseStruct
	modifiedPointer.IntPointerNoModify = ptr.To(100)

	// Can set initially
	st.Value(&withPointer).OldValue(&baseStruct).ExpectValid()

	// Can unset
	st.Value(&baseStruct).OldValue(&withPointer).ExpectValid()

	// Cannot modify content
	st.Value(&modifiedPointer).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("intPointerNoModify"), "field cannot be modified once set"),
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
	st.Value(&withTrue).OldValue(&withFalse).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("boolPointerNoModify"), "field cannot be modified once set"),
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
	st.Value(&modifiedPointer).OldValue(&withPointer).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("structPointerNoModify"), "field cannot be modified once set"),
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
	st.Value(&modified).OldValue(&withValue).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("customTypeNoModify"), "field cannot be modified once set"),
	})

	// Custom Type NoSet
	old = baseStruct
	withValue = baseStruct
	withValue.CustomTypeNoSet = 42

	// Cannot set
	st.Value(&withValue).OldValue(&old).ExpectMatches(field.ErrorMatcher{}, field.ErrorList{
		field.Forbidden(field.NewPath("customTypeNoSet"), "field cannot be set once created"),
	})
}
