/*
Copyright The Kubernetes Authors.

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

package deepequalfunc

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestDeepEqualFunc(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Create
	val1, val2 := 10, 20
	obj := &Struct{
		NonComparableField: NonComparableStruct{Ptr: &val1},
		MapField: map[string]NonComparableStruct{
			"a": {Ptr: &val1},
		},
		SetField: []NonComparableStruct{{Ptr: &val1}},
	}
	st.Value(obj).ExpectValidateFalseByPath(map[string][]string{
		"nonComparableField": {"field Struct.NonComparableField", "type NonComparableStruct"},
		"mapField[a]":        {"field Struct.MapField[*]", "type NonComparableStruct"},
		"setField[0]":        {"field Struct.SetField[*]", "type NonComparableStruct"},
	})

	// CustomDeepEqual treats two non-nil Ptrs as equal, so changing only the
	// value they point to skips revalidation.  Semantic deep-equal would not.
	// Each field changes on its own to isolate one call site.
	CustomDeepEqualCalls = 0
	oldObj := &Struct{
		NonComparableField: NonComparableStruct{Ptr: &val1},
		MapField: map[string]NonComparableStruct{
			"a": {Ptr: &val1},
		},
		SetField: []NonComparableStruct{{Ptr: &val1}},
	}
	// field ratcheting check
	changed := *oldObj
	changed.NonComparableField = NonComparableStruct{Ptr: &val2}
	st.Value(&changed).OldValue(oldObj).ExpectValid()
	// EachMapVal equiv
	changed = *oldObj
	changed.MapField = map[string]NonComparableStruct{"a": {Ptr: &val2}}
	st.Value(&changed).OldValue(oldObj).ExpectValid()
	// EachValSliceVal match
	changed = *oldObj
	changed.SetField = []NonComparableStruct{{Ptr: &val2}}
	st.Value(&changed).OldValue(oldObj).ExpectValid()
	if CustomDeepEqualCalls == 0 {
		t.Errorf("expected CustomDeepEqual to be called during update ratcheting, got %d calls", CustomDeepEqualCalls)
	}

	// Update with a new map key added: old key "a" ratchets via CustomDeepEqual,
	// only new key "b" fails.
	updateObj := &Struct{
		NonComparableField: NonComparableStruct{Ptr: &val1},
		MapField: map[string]NonComparableStruct{
			"a": {Ptr: &val1},
			"b": {Ptr: &val2},
		},
		SetField: []NonComparableStruct{{Ptr: &val1}},
	}
	st.Value(updateObj).OldValue(oldObj).ExpectValidateFalseByPath(map[string][]string{
		"mapField[b]": {"field Struct.MapField[*]", "type NonComparableStruct"},
	})

	// Uniqueness also runs through CustomDeepEqual: the first two elements are
	// duplicates despite different values, while the unset Ptr is distinct.
	st.Value(&Struct{
		SetField: []NonComparableStruct{{Ptr: &val1}, {Ptr: &val2}, {Ptr: nil}},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring(), field.ErrorList{
		field.Duplicate(field.NewPath("setField").Index(1), nil),
		field.Invalid(field.NewPath("nonComparableField"), nil, "field Struct.NonComparableField"),
		field.Invalid(field.NewPath("nonComparableField"), nil, "type NonComparableStruct"),
		field.Invalid(field.NewPath("setField").Index(0), nil, "field Struct.SetField[*]"),
		field.Invalid(field.NewPath("setField").Index(0), nil, "type NonComparableStruct"),
		field.Invalid(field.NewPath("setField").Index(1), nil, "field Struct.SetField[*]"),
		field.Invalid(field.NewPath("setField").Index(1), nil, "type NonComparableStruct"),
		field.Invalid(field.NewPath("setField").Index(2), nil, "field Struct.SetField[*]"),
		field.Invalid(field.NewPath("setField").Index(2), nil, "type NonComparableStruct"),
	})
}
