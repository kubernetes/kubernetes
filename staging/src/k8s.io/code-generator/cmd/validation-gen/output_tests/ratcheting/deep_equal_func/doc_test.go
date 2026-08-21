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
)

func TestDeepEqualFunc(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Create
	val1 := 10
	obj := &Struct{
		NonComparableField: NonComparableStruct{Ptr: &val1},
		MapField: map[string]NonComparableStruct{
			"a": {Ptr: &val1},
		},
	}
	st.Value(obj).ExpectValidateFalseByPath(map[string][]string{
		"nonComparableField": {"field Struct.NonComparableField", "type NonComparableStruct"},
		"mapField[a]":        {"field Struct.MapField[*]", "type NonComparableStruct"},
	})

	// Update with same values should ratchet (skip validation) using CustomDeepEqual
	CustomDeepEqualCalls = 0
	oldObj := &Struct{
		NonComparableField: NonComparableStruct{Ptr: &val1},
		MapField: map[string]NonComparableStruct{
			"a": {Ptr: &val1},
		},
	}
	newObj := &Struct{
		NonComparableField: NonComparableStruct{Ptr: &val1},
		MapField: map[string]NonComparableStruct{
			"a": {Ptr: &val1},
		},
	}
	st.Value(newObj).OldValue(oldObj).ExpectValid()
	if CustomDeepEqualCalls == 0 {
		t.Errorf("expected CustomDeepEqual to be called during update ratcheting, got %d calls", CustomDeepEqualCalls)
	}

	// Update with a new map key added: old key "a" ratchets via deepEqualImpl_ -> CustomDeepEqual,
	// only new key "b" fails.
	val2 := 20
	updateObj := &Struct{
		NonComparableField: NonComparableStruct{Ptr: &val1},
		MapField: map[string]NonComparableStruct{
			"a": {Ptr: &val1},
			"b": {Ptr: &val2},
		},
	}
	st.Value(updateObj).OldValue(oldObj).ExpectValidateFalseByPath(map[string][]string{
		"mapField[b]": {"field Struct.MapField[*]", "type NonComparableStruct"},
	})
}
