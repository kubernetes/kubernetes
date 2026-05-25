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

package maps

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestUpdateMapTags(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	matcher := field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin()
	base := UpdateMapStruct{}

	// Map NoSet/NoUnset
	{
		old := base
		cur := base
		cur.MapNoSet = map[string]string{"a": "1"}
		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Invalid(field.NewPath("mapNoSet"), nil, "field cannot be set once created").WithOrigin("update"),
		})
	}
	{
		old := base
		old.MapNoUnset = map[string]string{"a": "1"}
		cur := base
		cur.MapNoUnset = nil
		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Invalid(field.NewPath("mapNoUnset"), nil, "field cannot be cleared once set").WithOrigin("update"),
		})
	}

	// Map NoAddItem/NoRemoveItem
	{
		old := base
		old.MapNoAdd = map[string]string{"a": "1"}
		cur := base
		cur.MapNoAdd = map[string]string{"a": "1", "b": "2"}
		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("mapNoAdd").Key("b"), "item may not be added").WithOrigin("update"),
		})

		// Value-only change to an existing key is allowed.
		modify := base
		modify.MapNoAdd = map[string]string{"a": "99"}
		st.Value(&modify).OldValue(&old).ExpectValid()
	}
	{
		old := base
		old.MapNoRemove = map[string]string{"a": "1", "b": "2"}
		cur := base
		cur.MapNoRemove = map[string]string{"a": "1"}
		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("mapNoRemove").Key("b"), "item may not be removed").WithOrigin("update"),
		})
	}

	// Map frozen shape
	{
		old := base
		old.MapFrozenShape = map[string]string{"a": "1", "b": "2"}

		// Value-only change: allowed.
		modify := base
		modify.MapFrozenShape = map[string]string{"a": "11", "b": "22"}
		st.Value(&modify).OldValue(&old).ExpectValid()

		// Swap one key: one add + one remove.
		swap := base
		swap.MapFrozenShape = map[string]string{"a": "1", "c": "3"}
		st.Value(&swap).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("mapFrozenShape").Key("c"), "item may not be added").WithOrigin("update"),
			field.Forbidden(field.NewPath("mapFrozenShape").Key("b"), "item may not be removed").WithOrigin("update"),
		})
	}

	// Map combined NoSet + NoAddItem
	{
		old := base
		cur := base
		cur.MapSetThenFreeze = map[string]MapItem{"k": {Name: "k", Value: "v"}}

		// Both constraints fire: NoSet on the empty->non-empty transition
		// and NoAddItem on the new key.
		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Invalid(field.NewPath("mapSetThenFreeze"), nil, "field cannot be set once created").WithOrigin("update"),
			field.Forbidden(field.NewPath("mapSetThenFreeze").Key("k"), "item may not be added").WithOrigin("update"),
		})
	}

	// eachVal composition on a map: per-value NoModify
	{
		old := base
		old.EachValNoModifyMap = map[string]MapItem{
			"a": {Name: "a", Value: "1"},
		}

		// Adding/removing keys is fine.
		add := base
		add.EachValNoModifyMap = map[string]MapItem{
			"a": {Name: "a", Value: "1"},
			"b": {Name: "b", Value: "2"},
		}
		st.Value(&add).OldValue(&old).ExpectValid()

		// Mutating an existing value fires.
		mutate := base
		mutate.EachValNoModifyMap = map[string]MapItem{
			"a": {Name: "a", Value: "99"},
		}
		st.Value(&mutate).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Invalid(field.NewPath("eachValNoModifyMap").Key("a"), nil, "cannot be modified").WithOrigin("update"),
		})
	}
}
