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

package lists

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestUpdateListTags(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	matcher := field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin()
	base := UpdateListStruct{}

	// Slice NoSet (len == 0 semantics)
	{
		old := base
		old.StringSliceNoSet = nil
		cur := base
		cur.StringSliceNoSet = []string{"a"}

		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Invalid(field.NewPath("stringSliceNoSet"), nil, "field cannot be set once created").WithOrigin("update"),
		})

		// nil -> nil is allowed
		st.Value(&old).OldValue(&old).ExpectValid()
	}

	// Slice NoUnset
	{
		old := base
		old.StringSliceNoUnset = []string{"a"}
		cur := base
		cur.StringSliceNoUnset = nil

		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Invalid(field.NewPath("stringSliceNoUnset"), nil, "field cannot be cleared once set").WithOrigin("update"),
		})
	}

	// listType=set + NoAddItem
	{
		old := base
		old.StringSetNoAdd = []string{"a", "b"}
		cur := base
		cur.StringSetNoAdd = []string{"a", "b", "c"}

		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("stringSetNoAdd").Index(2), "item may not be added").WithOrigin("update"),
		})

		// Reorder of existing items is allowed.
		reordered := base
		reordered.StringSetNoAdd = []string{"b", "a"}
		st.Value(&reordered).OldValue(&old).ExpectValid()
	}

	// listType=set + NoRemoveItem
	{
		old := base
		old.StringSetNoRemove = []string{"a", "b", "c"}
		cur := base
		cur.StringSetNoRemove = []string{"a", "c"}

		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("stringSetNoRemove"), "may not be removed").WithOrigin("update"),
		})
	}

	// Frozen-shape set: NoAddItem + NoRemoveItem
	{
		old := base
		old.StringSetFrozenShape = []string{"a", "b"}

		// Permutation is valid.
		perm := base
		perm.StringSetFrozenShape = []string{"b", "a"}
		st.Value(&perm).OldValue(&old).ExpectValid()

		// Swap one item: one add + one remove, both reported.
		swap := base
		swap.StringSetFrozenShape = []string{"a", "c"}
		st.Value(&swap).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("stringSetFrozenShape").Index(1), "item may not be added").WithOrigin("update"),
			field.Forbidden(field.NewPath("stringSetFrozenShape"), "may not be removed").WithOrigin("update"),
		})
	}

	// listType=map + single key + NoAddItem
	{
		old := base
		old.MapListNoAdd = []UpdateItem{{Name: "alpha", Value: "1"}}
		cur := base
		cur.MapListNoAdd = []UpdateItem{
			{Name: "alpha", Value: "1"},
			{Name: "beta", Value: "2"},
		}

		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("mapListNoAdd").Index(1), "item may not be added").WithOrigin("update"),
		})

		// Modifying the value of an existing keyed item is not an "add".
		modified := base
		modified.MapListNoAdd = []UpdateItem{{Name: "alpha", Value: "999"}}
		st.Value(&modified).OldValue(&old).ExpectValid()
	}

	// listType=map + single key + frozen shape
	{
		old := base
		old.MapListFrozenShape = []UpdateItem{
			{Name: "alpha", Value: "1"},
			{Name: "beta", Value: "2"},
		}

		// Value-only change to an existing keyed item is allowed.
		modify := base
		modify.MapListFrozenShape = []UpdateItem{
			{Name: "alpha", Value: "99"},
			{Name: "beta", Value: "2"},
		}
		st.Value(&modify).OldValue(&old).ExpectValid()

		// Removing a key produces a NoRemoveItem error.
		removed := base
		removed.MapListFrozenShape = []UpdateItem{{Name: "alpha", Value: "1"}}
		st.Value(&removed).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("mapListFrozenShape"), "may not be removed").WithOrigin("update"),
		})
	}

	// listType=map + composite key + NoRemoveItem
	{
		old := base
		old.CompositeKeyList = []CompositeKeyItem{
			{Name: "alpha", Priority: 1, Value: "x"},
			{Name: "alpha", Priority: 2, Value: "y"},
		}

		// Same key pair, different value: allowed.
		modify := base
		modify.CompositeKeyList = []CompositeKeyItem{
			{Name: "alpha", Priority: 1, Value: "x"},
			{Name: "alpha", Priority: 2, Value: "y2"},
		}
		st.Value(&modify).OldValue(&old).ExpectValid()

		// Remove one pair entirely: rejected.
		removed := base
		removed.CompositeKeyList = []CompositeKeyItem{
			{Name: "alpha", Priority: 1, Value: "x"},
		}
		st.Value(&removed).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("compositeKeyList"), "may not be removed").WithOrigin("update"),
		})
	}

	// listType=atomic + unique=set + NoAddItem
	{
		old := base
		old.AtomicUniqueSetNoAdd = []string{"a"}
		cur := base
		cur.AtomicUniqueSetNoAdd = []string{"a", "b"}
		st.Value(&cur).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("atomicUniqueSetNoAdd").Index(1), "item may not be added").WithOrigin("update"),
		})
	}

	// listType=atomic + unique=map: semanticMap path via unique=
	{
		old := base
		old.AtomicUniqueMapFrozenShape = []UpdateItem{
			{Name: "alpha", Value: "1"},
			{Name: "beta", Value: "2"},
		}

		// Value-only change on an existing keyed item is allowed. This
		// matches listType=map behavior.
		modify := base
		modify.AtomicUniqueMapFrozenShape = []UpdateItem{
			{Name: "alpha", Value: "99"},
			{Name: "beta", Value: "2"},
		}
		st.Value(&modify).OldValue(&old).ExpectValid()

		// Adding a new key is rejected.
		add := base
		add.AtomicUniqueMapFrozenShape = []UpdateItem{
			{Name: "alpha", Value: "1"},
			{Name: "beta", Value: "2"},
			{Name: "gamma", Value: "3"},
		}
		st.Value(&add).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("atomicUniqueMapFrozenShape").Index(2), "item may not be added").WithOrigin("update"),
		})
	}

	// listType=set over non-directly-comparable elements
	{
		old := base
		old.NonComparableSetFrozenShape = []NonComparableItem{
			{Name: "alpha", Tags: []string{"x", "y"}},
			{Name: "beta", Tags: []string{"z"}},
		}

		// Reordering full-value items is allowed (SemanticDeepEqual matches
		// each one in the old list).
		reordered := base
		reordered.NonComparableSetFrozenShape = []NonComparableItem{
			{Name: "beta", Tags: []string{"z"}},
			{Name: "alpha", Tags: []string{"x", "y"}},
		}
		st.Value(&reordered).OldValue(&old).ExpectValid()

		// Mutating a tag inside one element changes its identity under
		// set semantics (the whole element is the key), so this is
		// reported as both an add and a remove.
		mutated := base
		mutated.NonComparableSetFrozenShape = []NonComparableItem{
			{Name: "alpha", Tags: []string{"x", "y", "NEW"}},
			{Name: "beta", Tags: []string{"z"}},
		}
		st.Value(&mutated).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("nonComparableSetFrozenShape").Index(0), "item may not be added").WithOrigin("update"),
			field.Forbidden(field.NewPath("nonComparableSetFrozenShape"), "may not be removed").WithOrigin("update"),
		})
	}

	// Typedef list: metadata inherited from type
	{
		old := base
		old.TypedefFrozenList = FrozenUserList{{Name: "alpha", Value: "1"}}

		// Value-only mutation of the existing keyed item: allowed.
		modify := base
		modify.TypedefFrozenList = FrozenUserList{{Name: "alpha", Value: "2"}}
		st.Value(&modify).OldValue(&old).ExpectValid()

		// Add a new keyed item: rejected.
		add := base
		add.TypedefFrozenList = FrozenUserList{
			{Name: "alpha", Value: "1"},
			{Name: "beta", Value: "9"},
		}
		st.Value(&add).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Forbidden(field.NewPath("typedefFrozenList").Index(1), "item may not be added").WithOrigin("update"),
		})
	}

	// eachVal composition on a list: per-item NoModify
	{
		old := base
		old.EachValNoModifyList = []UpdateItem{
			{Name: "alpha", Value: "1"},
			{Name: "beta", Value: "2"},
		}

		// Adding a new keyed item is allowed. NoModify is per-item.
		add := base
		add.EachValNoModifyList = []UpdateItem{
			{Name: "alpha", Value: "1"},
			{Name: "beta", Value: "2"},
			{Name: "gamma", Value: "3"},
		}
		st.Value(&add).OldValue(&old).ExpectValid()

		// Removing a keyed item is allowed. NoModify is per-item.
		remove := base
		remove.EachValNoModifyList = []UpdateItem{{Name: "alpha", Value: "1"}}
		st.Value(&remove).OldValue(&old).ExpectValid()

		// Mutating the non-key Value field of an existing keyed item fires.
		mutate := base
		mutate.EachValNoModifyList = []UpdateItem{
			{Name: "alpha", Value: "99"},
			{Name: "beta", Value: "2"},
		}
		st.Value(&mutate).OldValue(&old).ExpectMatches(matcher, field.ErrorList{
			field.Invalid(field.NewPath("eachValNoModifyList").Index(0), nil, "cannot be modified").WithOrigin("update"),
		})
	}
}
