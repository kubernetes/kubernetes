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

package lists

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestListTypeMixed(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid case
	st.Value(&ListTypeMixed{
		List: []ComplexMapItem{
			{Key: "a", Inner: InnerItem{Value: 10}},
			{Key: "b", Inner: InnerItem{Value: 15}},
		},
		Set: []ComplexSetItem{
			{Inner: InnerItem{StringVal: "abc"}},
			{Inner: InnerItem{StringVal: "def"}},
		},
		MapField: map[string]string{"ab": "cd"},
	}).ExpectValid()

	// Fails shadow uniqueness, and normal item validation
	st.Value(&ListTypeMixed{
		List: []ComplexMapItem{
			{Key: "a", Inner: InnerItem{Value: 10}},
			{Key: "a", Inner: InnerItem{Value: 5}}, // Duplicate key (shadow), Value 5 (error)
		},
		Set: []ComplexSetItem{
			{Inner: InnerItem{StringVal: "abc"}},
			{Inner: InnerItem{StringVal: "abc"}},          // Duplicate value (shadow)
			{Inner: InnerItem{StringVal: "toolongvalue"}}, // Value "toolongvalue" (error)
		},
		MapField: map[string]string{"foo": "bar"},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByShadow(), field.ErrorList{
		// MapList: Duplicate key is SHADOWED
		field.Duplicate(field.NewPath("list").Index(1), map[string]interface{}{"key": "a"}).MarkShadow(),
		// MapList item: Value 5 < 10 is NORMAL ERROR (not shadowed)
		field.Invalid(field.NewPath("list").Index(1).Child("inner", "value"), 5, "").WithOrigin("minimum"),

		// Set: Duplicate value is SHADOWED
		// Note: The value in Duplicate error for struct is the struct itself.
		field.Duplicate(field.NewPath("set").Index(1), ComplexSetItem{Inner: InnerItem{StringVal: "abc"}}).MarkShadow(),
		// Set item: "toolongvalue" length 12 > 5 is NORMAL ERROR (not shadowed)
		field.TooLong(field.NewPath("set").Index(2).Child("inner", "stringVal"), "toolongvalue", 5).WithOrigin("maxLength"),

		// MapField: Too long key and value (Shadowed)
		field.TooLong(field.NewPath("mapField"), "foo", 2).WithOrigin("maxLength").MarkShadow(),
		field.TooLong(field.NewPath("mapField").Key("foo"), "bar", 2).WithOrigin("maxLength").MarkShadow(),
	})
}

func TestListItem(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&ListItemStruct{
		// Shadow list (duplicate key is shadow), Non-shadow item validation
		ShadowListNonShadowItem: []MapItem{
			{Key: "foo", Value: 5},  // Invalid value (normal) - First "foo" so it gets validated
			{Key: "foo", Value: 10}, // Duplicate key (shadow) - Second "foo"
		},
		// Shadow list (duplicate key is shadow), Shadow item validation
		ShadowListShadowItem: []MapItem{
			{Key: "foo", Value: 5},  // Invalid value (shadow) - First "foo"
			{Key: "foo", Value: 10}, // Duplicate key (shadow) - Second "foo"
		},
		// Non-shadow list (duplicate key is normal error), Mixed item validation
		MixedItems: []MapItem{
			{Key: "shadow", Value: 5},  // Invalid value (shadow)
			{Key: "normal", Value: 5},  // Invalid value (normal) - First "normal"
			{Key: "normal", Value: 10}, // Duplicate key (normal) - Second "normal"
		},
		// MultiKey Items
		MultiKeyItems: []MultiKeyItem{
			{Key1: "a", Key2: 10, Value: 5},  // Invalid value (shadow) - First "a"
			{Key1: "a", Key2: 10, Value: 10}, // Duplicate keys (normal) - Second "a"
		},
		// MapList
		MapList: []MapItem{
			{Key: "foo", Value: 5},  // Invalid value
			{Key: "foo", Value: 10}, // Duplicate key
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByShadow(), field.ErrorList{
		// ShadowListNonShadowItem
		field.Invalid(field.NewPath("shadowListNonShadowItem").Index(0).Child("value"), 5, "").WithOrigin("minimum"),
		field.Duplicate(field.NewPath("shadowListNonShadowItem").Index(1), map[string]interface{}{"key": "foo"}).MarkShadow(),

		// ShadowListShadowItem
		field.Invalid(field.NewPath("shadowListShadowItem").Index(0).Child("value"), 5, "").WithOrigin("minimum").MarkShadow(),
		field.Duplicate(field.NewPath("shadowListShadowItem").Index(1), map[string]interface{}{"key": "foo"}).MarkShadow(),

		// MixedItems
		field.Invalid(field.NewPath("mixedItems").Index(0).Child("value"), 5, "").WithOrigin("minimum").MarkShadow(),
		field.Invalid(field.NewPath("mixedItems").Index(1).Child("value"), 5, "").WithOrigin("minimum"),
		field.Duplicate(field.NewPath("mixedItems").Index(2), map[string]interface{}{"key": "normal"}),

		// MultiKeyItems
		field.Invalid(field.NewPath("multiKeyItems").Index(0).Child("value"), 5, "").WithOrigin("minimum").MarkShadow(),
		field.Duplicate(field.NewPath("multiKeyItems").Index(1), map[string]interface{}{"key1": "a", "key2": 10}),

		// MapList
		field.Duplicate(field.NewPath("mapList").Index(1), map[string]interface{}{"key": "foo"}).MarkShadow(),
	})
}
