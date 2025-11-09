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

package unique

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestUnique(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Test empty struct (should be valid)
	st.Value(&Struct{}).ExpectValid()

	// Test valid cases with no duplicates
	st.Value(&Struct{
		PrimitiveListUniqueSet: []string{"aaa", "bbb"},
		SliceMapFieldWithMultipleKeys: []ItemWithMultipleKeys{
			{Key1: "a", Key2: "x", Data: "first"},
			{Key1: "a", Key2: "y", Data: "second"},
		},
		AtomicListUniqueSet: []Item{
			{Key: "key1", Data: "one"},
			{Key: "key2", Data: "two"},
		},
		AtomicListUniqueMap: []Item{
			{Key: "key1", Data: "one"},
			{Key: "key2", Data: "two"},
		},
		CustomUniqueListWithTypeSet: []string{"a", "b", "a"},
		CustomUniqueListWithTypeMap: []Item{{Key: "a"}, {Key: "b"}, {Key: "a"}},
	}).ExpectValid()

	// Test empty lists
	st.Value(&Struct{
		PrimitiveListUniqueSet:        []string{},
		SliceMapFieldWithMultipleKeys: []ItemWithMultipleKeys{},
		AtomicListUniqueSet:           []Item{},
		AtomicListUniqueMap:           []Item{},
		CustomUniqueListWithTypeSet:   []string{},
		CustomUniqueListWithTypeMap:   []Item{},
	}).ExpectValid()

	// Test single element lists
	st.Value(&Struct{
		PrimitiveListUniqueSet:        []string{"single"},
		SliceMapFieldWithMultipleKeys: []ItemWithMultipleKeys{{Key1: "a", Key2: "b", Data: "one"}},
		AtomicListUniqueSet:           []Item{{Key: "single", Data: "one"}},
		AtomicListUniqueMap:           []Item{{Key: "single", Data: "one"}},
		CustomUniqueListWithTypeSet:   []string{"single"},
		CustomUniqueListWithTypeMap:   []Item{{Key: "single"}},
	}).ExpectValid()

	// Test duplicate values (should fail validation)
	st.Value(&Struct{
		PrimitiveListUniqueSet: []string{"aaa", "bbb", "ccc", "ccc", "bbb", "aaa"},
		SliceMapFieldWithMultipleKeys: []ItemWithMultipleKeys{
			{Key1: "a", Key2: "x", Data: "first"},
			{Key1: "a", Key2: "y", Data: "second"},
			{Key1: "a", Key2: "x", Data: "third"},
		},
		AtomicListUniqueSet: []Item{
			{Key: "key1", Data: "one"},
			{Key: "key2", Data: "two"},
			{Key: "key1", Data: "one"},
		},
		AtomicListUniqueMap: []Item{
			{Key: "key1", Data: "one"},
			{Key: "key2", Data: "two"},
			{Key: "key1", Data: "three"},
		},
		CustomUniqueListWithTypeSet: []string{"a", "b", "a"},
		CustomUniqueListWithTypeMap: []Item{{Key: "a"}, {Key: "b"}, {Key: "a"}},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Duplicate(field.NewPath("primitiveListUniqueSet").Index(3), nil),
		field.Duplicate(field.NewPath("primitiveListUniqueSet").Index(4), nil),
		field.Duplicate(field.NewPath("primitiveListUniqueSet").Index(5), nil),
		field.Duplicate(field.NewPath("sliceMapFieldWithMultipleKeys").Index(2), nil),
		field.Duplicate(field.NewPath("atomicListUniqueSet").Index(2), nil),
		field.Duplicate(field.NewPath("atomicListUniqueMap").Index(2), nil),
	})

	// Test with zero values and empty strings
	st.Value(&Struct{
		PrimitiveListUniqueSet: []string{"", "a", ""},
		AtomicListUniqueMap: []Item{
			{Key: "", Data: "one"},
			{Key: "a", Data: "two"},
			{Key: "", Data: "three"},
		},
		CustomUniqueListWithTypeSet: []string{"", "a", ""},
		CustomUniqueListWithTypeMap: []Item{{Key: ""}, {Key: "a"}, {Key: ""}},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Duplicate(field.NewPath("primitiveListUniqueSet").Index(2), nil),
		field.Duplicate(field.NewPath("atomicListUniqueMap").Index(2), nil),
	})
}

func TestRatcheting(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	struct1 := Struct{
		PrimitiveListUniqueSet: []string{"aaa", "bbb"},
		SliceMapFieldWithMultipleKeys: []ItemWithMultipleKeys{
			{Key1: "a", Key2: "x", Data: "first"},
			{Key1: "a", Key2: "y", Data: "second"},
		},
		AtomicListUniqueSet: []Item{
			{Key: "key1", Data: "one"},
			{Key: "key2", Data: "two"},
		},
		AtomicListUniqueMap: []Item{
			{Key: "key1", Data: "one"},
			{Key: "key2", Data: "two"},
		},
		CustomUniqueListWithTypeSet: []string{"a", "b", "a"},
		CustomUniqueListWithTypeMap: []Item{{Key: "a"}, {Key: "b"}, {Key: "a"}},
	}

	// Same data, different order.
	struct2 := Struct{
		PrimitiveListUniqueSet: []string{"bbb", "aaa"},
		SliceMapFieldWithMultipleKeys: []ItemWithMultipleKeys{
			{Key1: "a", Key2: "y", Data: "second"},
			{Key1: "a", Key2: "x", Data: "first"},
		},
		AtomicListUniqueSet: []Item{
			{Key: "key2", Data: "two"},
			{Key: "key1", Data: "one"},
		},
		AtomicListUniqueMap: []Item{
			{Key: "key2", Data: "two"},
			{Key: "key1", Data: "one"},
		},
		CustomUniqueListWithTypeSet: []string{"a", "a", "b"},
		CustomUniqueListWithTypeMap: []Item{{Key: "a"}, {Key: "a"}, {Key: "b"}},
	}

	// Test that reordering doesn't trigger validation errors
	st.Value(&struct1).OldValue(&struct2).ExpectValid()
	st.Value(&struct2).OldValue(&struct1).ExpectValid()

	// Test that the same data is considered valid regardless of order
	st.Value(&struct1).ExpectValid()
	st.Value(&struct2).ExpectValid()
}
