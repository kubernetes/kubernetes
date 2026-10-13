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
		SliceMapFieldWithPtrKey: []PtrKeyStruct{
			{Key: new("a"), Data: "first"},
			{Key: new("b"), Data: "second"},
		},
		SliceMapFieldWithMixedKeys: []ItemWithMixedKeys{
			{Key1: new("a"), Key2: "x", Data: "first"},
			{Key1: new("a"), Key2: "y", Data: "second"},
		},
		SliceMapFieldWithMultiplePtrKeys: []ItemWithMultiplePtrKeys{
			{Key1: new("a"), Key2: new("x"), Data: "first"},
			{Key1: new("a"), Key2: new("y"), Data: "second"},
		},
		PrimitivePointerListUniqueSet: []*string{new("aaa"), new("bbb")},
		PointerListUniqueMap: []*Item{
			{Key: "key1", Data: "one"},
			{Key: "key2", Data: "two"},
		},
	}).ExpectValid()

	// Test empty lists
	st.Value(&Struct{
		PrimitiveListUniqueSet:           []string{},
		SliceMapFieldWithMultipleKeys:    []ItemWithMultipleKeys{},
		AtomicListUniqueSet:              []Item{},
		AtomicListUniqueMap:              []Item{},
		CustomUniqueListWithTypeSet:      []string{},
		CustomUniqueListWithTypeMap:      []Item{},
		SliceMapFieldWithMixedKeys:       []ItemWithMixedKeys{},
		SliceMapFieldWithMultiplePtrKeys: []ItemWithMultiplePtrKeys{},
		PrimitivePointerListUniqueSet:    []*string{},
		PointerListUniqueMap:             []*Item{},
	}).ExpectValid()

	// Test single element lists
	st.Value(&Struct{
		PrimitiveListUniqueSet:           []string{"single"},
		SliceMapFieldWithMultipleKeys:    []ItemWithMultipleKeys{{Key1: "a", Key2: "b", Data: "one"}},
		AtomicListUniqueSet:              []Item{{Key: "single", Data: "one"}},
		AtomicListUniqueMap:              []Item{{Key: "single", Data: "one"}},
		CustomUniqueListWithTypeSet:      []string{"single"},
		CustomUniqueListWithTypeMap:      []Item{{Key: "single"}},
		SliceMapFieldWithMixedKeys:       []ItemWithMixedKeys{{Key1: new("a"), Key2: "b", Data: "one"}},
		SliceMapFieldWithMultiplePtrKeys: []ItemWithMultiplePtrKeys{{Key1: new("a"), Key2: new("b"), Data: "one"}},
		PrimitivePointerListUniqueSet:    []*string{new("single")},
		PointerListUniqueMap:             []*Item{{Key: "single", Data: "one"}},
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
		SliceMapFieldWithPtrKey: []PtrKeyStruct{
			{Key: new("a"), Data: "first"},
			{Key: new("b"), Data: "second"},
			{Key: new("a"), Data: "third"},
		},
		SliceMapFieldWithMixedKeys: []ItemWithMixedKeys{
			{Key1: new("a"), Key2: "x", Data: "first"},
			{Key1: new("a"), Key2: "y", Data: "second"},
			{Key1: new("a"), Key2: "x", Data: "third"},
		},
		SliceMapFieldWithMultiplePtrKeys: []ItemWithMultiplePtrKeys{
			{Key1: new("a"), Key2: new("x"), Data: "first"},
			{Key1: new("a"), Key2: new("y"), Data: "second"},
			{Key1: new("a"), Key2: new("x"), Data: "third"},
		},
		PrimitivePointerListUniqueSet: []*string{new("aaa"), new("bbb"), new("aaa")},
		PointerListUniqueMap: []*Item{
			{Key: "key1", Data: "one"},
			{Key: "key2", Data: "two"},
			{Key: "key1", Data: "three"},
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Duplicate(field.NewPath("primitiveListUniqueSet").Index(3), nil),
		field.Duplicate(field.NewPath("primitiveListUniqueSet").Index(4), nil),
		field.Duplicate(field.NewPath("primitiveListUniqueSet").Index(5), nil),
		field.Duplicate(field.NewPath("sliceMapFieldWithMultipleKeys").Index(2), nil),
		field.Duplicate(field.NewPath("atomicListUniqueSet").Index(2), nil),
		field.Duplicate(field.NewPath("atomicListUniqueMap").Index(2), nil),
		field.Duplicate(field.NewPath("sliceMapFieldWithPtrKey").Index(2), nil),
		field.Duplicate(field.NewPath("sliceMapFieldWithMixedKeys").Index(2), nil),
		field.Duplicate(field.NewPath("sliceMapFieldWithMultiplePtrKeys").Index(2), nil),
		field.Duplicate(field.NewPath("primitivePointerListUniqueSet").Index(2), nil),
		field.Duplicate(field.NewPath("pointerListUniqueMap").Index(2), nil),
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

	// Test nil elements in pointer lists (should trigger Required errors)
	st.Value(&Struct{
		PrimitivePointerListUniqueSet: []*string{new("a"), nil, new("b")},
		PointerListUniqueMap:          []*Item{{Key: "key1"}, nil},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("primitivePointerListUniqueSet").Index(1), ""),
		field.Required(field.NewPath("pointerListUniqueMap").Index(1), ""),
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
		SliceMapFieldWithPtrKey: []PtrKeyStruct{
			{Key: new("a"), Data: "first"},
			{Key: new("b"), Data: "second"},
		},
		SliceMapFieldWithMixedKeys: []ItemWithMixedKeys{
			{Key1: new("a"), Key2: "x", Data: "first"},
			{Key1: new("a"), Key2: "y", Data: "second"},
		},
		SliceMapFieldWithMultiplePtrKeys: []ItemWithMultiplePtrKeys{
			{Key1: new("a"), Key2: new("x"), Data: "first"},
			{Key1: new("a"), Key2: new("y"), Data: "second"},
		},
		PrimitivePointerListUniqueSet: []*string{new("aaa"), new("bbb")},
		PointerListUniqueMap: []*Item{
			{Key: "key1", Data: "one"},
			{Key: "key2", Data: "two"},
		},
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
		SliceMapFieldWithPtrKey: []PtrKeyStruct{
			{Key: new("b"), Data: "second"},
			{Key: new("a"), Data: "first"},
		},
		SliceMapFieldWithMixedKeys: []ItemWithMixedKeys{
			{Key1: new("a"), Key2: "y", Data: "second"},
			{Key1: new("a"), Key2: "x", Data: "first"},
		},
		SliceMapFieldWithMultiplePtrKeys: []ItemWithMultiplePtrKeys{
			{Key1: new("a"), Key2: new("y"), Data: "second"},
			{Key1: new("a"), Key2: new("x"), Data: "first"},
		},
		PrimitivePointerListUniqueSet: []*string{new("bbb"), new("aaa")},
		PointerListUniqueMap: []*Item{
			{Key: "key2", Data: "two"},
			{Key: "key1", Data: "one"},
		},
	}

	// Test that reordering doesn't trigger validation errors
	st.Value(&struct1).OldValue(&struct2).ExpectValid()
	st.Value(&struct2).OldValue(&struct1).ExpectValid()

	// Test that the same data is considered valid regardless of order
	st.Value(&struct1).ExpectValid()
	st.Value(&struct2).ExpectValid()
}
