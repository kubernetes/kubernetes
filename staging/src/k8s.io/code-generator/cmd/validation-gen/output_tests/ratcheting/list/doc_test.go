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

package list

import (
	"testing"

	"k8s.io/utils/ptr"
)

func Test_StructSlice(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	invalidStructSlice := &StructSlice{
		AtomicSliceStringField:        []StringType{""},
		AtomicSliceTypeField:          IntSliceType{1},
		AtomicSliceComparableField:    []ComparableStruct{{IntField: 1}},
		AtomicSliceNonComparableField: []NonComparableStruct{{IntPtrField: ptr.To(1)}},
		SetSliceComparableField:       []ComparableStruct{{IntField: 1}},
		SetSliceNonComparableField:    []NonComparableStruct{{IntPtrField: ptr.To(1)}},
		MapSliceComparableField:       []ComparableStructWithKey{{Key: "x", IntField: 1}},
		MapSliceNonComparableField:    []NonComparableStructWithKey{{Key: "x", IntPtrField: ptr.To(1)}},
		MapSlicePtrKeyField:           []PtrKeyStruct{{Key: ptr.To("x"), Data: "y"}},
		MapSliceMixedKeyField:         []MixedKeyStruct{{Key1: ptr.To("x"), Key2: "y", Data: "z"}},
	}
	st.Value(invalidStructSlice).ExpectValidateFalseByPath(map[string][]string{
		"atomicSliceStringField[0]":        {"field AtomicSliceStringField[*]"},
		"atomicSliceTypeField[0]":          {"field AtomicSliceTypeField[*]"},
		"atomicSliceComparableField[0]":    {"field AtomicSliceComparableField[*]"},
		"atomicSliceNonComparableField[0]": {"field AtomicSliceNonComparableField[*]", "type NonComparableStruct"},
		"setSliceComparableField[0]":       {"field SetSliceComparableField[*]"},
		"setSliceNonComparableField[0]":    {"field SetSliceNonComparableField[*]", "type NonComparableStruct"},
		"mapSliceComparableField[0]":       {"field MapSliceComparableField[*]"},
		"mapSliceNonComparableField[0]":    {"field MapSliceNonComparableField[*]", "type NonComparableStructWithKey"},
		"mapSlicePtrKeyField[0]":           {"field MapSlicePtrKeyField[*]", "type PtrKeyStruct"},
		"mapSliceMixedKeyField[0]":         {"field MapSliceMixedKeyField[*]", "type MixedKeyStruct"},
	})

	// No changes.
	st.Value(&invalidStructSlice).OldValue(&invalidStructSlice).ExpectValid()

	//  Removed elements - errors on atomic.
	st.Value(&StructSlice{
		AtomicSliceStringField:        []StringType{""},
		AtomicSliceTypeField:          IntSliceType{1},
		AtomicSliceComparableField:    []ComparableStruct{{IntField: 1}},
		AtomicSliceNonComparableField: []NonComparableStruct{{IntPtrField: ptr.To(1)}},
		SetSliceComparableField:       []ComparableStruct{{IntField: 1}},
		SetSliceNonComparableField:    []NonComparableStruct{{IntPtrField: ptr.To(1)}},
		MapSliceComparableField:       []ComparableStructWithKey{{Key: "x", IntField: 1}},
		MapSliceNonComparableField:    []NonComparableStructWithKey{{Key: "x", IntPtrField: ptr.To(1)}},
		MapSlicePtrKeyField:           []PtrKeyStruct{{Key: ptr.To("x"), Data: "y"}},
		MapSliceMixedKeyField:         []MixedKeyStruct{{Key1: ptr.To("x"), Key2: "y", Data: "z"}},
	}).OldValue(&StructSlice{
		AtomicSliceStringField:        []StringType{"", "x"},
		AtomicSliceTypeField:          IntSliceType{1, 2},
		AtomicSliceComparableField:    []ComparableStruct{{IntField: 2}, {IntField: 1}},
		AtomicSliceNonComparableField: []NonComparableStruct{{IntPtrField: ptr.To(2)}, {IntPtrField: ptr.To(1)}},
		SetSliceComparableField:       []ComparableStruct{{IntField: 2}, {IntField: 1}},
		SetSliceNonComparableField:    []NonComparableStruct{{IntPtrField: ptr.To(2)}, {IntPtrField: ptr.To(1)}},
		MapSliceComparableField:       []ComparableStructWithKey{{Key: "y", IntField: 2}, {Key: "x", IntField: 1}},
		MapSliceNonComparableField:    []NonComparableStructWithKey{{Key: "y", IntPtrField: ptr.To(2)}, {Key: "x", IntPtrField: ptr.To(1)}},
		MapSlicePtrKeyField:           []PtrKeyStruct{{Key: ptr.To("a"), Data: "b"}, {Key: ptr.To("x"), Data: "y"}},
		MapSliceMixedKeyField:         []MixedKeyStruct{{Key1: ptr.To("a"), Key2: "b", Data: "c"}, {Key1: ptr.To("x"), Key2: "y", Data: "z"}},
	}).ExpectValidateFalseByPath(map[string][]string{"atomicSliceStringField[0]": {"field AtomicSliceStringField[*]"},
		"atomicSliceTypeField[0]":          {"field AtomicSliceTypeField[*]"},
		"atomicSliceComparableField[0]":    {"field AtomicSliceComparableField[*]"},
		"atomicSliceNonComparableField[0]": {"field AtomicSliceNonComparableField[*]", "type NonComparableStruct"},
	})

	// Same data, different order.
	st.Value(&StructSlice{
		SetSliceComparableField:    []ComparableStruct{{IntField: 2}, {IntField: 1}},
		SetSliceNonComparableField: []NonComparableStruct{{IntPtrField: ptr.To(2)}, {IntPtrField: ptr.To(1)}},
		MapSliceComparableField:    []ComparableStructWithKey{{Key: "y", IntField: 2}, {Key: "x", IntField: 1}},
		MapSliceNonComparableField: []NonComparableStructWithKey{{Key: "y", IntPtrField: ptr.To(2)}, {Key: "x", IntPtrField: ptr.To(1)}},
		MapSlicePtrKeyField:        []PtrKeyStruct{{Key: ptr.To("b"), Data: "2"}, {Key: ptr.To("a"), Data: "1"}},
		MapSliceMixedKeyField:      []MixedKeyStruct{{Key1: ptr.To("b"), Key2: "2", Data: "B"}, {Key1: ptr.To("a"), Key2: "1", Data: "A"}},
	}).OldValue(&StructSlice{
		SetSliceComparableField:    []ComparableStruct{{IntField: 1}, {IntField: 2}},
		SetSliceNonComparableField: []NonComparableStruct{{IntPtrField: ptr.To(1)}, {IntPtrField: ptr.To(2)}},
		MapSliceComparableField:    []ComparableStructWithKey{{Key: "x", IntField: 1}, {Key: "y", IntField: 2}},
		MapSliceNonComparableField: []NonComparableStructWithKey{{Key: "x", IntPtrField: ptr.To(1)}, {Key: "y", IntPtrField: ptr.To(2)}},
		MapSlicePtrKeyField:        []PtrKeyStruct{{Key: ptr.To("a"), Data: "1"}, {Key: ptr.To("b"), Data: "2"}},
		MapSliceMixedKeyField:      []MixedKeyStruct{{Key1: ptr.To("a"), Key2: "1", Data: "A"}, {Key1: ptr.To("b"), Key2: "2", Data: "B"}},
	}).ExpectValid()
}

func Test_Items(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&ItemList{
		Items: []Item{
			{Key: "valid2"},
		},
	}).OldValue(&ItemList{
		Items: []Item{
			{Key: "valid1"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"items[0].data": {"field Data"},
	})
}
