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

package listset

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{}).ExpectValid()

	st.Value(&Struct{
		SliceStringField:     []string{"aaa", "bbb"},
		SliceIntField:        []int{1, 2},
		SliceComparableField: []ComparableStruct{{"aaa"}, {"bbb"}},
		SliceNonComparableField: []NonComparableStruct{
			{[]string{"aaa", "bbb"}},
			{[]string{"bbb", "aaa"}},
		},
	}).ExpectValid()

	ptrS1 := ptr.To("same value")
	ptrS2 := ptr.To("same value")
	st.Value(&Struct{
		SliceStringField:     []string{"aaa", "bbb", "ccc", "ccc", "bbb", "aaa"},
		SliceIntField:        []int{1, 2, 3, 3, 2, 1},
		SliceComparableField: []ComparableStruct{{"aaa"}, {"bbb"}, {"ccc"}, {"ccc"}, {"bbb"}, {"aaa"}},
		SliceNonComparableField: []NonComparableStruct{
			{[]string{"aaa", "111"}},
			{[]string{"bbb", "222"}},
			{[]string{"ccc", "333"}},
			{[]string{"ccc", "333"}},
			{[]string{"bbb", "222"}},
			{[]string{"aaa", "111"}},
		},
		SliceFalselyComparableField: []FalselyComparableStruct{
			{StringPtrField: ptrS1},
			{StringPtrField: ptrS2},
		},
	}).ExpectInvalid(
		field.Duplicate(field.NewPath("sliceStringField").Index(3), "ccc"),
		field.Duplicate(field.NewPath("sliceStringField").Index(4), "bbb"),
		field.Duplicate(field.NewPath("sliceStringField").Index(5), "aaa"),
		field.Duplicate(field.NewPath("sliceIntField").Index(3), 3),
		field.Duplicate(field.NewPath("sliceIntField").Index(4), 2),
		field.Duplicate(field.NewPath("sliceIntField").Index(5), 1),
		field.Duplicate(field.NewPath("sliceComparableField").Index(3), ComparableStruct{"ccc"}),
		field.Duplicate(field.NewPath("sliceComparableField").Index(4), ComparableStruct{"bbb"}),
		field.Duplicate(field.NewPath("sliceComparableField").Index(5), ComparableStruct{"aaa"}),
		field.Duplicate(field.NewPath("sliceNonComparableField").Index(3), NonComparableStruct{[]string{"ccc", "333"}}),
		field.Duplicate(field.NewPath("sliceNonComparableField").Index(4), NonComparableStruct{[]string{"bbb", "222"}}),
		field.Duplicate(field.NewPath("sliceNonComparableField").Index(5), NonComparableStruct{[]string{"aaa", "111"}}),
		field.Duplicate(field.NewPath("sliceFalselyComparableField").Index(1), FalselyComparableStruct{StringPtrField: ptrS2}),
	)
}

func TestSetCorrelation(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	structNew := ImmutableStruct{SliceSetComparableField: []ComparableStruct{{"aaa"}, {"bbb"}}}
	structOld := ImmutableStruct{SliceSetComparableField: []ComparableStruct{{"bbb"}, {"aaa"}}}
	st.Value(&structNew).OldValue(&structOld).ExpectValid()

	structNew = ImmutableStruct{SliceSetNonComparableField: []NonComparableStruct{{[]string{"aaa"}}, {[]string{"bbb"}}}}
	structOld = ImmutableStruct{SliceSetNonComparableField: []NonComparableStruct{{[]string{"bbb"}}, {[]string{"aaa"}}}}
	st.Value(&structNew).OldValue(&structOld).ExpectValid()

	structNew = ImmutableStruct{SliceSetPrimitiveField: []int{1, 2}}
	structOld = ImmutableStruct{SliceSetPrimitiveField: []int{2, 1}}
	st.Value(&structNew).OldValue(&structOld).ExpectValid()

	structNew = ImmutableStruct{SliceSetFalselyComparableField: []FalselyComparableStruct{{StringPtrField: ptr.To("same value")}}}
	structOld = ImmutableStruct{SliceSetFalselyComparableField: []FalselyComparableStruct{{StringPtrField: ptr.To("same value")}}}
	st.Value(&structNew).OldValue(&structOld).ExpectValid()
}
