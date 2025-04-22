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

	field "k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{}).ExpectValid()

	st.Value(&Struct{
		SliceStringField: []string{"aaa", "bbb"},
		SliceIntField:    []int{1, 2},
		SliceComparableField: []ComparableStruct{
			ComparableStruct{"aaa"},
			ComparableStruct{"bbb"},
		},
		SliceNonComparableField: []NonComparableStruct{
			NonComparableStruct{[]string{"aaa", "bbb"}},
			NonComparableStruct{[]string{"bbb", "aaa"}},
		},
	}).ExpectValid()

	st.Value(&Struct{
		SliceStringField: []string{"aaa", "bbb", "ccc", "ccc", "bbb", "aaa"},
		SliceIntField:    []int{1, 2, 3, 3, 2, 1},
		SliceComparableField: []ComparableStruct{
			ComparableStruct{"aaa"},
			ComparableStruct{"bbb"},
			ComparableStruct{"ccc"},
			ComparableStruct{"ccc"},
			ComparableStruct{"bbb"},
			ComparableStruct{"aaa"},
		},
		SliceNonComparableField: []NonComparableStruct{
			NonComparableStruct{[]string{"aaa", "111"}},
			NonComparableStruct{[]string{"bbb", "222"}},
			NonComparableStruct{[]string{"ccc", "333"}},
			NonComparableStruct{[]string{"ccc", "333"}},
			NonComparableStruct{[]string{"bbb", "222"}},
			NonComparableStruct{[]string{"aaa", "111"}},
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
	)
}

func TestSetCorrelation(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	structNew := ImmutableStruct{SliceComparableField: []ComparableStruct{{"aaa"}, {"bbb"}}}
	structOld := ImmutableStruct{SliceComparableField: []ComparableStruct{{"bbb"}, {"aaa"}}}
	st.Value(&structOld).OldValue(&structNew).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("sliceComparableField").Index(0), ""),
		field.Forbidden(field.NewPath("sliceComparableField").Index(1), ""),
	})

	structNew = ImmutableStruct{SliceSetComparableField: []ComparableStruct{{"aaa"}, {"bbb"}}}
	structOld = ImmutableStruct{SliceSetComparableField: []ComparableStruct{{"bbb"}, {"aaa"}}}
	st.Value(&structOld).OldValue(&structNew).ExpectValid()

	structNew = ImmutableStruct{SliceNonComparableField: []NonComparableStruct{{[]string{"aaa"}}, {[]string{"bbb"}}}}
	structOld = ImmutableStruct{SliceNonComparableField: []NonComparableStruct{{[]string{"bbb"}}, {[]string{"aaa"}}}}
	st.Value(&structOld).OldValue(&structNew).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("sliceNonComparableField").Index(0), ""),
		field.Forbidden(field.NewPath("sliceNonComparableField").Index(1), ""),
	})

	structNew = ImmutableStruct{SliceSetNonComparableField: []NonComparableStruct{{[]string{"aaa"}}, {[]string{"bbb"}}}}
	structOld = ImmutableStruct{SliceSetNonComparableField: []NonComparableStruct{{[]string{"bbb"}}, {[]string{"aaa"}}}}
	st.Value(&structOld).OldValue(&structNew).ExpectValid()
}
