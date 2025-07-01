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

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test_StructSlice(t *testing.T) {
	st := localSchemeBuilder.Test(t)
	st.Value(&StructSlice{
		SliceField:                      []S{""},
		TypeDefSliceField:               MySlice{1},
		SliceStructField:                []DirectComparableStruct{{IntField: 1}},
		SliceNonComparableStructField:   []NonDirectComparableStruct{{IntPtrField: ptr.To(1)}},
		SliceStructWithKey:              []DirectComparableStructWithKey{{Key: "x", IntField: 1}},
		SliceNonComparableStructWithKey: []NonComparableStructWithKey{{Key: "x", IntPtrField: ptr.To(1)}},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("sliceField[0]"), "", ""),
		field.Invalid(field.NewPath("typedefSliceField[0]"), "", ""),
		field.Invalid(field.NewPath("sliceStructField[0]"), "", ""),
		field.Invalid(field.NewPath("sliceNonComparableStructField[0]"), "", ""),
		field.Invalid(field.NewPath("sliceStructWithKey[0]"), "", ""),
		field.Invalid(field.NewPath("sliceNonComparableStructWithKey[0]"), "", ""),
	})

	// No changes.
	st.Value(&StructSlice{
		SliceField:                    []S{""},
		TypeDefSliceField:             MySlice{1},
		SliceStructField:              []DirectComparableStruct{{IntField: 1}},
		SliceNonComparableStructField: []NonDirectComparableStruct{{IntPtrField: ptr.To(1)}},
	}).OldValue(&StructSlice{
		SliceField:                    []S{""},
		TypeDefSliceField:             MySlice{1},
		SliceStructField:              []DirectComparableStruct{{IntField: 1}},
		SliceNonComparableStructField: []NonDirectComparableStruct{{IntPtrField: ptr.To(1)}},
	}).ExpectValid()

	// New elements exist in old.
	st.Value(&StructSlice{
		SliceField:                    []S{""},
		TypeDefSliceField:             MySlice{1},
		SliceStructField:              []DirectComparableStruct{{IntField: 1}},
		SliceNonComparableStructField: []NonDirectComparableStruct{{IntPtrField: ptr.To(1)}},
	}).OldValue(&StructSlice{
		SliceField:                    []S{"", "x"},
		TypeDefSliceField:             MySlice{1, 2},
		SliceStructField:              []DirectComparableStruct{{IntField: 2}, {IntField: 1}},
		SliceNonComparableStructField: []NonDirectComparableStruct{{IntPtrField: ptr.To(2)}, {IntPtrField: ptr.To(1)}},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("sliceField[0]"), "", ""),
		field.Invalid(field.NewPath("typedefSliceField[0]"), "", ""),
		field.Invalid(field.NewPath("sliceStructField[0]"), "", ""),
		field.Invalid(field.NewPath("sliceNonComparableStructField[0]"), "", ""),
	})

	// No changes.
	st.Value(&StructSlice{
		SliceStructWithKey:              []DirectComparableStructWithKey{{Key: "x", IntField: 1}},
		SliceNonComparableStructWithKey: []NonComparableStructWithKey{{Key: "x", IntPtrField: ptr.To(1)}},
	}).OldValue(&StructSlice{
		SliceStructWithKey:              []DirectComparableStructWithKey{{Key: "x", IntField: 1}},
		SliceNonComparableStructWithKey: []NonComparableStructWithKey{{Key: "x", IntPtrField: ptr.To(1)}},
	}).ExpectValid()

	// Same data, different order.
	st.Value(&StructSlice{
		SliceStructWithKey:              []DirectComparableStructWithKey{{Key: "y", IntField: 1}, {Key: "x", IntField: 1}},
		SliceNonComparableStructWithKey: []NonComparableStructWithKey{{Key: "y", IntPtrField: ptr.To(2)}, {Key: "x", IntPtrField: ptr.To(1)}},
	}).OldValue(&StructSlice{
		SliceStructWithKey:              []DirectComparableStructWithKey{{Key: "x", IntField: 1}, {Key: "y", IntField: 1}},
		SliceNonComparableStructWithKey: []NonComparableStructWithKey{{Key: "x", IntPtrField: ptr.To(1)}, {Key: "y", IntPtrField: ptr.To(2)}},
	}).ExpectValid()

	// less data, elements exist in old.
	st.Value(&StructSlice{
		SliceNonComparableStructField:   []NonDirectComparableStruct{{IntPtrField: ptr.To(1)}},
		SliceNonComparableStructWithKey: []NonComparableStructWithKey{{Key: "x", IntPtrField: ptr.To(1)}},
	}).OldValue(&StructSlice{
		SliceNonComparableStructField:   []NonDirectComparableStruct{{IntPtrField: ptr.To(1)}, {IntPtrField: ptr.To(2)}},
		SliceNonComparableStructWithKey: []NonComparableStructWithKey{{Key: "x", IntPtrField: ptr.To(1)}, {Key: "y", IntPtrField: ptr.To(2)}},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("sliceNonComparableStructField[0]"), "", ""),
	})
}
