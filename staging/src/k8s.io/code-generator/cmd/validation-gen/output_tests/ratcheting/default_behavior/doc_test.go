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

package defaultbehavior

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test_StructPrimitive(t *testing.T) {
	st := localSchemeBuilder.Test(t)
	st.Value(&StructPrimitive{
		IntField:    1,
		IntPtrField: ptr.To(1),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), "", ""),
		field.Invalid(field.NewPath("intPtrField"), "", ""),
	})
	st.Value(&StructPrimitive{
		IntField:    1,
		IntPtrField: ptr.To(1),
	}).OldValue(&StructPrimitive{
		IntField:    1,
		IntPtrField: ptr.To(1), // Different pointers but value unchanged.
	}).ExpectValid()
}

func Test_StructSlice(t *testing.T) {
	st := localSchemeBuilder.Test(t)
	st.Value(&StructSlice{
		SliceField:        []S{""},
		TypeDefSliceField: MySlice{1},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("sliceField"), "", ""),
		field.Invalid(field.NewPath("sliceField[0]"), "", ""),
		field.Invalid(field.NewPath("typedefSliceField"), "", ""),
	})

	st.Value(&StructSlice{
		SliceField:        []S{""},
		TypeDefSliceField: MySlice{1},
	}).OldValue(&StructSlice{
		SliceField:        []S{""},
		TypeDefSliceField: MySlice{1},
	}).ExpectValid()
}

func Test_StructMap(t *testing.T) {
	st := localSchemeBuilder.Test(t)
	st.Value(&StructMap{
		MapKeyField:            map[S]string{S("k"): "v"},
		MapValueField:          map[string]S{"k": "v"},
		AliasMapKeyTypeField:   AliasMapKeyType{"k": "v"},
		AliasMapValueTypeField: AliasMapValueType{"k": "v"},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("mapKeyField"), "", ""),
		field.Invalid(field.NewPath("mapValueField"), "", ""),
		field.Invalid(field.NewPath("mapValueField[k]"), "", ""),
		field.Invalid(field.NewPath("aliasMapKeyTypeField"), "", ""),
		field.Invalid(field.NewPath("aliasMapValueTypeField"), "", ""),
		field.Invalid(field.NewPath("aliasMapValueTypeField[k]"), "", ""),
	})

	st.Value(&StructMap{
		MapKeyField:            map[S]string{S("k"): "v"},
		MapValueField:          map[string]S{"k": "v"},
		AliasMapKeyTypeField:   AliasMapKeyType{"k": "v"},
		AliasMapValueTypeField: AliasMapValueType{"k": "v"},
	}).OldValue(&StructMap{
		MapKeyField:            map[S]string{S("k"): "v"},
		MapValueField:          map[string]S{"k": "v"},
		AliasMapKeyTypeField:   AliasMapKeyType{"k": "v"},
		AliasMapValueTypeField: AliasMapValueType{"k": "v"},
	}).ExpectValid()
}

func Test_StructStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)
	st.Value(&StructStruct{
		DirectComparableStructField: DirectComparableStruct{
			IntField: 1,
		},
		NonDirectComparableStructField: NonDirectComparableStruct{
			IntPtrField: ptr.To(1),
		},
		DirectComparableStructPtr: &DirectComparableStruct{
			IntField: 1,
		},
		NonDirectComparableStructPtr: &NonDirectComparableStruct{
			IntPtrField: ptr.To(1),
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("directComparableStructField"), "", ""),
		field.Invalid(field.NewPath("nonDirectComparableStructField"), "", ""),
		field.Invalid(field.NewPath("directComparableStructField").Child("intField"), "", ""),
		field.Invalid(field.NewPath("nonDirectComparableStructField").Child("intPtrField"), "", ""),
		field.Invalid(field.NewPath("directComparableStructPtrField"), "", ""),
		field.Invalid(field.NewPath("nonDirectComparableStructPtrField"), "", ""),
		field.Invalid(field.NewPath("directComparableStructPtrField").Child("intField"), "", ""),
		field.Invalid(field.NewPath("nonDirectComparableStructPtrField").Child("intPtrField"), "", ""),
		field.Invalid(field.NewPath("DirectComparableStruct"), "", ""),
		field.Invalid(field.NewPath("NonDirectComparableStruct"), "", ""),
		field.Invalid(field.NewPath("DirectComparableStruct").Child("intField"), "", ""),
		field.Invalid(field.NewPath("NonDirectComparableStruct").Child("intPtrField"), "", ""),
	})

	st.Value(&StructStruct{
		DirectComparableStructField: DirectComparableStruct{
			IntField: 1,
		},
		NonDirectComparableStructField: NonDirectComparableStruct{
			IntPtrField: ptr.To(1),
		},
		DirectComparableStructPtr: &DirectComparableStruct{
			IntField: 1,
		},
		NonDirectComparableStructPtr: &NonDirectComparableStruct{
			IntPtrField: ptr.To(1),
		},
	}).OldValue(&StructStruct{
		DirectComparableStructField: DirectComparableStruct{
			IntField: 1,
		},
		NonDirectComparableStructField: NonDirectComparableStruct{
			IntPtrField: ptr.To(1),
		},
		DirectComparableStructPtr: &DirectComparableStruct{
			IntField: 1,
		},
		NonDirectComparableStructPtr: &NonDirectComparableStruct{
			IntPtrField: ptr.To(1),
		},
	}).ExpectValid()
}

func Test_StructEmbedded(t *testing.T) {
	st := localSchemeBuilder.Test(t)
	st.Value(&StructEmbedded{
		DirectComparableStruct: DirectComparableStruct{
			IntField: 1,
		},
		NonDirectComparableStruct: NonDirectComparableStruct{
			IntPtrField: ptr.To(1),
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("directComparableStruct"), "", ""),
		field.Invalid(field.NewPath("nonDirectComparableStruct"), "", ""),
		field.Invalid(field.NewPath("directComparableStruct").Child("intField"), "", ""),
		field.Invalid(field.NewPath("nonDirectComparableStruct").Child("intPtrField"), "", ""),
		field.Invalid(field.NewPath("nestedDirectComparableStructField"), "", ""),
		field.Invalid(field.NewPath("nestedDirectComparableStructField").Child("directComparableStructField"), "", ""),
		field.Invalid(field.NewPath("nestedDirectComparableStructField").Child("directComparableStructField").Child("intField"), "", ""),
		field.Invalid(field.NewPath("nestedNonDirectComparableStructField"), "", ""),
		field.Invalid(field.NewPath("nestedNonDirectComparableStructField").Child("nonDirectComparableStructField"), "", ""),
		field.Invalid(field.NewPath("nestedNonDirectComparableStructField").Child("nonDirectComparableStructField").Child("intPtrField"), "", ""),
	})

	st.Value(&StructEmbedded{
		DirectComparableStruct: DirectComparableStruct{
			IntField: 1,
		},
		NonDirectComparableStruct: NonDirectComparableStruct{
			IntPtrField: ptr.To(1),
		},
		NestedDirectComparableStructField: NestedDirectComparableStruct{
			DirectComparableStructField: DirectComparableStruct{
				IntField: 1,
			},
		},
		NestedNonDirectComparableStructField: NestedNonDirectComparableStruct{
			NonDirectComparableStructField: NonDirectComparableStruct{
				IntPtrField: ptr.To(1),
			},
		},
	}).OldValue(&StructEmbedded{
		DirectComparableStruct: DirectComparableStruct{
			IntField: 1,
		},
		NonDirectComparableStruct: NonDirectComparableStruct{
			IntPtrField: ptr.To(1),
		},
		NestedDirectComparableStructField: NestedDirectComparableStruct{
			DirectComparableStructField: DirectComparableStruct{
				IntField: 1,
			},
		},
		NestedNonDirectComparableStructField: NestedNonDirectComparableStruct{
			NonDirectComparableStructField: NonDirectComparableStruct{
				IntPtrField: ptr.To(1),
			},
		},
	}).ExpectValid()
}
