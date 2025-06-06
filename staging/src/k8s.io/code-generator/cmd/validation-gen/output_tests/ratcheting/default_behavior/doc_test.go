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
	mkTest := func() *StructPrimitive {
		return &StructPrimitive{
			IntField:    1,
			IntPtrField: ptr.To(1), // Different pointers each call, but same value.
		}
	}

	st := localSchemeBuilder.Test(t)
	st.Value(mkTest()).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("intField"), "", ""),
		field.Invalid(field.NewPath("intPtrField"), "", ""),
	})
	st.Value(mkTest()).OldValue(mkTest()).ExpectValid()
}

func Test_StructSlice(t *testing.T) {
	mkTest := func() *StructSlice {
		return &StructSlice{
			SliceField:        []S{""},
			TypeDefSliceField: MySlice{1},
		}
	}

	st := localSchemeBuilder.Test(t)
	st.Value(mkTest()).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("sliceField"), "", ""),
		field.Invalid(field.NewPath("sliceField[0]"), "", ""),
		field.Invalid(field.NewPath("typedefSliceField"), "", ""),
	})

	st.Value(mkTest()).OldValue(mkTest()).ExpectValid()
}

func Test_StructMap(t *testing.T) {
	mkTest := func() *StructMap {
		return &StructMap{
			MapKeyField:            map[S]string{S("k"): "v"},
			MapValueField:          map[string]S{"k": "v"},
			AliasMapKeyTypeField:   AliasMapKeyType{"k": "v"},
			AliasMapValueTypeField: AliasMapValueType{"k": "v"},
		}
	}

	st := localSchemeBuilder.Test(t)
	st.Value(mkTest()).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("mapKeyField"), "", ""),
		field.Invalid(field.NewPath("mapValueField"), "", ""),
		field.Invalid(field.NewPath("mapValueField[k]"), "", ""),
		field.Invalid(field.NewPath("aliasMapKeyTypeField"), "", ""),
		field.Invalid(field.NewPath("aliasMapValueTypeField"), "", ""),
		field.Invalid(field.NewPath("aliasMapValueTypeField[k]"), "", ""),
	})

	st.Value(mkTest()).OldValue(mkTest()).ExpectValid()
}

func Test_StructStruct(t *testing.T) {
	mkTest := func() *StructStruct {
		return &StructStruct{
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
		}
	}

	st := localSchemeBuilder.Test(t)
	st.Value(mkTest()).ExpectValidateFalseByPath(map[string][]string{
		"directComparableStructField":                   {"field directComparableStructField", "type DirectComparableStruct"},
		"directComparableStructField.intField":          {"field intField"},
		"nonDirectComparableStructField":                {"field nonDirectComparableStructField", "type NonDirectComparableStruct"},
		"nonDirectComparableStructField.intPtrField":    {"field intPtrField"},
		"directComparableStructPtrField":                {"field directComparableStructPtrField", "type DirectComparableStruct"},
		"directComparableStructPtrField.intField":       {"field intField"},
		"nonDirectComparableStructPtrField":             {"field nonDirectComparableStructPtrField", "type NonDirectComparableStruct"},
		"nonDirectComparableStructPtrField.intPtrField": {"field intPtrField"},
		"DirectComparableStruct":                        {"field DirectComparableStruct", "type DirectComparableStruct"},
		"DirectComparableStruct.intField":               {"field intField"},
		"NonDirectComparableStruct":                     {"field NonDirectComparableStruct", "type NonDirectComparableStruct"},
		"NonDirectComparableStruct.intPtrField":         {"field intPtrField"},
	})

	st.Value(mkTest()).OldValue(mkTest()).ExpectValid()
}

func Test_StructEmbedded(t *testing.T) {
	mkTest := func() *StructEmbedded {
		return &StructEmbedded{
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
		}
	}

	st := localSchemeBuilder.Test(t)
	st.Value(mkTest()).ExpectValidateFalseByPath(map[string][]string{
		"directComparableStruct": {
			"field DirectComparableStruct", "type DirectComparableStruct",
		},
		"directComparableStruct.intField": {
			"field intField",
		},
		"nonDirectComparableStruct": {
			"field NonDirectComparableStruct", "type NonDirectComparableStruct",
		},
		"nonDirectComparableStruct.intPtrField": {
			"field intPtrField",
		},
		"nestedDirectComparableStructField": {
			"field NestedDirectComparableStructField", "type NestedDirectComparableStruct",
		},
		"nestedDirectComparableStructField.directComparableStructField": {
			"field directComparableStructField", "type DirectComparableStruct",
		},
		"nestedDirectComparableStructField.directComparableStructField.intField": {
			"field intField",
		},
		"nestedNonDirectComparableStructField": {
			"field NestedNonDirectComparableStructField", "type NestedNonDirectComparableStruct",
		},
		"nestedNonDirectComparableStructField.nonDirectComparableStructField": {
			"field nonDirectComparableStructField", "type NonDirectComparableStruct",
		},
		"nestedNonDirectComparableStructField.nonDirectComparableStructField.intPtrField": {
			"field intPtrField",
		},
	})

	st.Value(mkTest()).OldValue(mkTest()).ExpectValid()
}
