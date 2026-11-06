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

package eqchained

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		StructField:          InnerStruct{StringField: "expected-subfield"},
		StructPtrField:       &InnerStruct{StringField: "expected-subfield-ptr"},
		StringSliceField:     []string{"expected-slice", "expected-slice"},
		StringMapField:       map[string]string{"k1": "expected-map-val", "k2": "expected-map-val"},
		StringMapKeyField:    map[string]string{"expected-key": "v1"},
		ValidatedSliceField:  []string{"expected-typedef", "expected-typedef"},
		ValidatedStructField: ValidatedInnerStruct{StringField: "expected-typedef-struct"},
	}).ExpectValid()

	// Test empty collections and unset.
	st.Value(&Struct{
		StructField:          InnerStruct{StringField: "expected-subfield"},
		StructPtrField:       nil,
		StringSliceField:     []string{},
		StringMapField:       map[string]string{},
		StringMapKeyField:    map[string]string{},
		ValidatedSliceField:  []string{},
		ValidatedStructField: ValidatedInnerStruct{StringField: "expected-typedef-struct"},
	}).ExpectValid()

	// Test invalid values trigger all expected validation errors
	invalidStruct := &Struct{
		StructField:          InnerStruct{StringField: "other-subfield"},
		StructPtrField:       &InnerStruct{StringField: "other-subfield-ptr"},
		StringSliceField:     []string{"expected-slice", "other", "other"},
		StringMapField:       map[string]string{"a": "other", "b": "expected-map-val", "c": "other"},
		StringMapKeyField:    map[string]string{"expected-key": "v1", "other-key": "v2"},
		ValidatedSliceField:  []string{"expected-typedef", "other", "other"},
		ValidatedStructField: ValidatedInnerStruct{StringField: "other-typedef-struct"},
	}

	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.NotSupported(field.NewPath("structField", "stringField"), nil, []string{"expected-subfield"}),
		field.NotSupported(field.NewPath("structPtrField", "stringField"), nil, []string{"expected-subfield-ptr"}),
		field.NotSupported(field.NewPath("stringSliceField").Index(1), nil, []string{"expected-slice"}),
		field.NotSupported(field.NewPath("stringSliceField").Index(2), nil, []string{"expected-slice"}),
		field.NotSupported(field.NewPath("stringMapField").Key("a"), nil, []string{"expected-map-val"}),
		field.NotSupported(field.NewPath("stringMapField").Key("c"), nil, []string{"expected-map-val"}),
		field.NotSupported(field.NewPath("stringMapKeyField"), nil, []string{"expected-key"}),
		field.NotSupported(field.NewPath("validatedSliceField").Index(1), nil, []string{"expected-typedef"}),
		field.NotSupported(field.NewPath("validatedSliceField").Index(2), nil, []string{"expected-typedef"}),
		field.NotSupported(field.NewPath("validatedStructField", "stringField"), nil, []string{"expected-typedef-struct"}),
	})

	// Test validation ratcheting allows existing invalid values
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
