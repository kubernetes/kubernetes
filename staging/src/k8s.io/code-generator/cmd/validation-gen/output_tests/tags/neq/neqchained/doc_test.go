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

package neqchained

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		StructField:          InnerStruct{StringField: "allowed"},
		StructPtrField:       &InnerStruct{StringField: "valid"},
		StringSliceField:     []string{"allowed", "valid"},
		StringMapField:       map[string]string{"k1": "allowed", "k2": "valid"},
		StringMapKeyField:    map[string]string{"allowed-key": "v1", "valid-key": "v2"},
		ValidatedSliceField:  []string{"allowed", "valid"},
		ValidatedStructField: ValidatedInnerStruct{StringField: "allowed"},
	}).ExpectValid()

	// Test empty collections and unset.
	st.Value(&Struct{
		StructField:          InnerStruct{},
		StructPtrField:       &InnerStruct{StringField: "valid"},
		StringSliceField:     []string{},
		StringMapField:       map[string]string{},
		StringMapKeyField:    map[string]string{},
		ValidatedSliceField:  []string{},
		ValidatedStructField: ValidatedInnerStruct{StringField: "allowed"},
	}).ExpectValid()

	// Test invalid values trigger all expected validation errors
	invalidStruct := &Struct{
		StructField:          InnerStruct{StringField: "disallowed-subfield"},
		StructPtrField:       &InnerStruct{StringField: "disallowed-subfield-ptr"},
		StringSliceField:     []string{"valid", "disallowed-slice", "disallowed-slice"},
		StringMapField:       map[string]string{"a": "disallowed-map-val", "b": "valid", "c": "disallowed-map-val"},
		StringMapKeyField:    map[string]string{"disallowed-key": "value", "allowed": "ok"},
		ValidatedSliceField:  []string{"valid", "disallowed-typedef", "disallowed-typedef"},
		ValidatedStructField: ValidatedInnerStruct{StringField: "disallowed-typedef-struct"},
	}

	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("structField", "stringField"), "disallowed-subfield", content.NEQError("disallowed-subfield")).WithOrigin("neq"),
		field.Invalid(field.NewPath("structPtrField", "stringField"), "disallowed-subfield-ptr", content.NEQError("disallowed-subfield-ptr")).WithOrigin("neq"),
		field.Invalid(field.NewPath("stringSliceField").Index(1), "disallowed-slice", content.NEQError("disallowed-slice")).WithOrigin("neq"),
		field.Invalid(field.NewPath("stringSliceField").Index(2), "disallowed-slice", content.NEQError("disallowed-slice")).WithOrigin("neq"),
		field.Invalid(field.NewPath("stringMapField").Key("a"), "disallowed-map-val", content.NEQError("disallowed-map-val")).WithOrigin("neq"),
		field.Invalid(field.NewPath("stringMapField").Key("c"), "disallowed-map-val", content.NEQError("disallowed-map-val")).WithOrigin("neq"),
		field.Invalid(field.NewPath("stringMapKeyField"), "disallowed-key", content.NEQError("disallowed-key")).WithOrigin("neq"),
		field.Invalid(field.NewPath("validatedSliceField").Index(1), "disallowed-typedef", content.NEQError("disallowed-typedef")).WithOrigin("neq"),
		field.Invalid(field.NewPath("validatedSliceField").Index(2), "disallowed-typedef", content.NEQError("disallowed-typedef")).WithOrigin("neq"),
		field.Invalid(field.NewPath("validatedStructField", "stringField"), "disallowed-typedef-struct", content.NEQError("disallowed-typedef-struct")).WithOrigin("neq"),
	})

	// Test validation ratcheting allows existing invalid values
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
