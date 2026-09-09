/*
Copyright 2024 The Kubernetes Authors.

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

package deep

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Every nilable field is nil, so each chain which traverses one stops
	// there. This asserts the full error set, not just the forced failures, to
	// prove that a nil intermediate produces no errors of its own: the only
	// nil hop which reports is the one whose chain marks it required, and it
	// reports once, without also running the deeper chain.
	st.Value(&Struct{}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailExact().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("structField", "structField", "stringField"), "", "forced failure: Struct.StructField.StructField 1").WithOrigin("validateFalse"),
		field.Invalid(field.NewPath("structField", "structField", "stringField"), "", "forced failure: Struct.StructField.StructField 2").WithOrigin("validateFalse"),
		field.Required(field.NewPath("requiredHopField", "ptrField"), ""),
	})

	st.Value(&Struct{
		StructField: OtherStruct{
			StructField: SmallStruct{StringField: "SF"},
			SliceField: []SmallStruct{{
				StringField: "SS1",
			}, {
				StringField: "SS2",
			}},
			MapField: map[string]SmallStruct{
				"a": {StringField: "SM1"},
				"b": {StringField: "SM2"},
			},
			PtrField: &SmallStruct{StringField: "PF"},
		},
		StructPtrField: &OtherStruct{
			StructField: SmallStruct{StringField: "SPF"},
			SliceField: []SmallStruct{{
				StringField: "SPS1",
			}, {
				StringField: "SPS2",
			}},
			MapField: map[string]SmallStruct{
				"b": {StringField: "SPM1"},
				"a": {StringField: "SPM2"},
			},
			PtrField: &SmallStruct{StringField: "PF"},
		},
		RequiredHopField: OtherStruct{
			PtrField: &SmallStruct{StringField: "RHF"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"structField.structField.stringField":      {"Struct.StructField.StructField 1", "Struct.StructField.StructField 2"},
		"structField.sliceField[0].stringField":    {"Struct.StructField.SliceField"},
		"structField.sliceField[1].stringField":    {"Struct.StructField.SliceField"},
		"structField.mapField[a].stringField":      {"Struct.StructField.MapField"},
		"structField.mapField[b].stringField":      {"Struct.StructField.MapField"},
		"structField.ptrField.stringField":         {"Struct.StructField.PtrField"},
		"structPtrField.structField.stringField":   {"Struct.StructPtrField.StructField 1", "Struct.StructPtrField.StructField 2"},
		"structPtrField.sliceField[0].stringField": {"Struct.StructPtrField.SliceField"},
		"structPtrField.sliceField[1].stringField": {"Struct.StructPtrField.SliceField"},
		"structPtrField.mapField[a].stringField":   {"Struct.StructPtrField.MapField"},
		"structPtrField.mapField[b].stringField":   {"Struct.StructPtrField.MapField"},
		"structPtrField.ptrField.stringField":      {"Struct.StructPtrField.PtrField"},
		"requiredHopField.ptrField.stringField":    {"Struct.RequiredHopField.PtrField"},
	})
}
