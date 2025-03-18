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
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

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
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"structField.structField.stringField":      {"StructField.StructField"},
		"structField.sliceField[0].stringField":    {"StructField.SliceField"},
		"structField.sliceField[1].stringField":    {"StructField.SliceField"},
		"structField.mapField[a].stringField":      {"StructField.MapField"},
		"structField.mapField[b].stringField":      {"StructField.MapField"},
		"structPtrField.structField.stringField":   {"StructPtrField.StructField"},
		"structPtrField.sliceField[0].stringField": {"StructPtrField.SliceField"},
		"structPtrField.sliceField[1].stringField": {"StructPtrField.SliceField"},
		"structPtrField.mapField[a].stringField":   {"StructPtrField.MapField"},
		"structPtrField.mapField[b].stringField":   {"StructPtrField.MapField"},
	})
}
