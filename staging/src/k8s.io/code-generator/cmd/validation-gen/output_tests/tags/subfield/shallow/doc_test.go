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

package shallow

import (
	"testing"

	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		StructField: OtherStruct{
			StringField:  "",
			PointerField: ptr.To(""),
			StructField:  SmallStruct{},
			SliceField:   []string{},
			MapField:     map[string]string{},
		},
		StructPtrField: &OtherStruct{
			StringField:  "",
			PointerField: ptr.To(""),
			StructField:  SmallStruct{},
			SliceField:   []string{},
			MapField:     map[string]string{},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"structField.stringField":     {"subfield Struct.StructField.StringField"},
		"structPtrField.stringField":  {"subfield Struct.StructPtrField.StringField"},
		"structField.pointerField":    {"subfield Struct.StructField.PointerField"},
		"structPtrField.pointerField": {"subfield Struct.StructPtrField.PointerField"},
		"structField.structField":     {"subfield Struct.StructField.StructField"},
		"structPtrField.structField":  {"subfield Struct.StructPtrField.StructField"},
		"structField.sliceField":      {"subfield Struct.StructField.SliceField"},
		"structPtrField.sliceField":   {"subfield Struct.StructPtrField.SliceField"},
		"structField.mapField":        {"subfield Struct.StructField.MapField"},
		"structPtrField.mapField":     {"subfield Struct.StructPtrField.MapField"},
	})
}
