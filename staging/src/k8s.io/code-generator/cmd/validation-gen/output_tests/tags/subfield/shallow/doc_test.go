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

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&SetByServerStruct{
		SubfieldSetByServer: StructWithOptionalField{
			OptionalField: "xyz",
		},
		EmbeddedSetByServer: UnvalidatedStruct{
			StringField: "abc",
		},
		OpaqueSubfieldRequired: StructWithOptionalField{
			OptionalField: "def",
		},
		SubfieldRequired: StructWithOptionalField{
			OptionalField: "ghi",
		},
		SubfieldPtrSetByServer: &StructWithOptionalField{
			OptionalField: "xyz",
		},
		SubfieldPtrRequired: &StructWithOptionalField{
			OptionalField: "xyz",
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByField().ByType(), field.ErrorList{})

	st.Value(&SetByServerStruct{
		SubfieldSetByServer: StructWithOptionalField{
			OptionalField: "",
		},
		EmbeddedSetByServer: UnvalidatedStruct{
			StringField: "",
		},
		OpaqueSubfieldRequired: StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldRequired: StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldPtrSetByServer: &StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldPtrRequired: &StructWithOptionalField{
			OptionalField: "",
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByField().ByType(), field.ErrorList{
		field.Required(field.NewPath("subfieldSetByServer", "optionalField"), ""),
		field.Required(field.NewPath("embeddedSetByServer", "stringField"), ""),
		field.Required(field.NewPath("opaqueSubfieldRequired", "optionalField"), ""),
		field.Required(field.NewPath("subfieldRequired", "optionalField"), ""),
		field.Required(field.NewPath("subfieldPtrSetByServer", "optionalField"), ""),
		field.Required(field.NewPath("subfieldPtrRequired", "optionalField"), ""),
	})

	// Test pointer fields being nil
	st.Value(&SetByServerStruct{
		SubfieldSetByServer: StructWithOptionalField{
			OptionalField: "xyz",
		},
		EmbeddedSetByServer: UnvalidatedStruct{
			StringField: "abc",
		},
		OpaqueSubfieldRequired: StructWithOptionalField{
			OptionalField: "def",
		},
		SubfieldRequired: StructWithOptionalField{
			OptionalField: "ghi",
		},
		SubfieldPtrSetByServer: nil,
		SubfieldPtrRequired:    nil,
	}).ExpectMatches(field.ErrorMatcher{}.ByField().ByType(), field.ErrorList{})

	// Update tests for SetByServerStruct
	// Update from non-empty to empty should fail.
	st.Value(&SetByServerStruct{
		SubfieldSetByServer: StructWithOptionalField{
			OptionalField: "",
		},
		EmbeddedSetByServer: UnvalidatedStruct{
			StringField: "",
		},
		OpaqueSubfieldRequired: StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldRequired: StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldPtrSetByServer: &StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldPtrRequired: &StructWithOptionalField{
			OptionalField: "",
		},
	}).OldValue(&SetByServerStruct{
		SubfieldSetByServer: StructWithOptionalField{
			OptionalField: "xyz",
		},
		EmbeddedSetByServer: UnvalidatedStruct{
			StringField: "abc",
		},
		OpaqueSubfieldRequired: StructWithOptionalField{
			OptionalField: "def",
		},
		SubfieldRequired: StructWithOptionalField{
			OptionalField: "ghi",
		},
		SubfieldPtrSetByServer: &StructWithOptionalField{
			OptionalField: "xyz",
		},
		SubfieldPtrRequired: &StructWithOptionalField{
			OptionalField: "xyz",
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByField().ByType(), field.ErrorList{
		field.Required(field.NewPath("subfieldSetByServer", "optionalField"), ""),
		field.Required(field.NewPath("embeddedSetByServer", "stringField"), ""),
		field.Required(field.NewPath("opaqueSubfieldRequired", "optionalField"), ""),
		field.Required(field.NewPath("subfieldRequired", "optionalField"), ""),
		field.Required(field.NewPath("subfieldPtrSetByServer", "optionalField"), ""),
		field.Required(field.NewPath("subfieldPtrRequired", "optionalField"), ""),
	})

	// Update from empty to empty should be skipped (success).
	st.Value(&SetByServerStruct{
		SubfieldSetByServer: StructWithOptionalField{
			OptionalField: "",
		},
		EmbeddedSetByServer: UnvalidatedStruct{
			StringField: "",
		},
		OpaqueSubfieldRequired: StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldRequired: StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldPtrSetByServer: &StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldPtrRequired: &StructWithOptionalField{
			OptionalField: "",
		},
	}).OldValue(&SetByServerStruct{
		SubfieldSetByServer: StructWithOptionalField{
			OptionalField: "",
		},
		EmbeddedSetByServer: UnvalidatedStruct{
			StringField: "",
		},
		OpaqueSubfieldRequired: StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldRequired: StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldPtrSetByServer: &StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldPtrRequired: &StructWithOptionalField{
			OptionalField: "",
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByField().ByType(), field.ErrorList{})

	// Update from empty to non-empty should succeed.
	st.Value(&SetByServerStruct{
		SubfieldSetByServer: StructWithOptionalField{
			OptionalField: "xyz",
		},
		EmbeddedSetByServer: UnvalidatedStruct{
			StringField: "abc",
		},
		OpaqueSubfieldRequired: StructWithOptionalField{
			OptionalField: "def",
		},
		SubfieldRequired: StructWithOptionalField{
			OptionalField: "ghi",
		},
		SubfieldPtrSetByServer: &StructWithOptionalField{
			OptionalField: "xyz",
		},
		SubfieldPtrRequired: &StructWithOptionalField{
			OptionalField: "xyz",
		},
	}).OldValue(&SetByServerStruct{
		SubfieldSetByServer: StructWithOptionalField{
			OptionalField: "",
		},
		EmbeddedSetByServer: UnvalidatedStruct{
			StringField: "",
		},
		OpaqueSubfieldRequired: StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldRequired: StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldPtrSetByServer: &StructWithOptionalField{
			OptionalField: "",
		},
		SubfieldPtrRequired: &StructWithOptionalField{
			OptionalField: "",
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByField().ByType(), field.ErrorList{})

	st.Value(&Struct{
		StructField: OtherStruct{
			StringField:  "",
			PointerField: ptr.To(""),
			StructField:  UnvalidatedStruct{},
			SliceField:   []string{},
			MapField:     map[string]string{},
		},
		StructPtrField: &OtherStruct{
			StringField:  "",
			PointerField: ptr.To(""),
			StructField:  UnvalidatedStruct{},
			SliceField:   []string{},
			MapField:     map[string]string{},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"structField.stringField":     {"subfield Struct.StructField.StringField 1", "subfield Struct.StructField.StringField 2"},
		"structField.pointerField":    {"subfield Struct.StructField.PointerField"},
		"structField.structField":     {"subfield Struct.StructField.StructField"},
		"structField.sliceField":      {"subfield Struct.StructField.SliceField"},
		"structField.mapField":        {"subfield Struct.StructField.MapField"},
		"structPtrField.stringField":  {"subfield Struct.StructPtrField.StringField 1", "subfield Struct.StructPtrField.StringField 2"},
		"structPtrField.pointerField": {"subfield Struct.StructPtrField.PointerField"},
		"structPtrField.structField":  {"subfield Struct.StructPtrField.StructField"},
		"structPtrField.sliceField":   {"subfield Struct.StructPtrField.SliceField"},
		"structPtrField.mapField":     {"subfield Struct.StructPtrField.MapField"},
	})
}
