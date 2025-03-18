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

package opaque

import (
	"testing"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		StructPtrField:                 &OtherStruct{},
		OpaqueStructPtrField:           &OtherStruct{},
		SliceOfStructField:             []OtherStruct{{}, {}},
		SliceOfOpaqueStructField:       []OtherStruct{{}, {}},
		ListMapOfStructField:           []OtherStruct{{"foo"}, {"bar"}},
		ListMapOfOpaqueStructField:     []OtherStruct{{"foo"}, {"bar"}},
		MapOfStringToStructField:       map[OtherString]OtherStruct{"a": {"foo"}, "b": {"bar"}},
		MapOfStringToOpaqueStructField: map[OtherString]OtherStruct{"a": {"foo"}, "b": {"bar"}},
	}).ExpectValidateFalseByPath(map[string][]string{
		"structField":                         {"field Struct.StructField", "type OtherStruct"},
		"structPtrField":                      {"field Struct.StructPtrField", "type OtherStruct"},
		"structField.stringField":             {"field OtherStruct.StringField"},
		"structPtrField.stringField":          {"field OtherStruct.StringField"},
		"opaqueStructField":                   {"field Struct.OpaqueStructField"},
		"opaqueStructPtrField":                {"field Struct.OpaqueStructPtrField"},
		"sliceOfStructField":                  {"field Struct.SliceOfStructField"},
		"sliceOfStructField[0]":               {"field Struct.SliceOfStructField vals", "type OtherStruct"},
		"sliceOfStructField[0].stringField":   {"field OtherStruct.StringField"},
		"sliceOfStructField[1]":               {"field Struct.SliceOfStructField vals", "type OtherStruct"},
		"sliceOfStructField[1].stringField":   {"field OtherStruct.StringField"},
		"sliceOfOpaqueStructField":            {"field Struct.SliceOfOpaqueStructField"},
		"sliceOfOpaqueStructField[0]":         {"field Struct.SliceOfOpaqueStructField vals"},
		"sliceOfOpaqueStructField[1]":         {"field Struct.SliceOfOpaqueStructField vals"},
		"listMapOfStructField":                {"field Struct.ListMapOfStructField"},
		"listMapOfStructField[0]":             {"field Struct.ListMapOfStructField vals", "type OtherStruct"},
		"listMapOfStructField[0].stringField": {"field OtherStruct.StringField"},
		"listMapOfStructField[1]":             {"field Struct.ListMapOfStructField vals", "type OtherStruct"},
		"listMapOfStructField[1].stringField": {"field OtherStruct.StringField"},
		"listMapOfOpaqueStructField":          {"field Struct.ListMapOfOpaqueStructField"},
		"listMapOfOpaqueStructField[0]":       {"field Struct.ListMapOfOpaqueStructField vals"},
		"listMapOfOpaqueStructField[1]":       {"field Struct.ListMapOfOpaqueStructField vals"},
		"mapOfStringToStructField": {
			"field Struct.MapOfStringToStructField",
			"field Struct.MapOfStringToStructField keys",
			"field Struct.MapOfStringToStructField keys",
			"type OtherString",
			"type OtherString",
		},
		"mapOfStringToStructField[a]":             {"field Struct.MapOfStringToStructField vals", "type OtherStruct"},
		"mapOfStringToStructField[a].stringField": {"field OtherStruct.StringField"},
		"mapOfStringToStructField[b]":             {"field Struct.MapOfStringToStructField vals", "type OtherStruct"},
		"mapOfStringToStructField[b].stringField": {"field OtherStruct.StringField"},
		"mapOfStringToOpaqueStructField": {
			"field Struct.MapOfStringToOpaqueStructField",
			"field Struct.MapOfStringToOpaqueStructField keys",
			"field Struct.MapOfStringToOpaqueStructField keys",
		},
		"mapOfStringToOpaqueStructField[a]": {"field Struct.MapOfStringToOpaqueStructField vals"},
		"mapOfStringToOpaqueStructField[b]": {"field Struct.MapOfStringToOpaqueStructField vals"},
	})
}
