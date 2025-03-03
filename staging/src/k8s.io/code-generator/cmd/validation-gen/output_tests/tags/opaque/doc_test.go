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
		StructPtrField:                    &OtherStruct{},
		OpaqueStructPtrField:              &OtherStruct{},
		SliceOfStructField:                []OtherStruct{{}, {}},
		SliceOfStructPtrField:             []*OtherStruct{{}, {}},
		SliceOfOpaqueStructField:          []OtherStruct{{}, {}},
		SliceOfOpaqueStructPtrField:       []*OtherStruct{{}, {}},
		ListMapOfStructField:              []OtherStruct{{"foo"}, {"bar"}},
		ListMapOfStructPtrField:           []*OtherStruct{{"foo"}, {"bar"}},
		ListMapOfOpaqueStructField:        []OtherStruct{{"foo"}, {"bar"}},
		ListMapOfOpaqueStructPtrField:     []*OtherStruct{{"foo"}, {"bar"}},
		MapOfStringToStructField:          map[OtherString]OtherStruct{"a": {"foo"}, "b": {"bar"}},
		MapOfStringToStructPtrField:       map[OtherString]*OtherStruct{"a": {"foo"}, "b": {"bar"}},
		MapOfStringToOpaqueStructField:    map[OtherString]OtherStruct{"a": {"foo"}, "b": {"bar"}},
		MapOfStringToOpaqueStructPtrField: map[OtherString]*OtherStruct{"a": {"foo"}, "b": {"bar"}},
	}).ExpectValidateFalseByPath(map[string][]string{
		"structField":                            {"field Struct.StructField", "type OtherStruct"},
		"structPtrField":                         {"field Struct.StructPtrField", "type OtherStruct"},
		"structField.stringField":                {"field OtherStruct.StringField"},
		"structPtrField.stringField":             {"field OtherStruct.StringField"},
		"opaqueStructField":                      {"field Struct.OpaqueStructField"},
		"opaqueStructPtrField":                   {"field Struct.OpaqueStructPtrField"},
		"sliceOfStructField":                     {"field Struct.SliceOfStructField"},
		"sliceOfStructField[0]":                  {"field Struct.SliceOfStructField vals", "type OtherStruct"},
		"sliceOfStructField[0].stringField":      {"field OtherStruct.StringField"},
		"sliceOfStructField[1]":                  {"field Struct.SliceOfStructField vals", "type OtherStruct"},
		"sliceOfStructField[1].stringField":      {"field OtherStruct.StringField"},
		"sliceOfStructPtrField":                  {"field Struct.SliceOfStructPtrField"},
		"sliceOfStructPtrField[0]":               {"field Struct.SliceOfStructPtrField vals", "type OtherStruct"},
		"sliceOfStructPtrField[0].stringField":   {"field OtherStruct.StringField"},
		"sliceOfStructPtrField[1]":               {"field Struct.SliceOfStructPtrField vals", "type OtherStruct"},
		"sliceOfStructPtrField[1].stringField":   {"field OtherStruct.StringField"},
		"sliceOfOpaqueStructField":               {"field Struct.SliceOfOpaqueStructField"},
		"sliceOfOpaqueStructField[0]":            {"field Struct.SliceOfOpaqueStructField vals"},
		"sliceOfOpaqueStructField[1]":            {"field Struct.SliceOfOpaqueStructField vals"},
		"sliceOfOpaqueStructPtrField":            {"field Struct.SliceOfOpaqueStructPtrField"},
		"sliceOfOpaqueStructPtrField[0]":         {"field Struct.SliceOfOpaqueStructPtrField vals"},
		"sliceOfOpaqueStructPtrField[1]":         {"field Struct.SliceOfOpaqueStructPtrField vals"},
		"listMapOfStructField":                   {"field Struct.ListMapOfStructField"},
		"listMapOfStructField[0]":                {"field Struct.ListMapOfStructField vals", "type OtherStruct"},
		"listMapOfStructField[0].stringField":    {"field OtherStruct.StringField"},
		"listMapOfStructField[1]":                {"field Struct.ListMapOfStructField vals", "type OtherStruct"},
		"listMapOfStructField[1].stringField":    {"field OtherStruct.StringField"},
		"listMapOfStructPtrField":                {"field Struct.ListMapOfStructPtrField"},
		"listMapOfStructPtrField[0]":             {"field Struct.ListMapOfStructPtrField vals", "type OtherStruct"},
		"listMapOfStructPtrField[0].stringField": {"field OtherStruct.StringField"},
		"listMapOfStructPtrField[1]":             {"field Struct.ListMapOfStructPtrField vals", "type OtherStruct"},
		"listMapOfStructPtrField[1].stringField": {"field OtherStruct.StringField"},
		"listMapOfOpaqueStructField":             {"field Struct.ListMapOfOpaqueStructField"},
		"listMapOfOpaqueStructField[0]":          {"field Struct.ListMapOfOpaqueStructField vals"},
		"listMapOfOpaqueStructField[1]":          {"field Struct.ListMapOfOpaqueStructField vals"},
		"listMapOfOpaqueStructPtrField":          {"field Struct.ListMapOfOpaqueStructPtrField"},
		"listMapOfOpaqueStructPtrField[0]":       {"field Struct.ListMapOfOpaqueStructPtrField vals"},
		"listMapOfOpaqueStructPtrField[1]":       {"field Struct.ListMapOfOpaqueStructPtrField vals"},
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
		"mapOfStringToStructPtrField": {
			"field Struct.MapOfStringToStructPtrField",
			"field Struct.MapOfStringToStructPtrField keys",
			"field Struct.MapOfStringToStructPtrField keys",
			"type OtherString",
			"type OtherString",
		},
		"mapOfStringToStructPtrField[a]":             {"field Struct.MapOfStringToStructPtrField vals", "type OtherStruct"},
		"mapOfStringToStructPtrField[a].stringField": {"field OtherStruct.StringField"},
		"mapOfStringToStructPtrField[b]":             {"field Struct.MapOfStringToStructPtrField vals", "type OtherStruct"},
		"mapOfStringToStructPtrField[b].stringField": {"field OtherStruct.StringField"},
		"mapOfStringToOpaqueStructField": {
			"field Struct.MapOfStringToOpaqueStructField",
			"field Struct.MapOfStringToOpaqueStructField keys",
			"field Struct.MapOfStringToOpaqueStructField keys",
		},
		"mapOfStringToOpaqueStructField[a]": {"field Struct.MapOfStringToOpaqueStructField vals"},
		"mapOfStringToOpaqueStructField[b]": {"field Struct.MapOfStringToOpaqueStructField vals"},
		"mapOfStringToOpaqueStructPtrField": {
			"field Struct.MapOfStringToOpaqueStructPtrField",
			"field Struct.MapOfStringToOpaqueStructPtrField keys",
			"field Struct.MapOfStringToOpaqueStructPtrField keys",
		},
		"mapOfStringToOpaqueStructPtrField[a]": {"field Struct.MapOfStringToOpaqueStructPtrField vals"},
		"mapOfStringToOpaqueStructPtrField[b]": {"field Struct.MapOfStringToOpaqueStructPtrField vals"},
	})
}
