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

package required

import (
	"testing"

	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero-values.
	}).ExpectRegexpsByPath(map[string][]string{
		"stringField":           []string{"Required value"},
		"stringPtrField":        []string{"Required value"},
		"stringTypedefField":    []string{"Required value"},
		"stringTypedefPtrField": []string{"Required value"},
		"intField":              []string{"Required value"},
		"intPtrField":           []string{"Required value"},
		"intTypedefField":       []string{"Required value"},
		"intTypedefPtrField":    []string{"Required value"},
		"otherStructPtrField":   []string{"Required value"},
		"sliceField":            []string{"Required value"},
		"sliceTypedefField":     []string{"Required value"},
		"mapField":              []string{"Required value"},
		"mapTypedefField":       []string{"Required value"},
	})

	st.Value(&Struct{
		StringPtrField:        ptr.To(""),             // satisfies required
		StringTypedefPtrField: ptr.To(StringType("")), // satisfies required
		IntPtrField:           ptr.To(0),              // satisfies required
		IntTypedefPtrField:    ptr.To(IntType(0)),     // satisfies required
		SliceField:            []string{},             // does not satisfy required
		SliceTypedefField:     []string{},             // does not satisfy required
		MapField:              map[string]string{},    // does not satisfy required
		MapTypedefField:       map[string]string{},    // does not satisfy required
	}).ExpectRegexpsByPath(map[string][]string{
		"stringField":           []string{"Required value"},
		"stringPtrField":        []string{"field Struct.StringPtrField"},
		"stringTypedefField":    []string{"Required value"},
		"stringTypedefPtrField": []string{"field Struct.StringTypedefPtrField", "type StringType"},
		"intField":              []string{"Required value"},
		"intPtrField":           []string{"field Struct.IntPtrField"},
		"intTypedefField":       []string{"Required value"},
		"intTypedefPtrField":    []string{"field Struct.IntTypedefPtrField", "type IntType"},
		"otherStructPtrField":   []string{"Required value"},
		"sliceField":            []string{"Required value"},
		"sliceTypedefField":     []string{"Required value"},
		"mapField":              []string{"Required value"},
		"mapTypedefField":       []string{"Required value"},
	})

	st.Value(&Struct{
		StringField:           "abc",
		StringPtrField:        ptr.To("xyz"),
		StringTypedefField:    StringType("abc"),
		StringTypedefPtrField: ptr.To(StringType("xyz")),
		IntField:              123,
		IntPtrField:           ptr.To(456),
		IntTypedefField:       IntType(123),
		IntTypedefPtrField:    ptr.To(IntType(456)),
		OtherStructPtrField:   &OtherStruct{},
		SliceField:            []string{"a", "b"},
		SliceTypedefField:     SliceType([]string{"a", "b"}),
		MapField:              map[string]string{"a": "b", "c": "d"},
		MapTypedefField:       MapType(map[string]string{"a": "b", "c": "d"}),
	}).ExpectRegexpsByPath(map[string][]string{
		"stringField":           []string{"field Struct.StringField"},
		"stringPtrField":        []string{"field Struct.StringPtrField"},
		"stringTypedefField":    []string{"field Struct.StringTypedefField", "type StringType"},
		"stringTypedefPtrField": []string{"field Struct.StringTypedefPtrField", "type StringType"},
		"intField":              []string{"field Struct.IntField"},
		"intPtrField":           []string{"field Struct.IntPtrField"},
		"intTypedefField":       []string{"field Struct.IntTypedefField", "type IntType"},
		"intTypedefPtrField":    []string{"field Struct.IntTypedefPtrField", "type IntType"},
		"otherStructPtrField":   []string{"field Struct.OtherStructPtrField", "type OtherStruct"},
		"sliceField":            []string{"field Struct.SliceField"},
		"sliceTypedefField":     []string{"field Struct.SliceTypedefField", "type SliceType"},
		"mapField":              []string{"field Struct.MapField"},
		"mapTypedefField":       []string{"field Struct.MapTypedefField", "type MapType"},
	})
}
