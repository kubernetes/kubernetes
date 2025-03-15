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
		"stringField":         []string{"Required value"},
		"stringPtrField":      []string{"Required value"},
		"otherStructPtrField": []string{"Required value"},
		"sliceField":          []string{"Required value"},
		"mapField":            []string{"Required value"},
	})

	st.Value(&Struct{
		StringPtrField: ptr.To(""),          // satisfies required
		SliceField:     []string{},          // does not satisfy required
		MapField:       map[string]string{}, // does not satisfy required
	}).ExpectRegexpsByPath(map[string][]string{
		"stringField":         []string{"Required value"},
		"stringPtrField":      []string{"forced failure: field Struct.StringPtrField"},
		"otherStructPtrField": []string{"Required value"},
		"sliceField":          []string{"Required value"},
		"mapField":            []string{"Required value"},
	})

	st.Value(&Struct{
		StringField:         "abc",
		StringPtrField:      ptr.To("xyz"),
		OtherStructPtrField: &OtherStruct{},
		SliceField:          []string{"a", "b"},
		MapField:            map[string]string{"a": "b", "c": "d"},
	}).ExpectRegexpsByPath(map[string][]string{
		"stringField":    []string{"forced failure: field Struct.StringField"},
		"stringPtrField": []string{"forced failure: field Struct.StringPtrField"},
		"otherStructPtrField": []string{
			"forced failure: type OtherStruct",
			"forced failure: field Struct.OtherStructPtrField",
		},
		"sliceField": []string{"forced failure: field Struct.SliceField"},
		"mapField":   []string{"forced failure: field Struct.MapField"},
	})
}
