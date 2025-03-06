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

package optional

import (
	"testing"

	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero-values.
	}).ExpectValid()

	st.Value(&Struct{
		StringField:         "abc",
		StringPtrField:      ptr.To("xyz"),
		OtherStructPtrField: &OtherStruct{},
		SliceField:          []string{"a", "b"},
		MapField:            map[string]string{"a": "b", "c": "d"},
	}).ExpectValidateFalseByPath(map[string][]string{
		"stringField":         {"field Struct.StringField"},
		"stringPtrField":      {"field Struct.StringPtrField"},
		"otherStructPtrField": {"type OtherStruct", "field Struct.OtherStructPtrField"},
		"sliceField":          {"field Struct.SliceField"},
		"mapField":            {"field Struct.MapField"},
	})
}
