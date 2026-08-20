/*
Copyright The Kubernetes Authors.

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

package mapofpointers

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// 1. Zero values should be valid
	st.Value(&Struct{
		// All zero values.
	}).ExpectValid()

	// 2. Non-nil elements trigger validation errors
	st.Value(&Struct{
		MapField:          map[string]*OtherStruct{"k1": {}, "k2": {}},
		MapPrimitiveField: map[string]*string{"k1": new("a"), "k2": new("b")},
	}).ExpectValidateFalseByPath(map[string][]string{
		"mapField[k1]":          {"field Struct.MapField[*]", "type OtherStruct"},
		"mapField[k2]":          {"field Struct.MapField[*]", "type OtherStruct"},
		"mapPrimitiveField[k1]": {"field Struct.MapPrimitiveField[*]"},
		"mapPrimitiveField[k2]": {"field Struct.MapPrimitiveField[*]"},
	})

	// 3. Nil elements trigger Required errors from PtrMapNoNils
	st.Value(&Struct{
		MapField:          map[string]*OtherStruct{"k1": nil},
		MapPrimitiveField: map[string]*string{"k1": nil},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("mapField").Key("k1"), ""),
		field.Required(field.NewPath("mapPrimitiveField").Key("k1"), ""),
	})
}
