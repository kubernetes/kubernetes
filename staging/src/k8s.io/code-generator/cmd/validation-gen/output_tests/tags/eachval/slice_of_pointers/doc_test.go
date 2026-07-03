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

package sliceofpointers

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// 1. Zero values should be valid
	st.Value(&Struct{
		// All zero values.
	}).ExpectValid()

	// 2. Non-nil elements trigger validation errors
	st.Value(&Struct{
		ListField:          []*OtherStruct{{}, {}},
		ListPrimitiveField: []*string{ptr.To("a"), ptr.To("b")},
	}).ExpectValidateFalseByPath(map[string][]string{
		"listField[0]":          {"field Struct.ListField[*]", "type OtherStruct"},
		"listField[1]":          {"field Struct.ListField[*]", "type OtherStruct"},
		"listPrimitiveField[0]": {"field Struct.ListPrimitiveField[*]"},
		"listPrimitiveField[1]": {"field Struct.ListPrimitiveField[*]"},
	})

	// 3. Nil elements trigger Required errors from EachPtrSliceVal
	st.Value(&Struct{
		ListField:          []*OtherStruct{nil},
		ListPrimitiveField: []*string{nil},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Required(field.NewPath("listField").Index(0), ""),
		field.Required(field.NewPath("listPrimitiveField").Index(0), ""),
	})
}
