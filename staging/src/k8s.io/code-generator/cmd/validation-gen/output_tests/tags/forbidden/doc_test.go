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

package forbidden

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	mkTest := func() *Struct {
		return &Struct{
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
		}
	}

	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{ /* All zero-values */ }).ExpectValid()

	testVal := mkTest()
	st.Value(testVal).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("stringField"), ""),
		field.Forbidden(field.NewPath("stringPtrField"), ""),
		field.Forbidden(field.NewPath("stringTypedefField"), ""),
		field.Forbidden(field.NewPath("stringTypedefPtrField"), ""),
		field.Forbidden(field.NewPath("intField"), ""),
		field.Forbidden(field.NewPath("intPtrField"), ""),
		field.Forbidden(field.NewPath("intTypedefField"), ""),
		field.Forbidden(field.NewPath("intTypedefPtrField"), ""),
		field.Forbidden(field.NewPath("otherStructPtrField"), ""),
		field.Forbidden(field.NewPath("sliceField"), ""),
		field.Forbidden(field.NewPath("sliceTypedefField"), ""),
		field.Forbidden(field.NewPath("mapField"), ""),
		field.Forbidden(field.NewPath("mapTypedefField"), ""),
	})

	// Test validation ratcheting
	st.Value(mkTest()).OldValue(mkTest()).ExpectValid()
}
