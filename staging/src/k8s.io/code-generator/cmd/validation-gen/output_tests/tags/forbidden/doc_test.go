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

package forbidden

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero-values (nil slices/maps).
	}).ExpectValid()

	st.Value(&Struct{
		// Explicit zero-values and empty slices/maps.
		StringField:           "",
		StringPtrField:        nil,
		StringTypedefField:    "",
		StringTypedefPtrField: nil,
		IntField:              0,
		IntPtrField:           nil,
		IntTypedefField:       0,
		IntTypedefPtrField:    nil,
		BoolField:             false,
		FloatField:            0.0,
		ByteField:             0,
		OtherStructPtrField:   nil,
		SliceField:            []string{},
		SliceTypedefField:     SliceType{},
		ByteArrayField:        []byte{},
		MapField:              map[string]string{},
		MapTypedefField:       MapType{},
	}).ExpectValid()

	st.Value(&Struct{
		StringField:           "abc",
		StringPtrField:        ptr.To("xyz"),
		StringTypedefField:    StringType("abc"),
		StringTypedefPtrField: ptr.To(StringType("xyz")),
		IntField:              123,
		IntPtrField:           ptr.To(456),
		IntTypedefField:       IntType(123),
		IntTypedefPtrField:    ptr.To(IntType(456)),
		BoolField:             true,
		FloatField:            1.23,
		ByteField:             'a',
		OtherStructPtrField:   &OtherStruct{},
		SliceField:            []string{"a", "b"},
		SliceTypedefField:     SliceType([]string{"a", "b"}),
		ByteArrayField:        []byte("abc"),
		MapField:              map[string]string{"a": "b", "c": "d"},
		MapTypedefField:       MapType(map[string]string{"a": "b", "c": "d"}),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.Forbidden(field.NewPath("stringField"), ""),
		field.Forbidden(field.NewPath("stringPtrField"), ""),
		field.Forbidden(field.NewPath("stringTypedefField"), ""),
		field.Forbidden(field.NewPath("stringTypedefPtrField"), ""),
		field.Forbidden(field.NewPath("intField"), ""),
		field.Forbidden(field.NewPath("intPtrField"), ""),
		field.Forbidden(field.NewPath("intTypedefField"), ""),
		field.Forbidden(field.NewPath("intTypedefPtrField"), ""),
		field.Forbidden(field.NewPath("boolField"), ""),
		field.Forbidden(field.NewPath("floatField"), ""),
		field.Forbidden(field.NewPath("byteField"), ""),
		field.Forbidden(field.NewPath("otherStructPtrField"), ""),
		field.Forbidden(field.NewPath("sliceField"), ""),
		field.Forbidden(field.NewPath("sliceTypedefField"), ""),
		field.Forbidden(field.NewPath("byteArrayField"), ""),
		field.Forbidden(field.NewPath("mapField"), ""),
		field.Forbidden(field.NewPath("mapTypedefField"), ""),
	})
}
