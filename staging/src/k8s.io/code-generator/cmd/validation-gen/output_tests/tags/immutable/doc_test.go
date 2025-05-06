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

package immutable

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	structA := Struct{
		StringField:                 "aaa",
		StringPtrField:              ptr.To("aaa"),
		StructField:                 ComparableStruct{"bbb", ptr.To("BBB")},
		StructPtrField:              ptr.To(ComparableStruct{"bbb", ptr.To("BBB")}),
		NonComparableStructField:    NonComparableStruct{[]string{"ccc"}},
		NonComparableStructPtrField: ptr.To(NonComparableStruct{[]string{"ccc"}}),
		SliceField:                  []string{"ddd"},
		MapField:                    map[string]string{"eee": "eee"},
		ImmutableField:              "fff",
		ImmutablePtrField:           ptr.To(ImmutableType("fff")),
	}

	structA2 := structA // dup of A but with different pointer values
	structA2.StringPtrField = ptr.To(*structA2.StringPtrField)
	structA2.StructField.StringPtrField = ptr.To("BBB")
	structA2.StructPtrField = ptr.To(*structA2.StructPtrField)
	structA2.StructPtrField.StringPtrField = ptr.To("BBB")
	structA2.NonComparableStructPtrField = ptr.To(*structA2.NonComparableStructPtrField)
	structA2.ImmutablePtrField = ptr.To(*structA2.ImmutablePtrField)

	structB := Struct{
		StringField:                 "uuu",
		StringPtrField:              ptr.To("uuu"),
		StructField:                 ComparableStruct{"vvv", ptr.To("VVV")},
		StructPtrField:              ptr.To(ComparableStruct{"vvv", ptr.To("VVV")}),
		NonComparableStructField:    NonComparableStruct{[]string{"www"}},
		NonComparableStructPtrField: ptr.To(NonComparableStruct{[]string{"www"}}),
		SliceField:                  []string{"xxx"},
		MapField:                    map[string]string{"yyy": "yyy"},
		ImmutableField:              "zzz",
		ImmutablePtrField:           ptr.To(ImmutableType("zzz")),
	}

	st.Value(&structA).OldValue(&structA).ExpectValid()
	st.Value(&structA).OldValue(&structA2).ExpectValid()
	st.Value(&structA2).OldValue(&structA).ExpectValid()

	st.Value(&structA).OldValue(&structB).ExpectInvalid(
		field.Forbidden(field.NewPath("stringField"), "field is immutable"),
		field.Forbidden(field.NewPath("stringPtrField"), "field is immutable"),
		field.Forbidden(field.NewPath("structField"), "field is immutable"),
		field.Forbidden(field.NewPath("structPtrField"), "field is immutable"),
		field.Forbidden(field.NewPath("noncomparableStructField"), "field is immutable"),
		field.Forbidden(field.NewPath("noncomparableStructPtrField"), "field is immutable"),
		field.Forbidden(field.NewPath("sliceField"), "field is immutable"),
		field.Forbidden(field.NewPath("mapField"), "field is immutable"),
		field.Forbidden(field.NewPath("immutableField"), "field is immutable"),
		field.Forbidden(field.NewPath("immutablePtrField"), "field is immutable"),
	)
}
