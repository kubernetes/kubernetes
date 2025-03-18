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

package singlekey

import (
	"testing"

	field "k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	structA1 := Struct{
		ListField: []OtherStruct{
			{"key1", "one"},
			{"key2", "two"},
		},
		ListTypedefField: []OtherTypedefStruct{
			{"key1", "one"},
			{"key2", "two"},
		},
	}

	// Same data, different order.
	structA2 := Struct{
		ListField: []OtherStruct{
			{"key2", "two"},
			{"key1", "one"},
		},
		ListTypedefField: []OtherTypedefStruct{
			{"key2", "two"},
			{"key1", "one"},
		},
	}

	// Different data.
	structB := Struct{
		ListField: []OtherStruct{
			{"key3", "THREE"},
			{"key1", "ONE"},
			{"key2", "TWO"},
		},
		ListTypedefField: []OtherTypedefStruct{
			{"key3", "THREE"},
			{"key1", "ONE"},
			{"key2", "TWO"},
		},
	}

	st.Value(&structA1).OldValue(&structA2).ExpectValid()

	st.Value(&structA2).OldValue(&structA1).ExpectValid()

	st.Value(&structA1).OldValue(&structB).ExpectInvalid(
		field.Forbidden(field.NewPath("listField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("listField").Index(1), "field is immutable"),
		field.Forbidden(field.NewPath("listTypedefField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("listTypedefField").Index(1), "field is immutable"),
	)

	st.Value(&structB).OldValue(&structA1).ExpectInvalid(
		field.Forbidden(field.NewPath("listField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("listField").Index(1), "field is immutable"),
		field.Forbidden(field.NewPath("listField").Index(2), "field is immutable"),
		field.Forbidden(field.NewPath("listTypedefField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("listTypedefField").Index(1), "field is immutable"),
		field.Forbidden(field.NewPath("listTypedefField").Index(2), "field is immutable"),
	)
}
