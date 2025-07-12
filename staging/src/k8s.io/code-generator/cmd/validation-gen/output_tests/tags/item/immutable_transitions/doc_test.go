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

package transitions

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	old := &Struct{
		ListField: []Item{
			{Key1: "a", StringField: "s1"},
			{Key1: "b", StringField: "s2"},
			{Key1: "c", StringField: "s3"},
		},
	}

	new := &Struct{
		ListField: []Item{
			{Key1: "a", StringField: "changed"},
			{Key1: "b", StringField: "changed"},
			{Key1: "c", StringField: "changed"},
		},
	}

	st.Value(new).OldValue(old).ExpectInvalid(
		field.Forbidden(field.NewPath("listField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("listField").Index(1).Child("stringField"), "field is immutable"),
	)

	st.Value(new).OldValue(&Struct{ListField: []Item{}}).ExpectInvalid(
		field.Forbidden(field.NewPath("listField").Index(0), "field is immutable"),
		field.Forbidden(field.NewPath("listField").Index(1).Child("stringField"), "field is immutable"),
	)

	// Test that "c" can change independently
	st.Value(&Struct{
		ListField: []Item{
			{Key1: "a", StringField: "s1"},
			{Key1: "b", StringField: "s2"},
			{Key1: "c", StringField: "changed"},
		},
	}).OldValue(old).ExpectValid()
}
