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

package subfield

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		Items: []Item{
			{Key: "other", StringField: "anything"},
			{Key: "target", StringField: "fails"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		`items[1].stringField`: {"item Items[key=target].stringField"},
	})

	st.Value(&Struct{
		Items: []Item{
			{Key: "other", StringField: "anything"},
		},
	}).ExpectValid()

	st.Value(&Struct{
		RatchetItems: []RatchetItem{
			{Key: "ratchet", Status: "forbidden", Version: 1},
		},
	}).ExpectInvalid(
		field.Invalid(field.NewPath("ratchetItems").Index(0).Child("status"), "forbidden", "must not be equal to \"forbidden\""),
	)

	st.Value(&Struct{
		RatchetItems: []RatchetItem{
			{Key: "ratchet", Status: "allowed", Version: 1},
		},
	}).ExpectValid()

	oldStruct := &Struct{
		RatchetItems: []RatchetItem{
			{Key: "ratchet", Status: "forbidden", Version: 1},
		},
	}
	newStruct := &Struct{
		RatchetItems: []RatchetItem{
			{Key: "ratchet", Status: "forbidden", Version: 2},
		},
	}
	st.Value(newStruct).OldValue(oldStruct).ExpectValid()

	st.Value(&Struct{
		RatchetItems: []RatchetItem{
			{Key: "ratchet", Status: "forbidden", Version: 2},
		},
	}).OldValue(&Struct{
		RatchetItems: []RatchetItem{
			{Key: "ratchet", Status: "allowed", Version: 1},
		},
	}).ExpectInvalid(
		field.Invalid(field.NewPath("ratchetItems").Index(0).Child("status"), "forbidden", "must not be equal to \"forbidden\""),
	)
}
