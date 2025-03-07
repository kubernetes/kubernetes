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

package primitivepointers

import (
	"testing"

	field "k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	structA1 := Struct{
		SP: ptr.To("zero"),
		IP: ptr.To(0),
		BP: ptr.To(false),
		FP: ptr.To(0.0),
	}

	// Same data, different pointers
	structA2 := Struct{
		SP: ptr.To("zero"),
		IP: ptr.To(0),
		BP: ptr.To(false),
		FP: ptr.To(0.0),
	}
	// Different data.
	structB := Struct{
		SP: ptr.To("one"),
		IP: ptr.To(1),
		BP: ptr.To(true),
		FP: ptr.To(1.1),
	}

	st.Value(&structA1).OldValue(&structA1).ExpectValid()

	st.Value(&structA1).OldValue(&structA2).ExpectValid()

	st.Value(&structA1).OldValue(&structB).ExpectInvalid(
		field.Forbidden(field.NewPath("sp"), "field is immutable"),
		field.Forbidden(field.NewPath("ip"), "field is immutable"),
		field.Forbidden(field.NewPath("bp"), "field is immutable"),
		field.Forbidden(field.NewPath("fp"), "field is immutable"),
	)
}
