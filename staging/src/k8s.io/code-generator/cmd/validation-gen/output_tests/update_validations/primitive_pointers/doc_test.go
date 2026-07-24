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
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	structA1 := Struct{
		SP: new("zero"),
		IP: new(0),
		BP: new(false),
		FP: new(0.0),
	}

	// Same data, different pointers
	structA2 := Struct{
		SP: new("zero"),
		IP: new(0),
		BP: new(false),
		FP: new(0.0),
	}
	// Different data.
	structB := Struct{
		SP: new("one"),
		IP: new(1),
		BP: new(true),
		FP: new(1.1),
	}

	st.Value(&structA1).OldValue(&structA1).ExpectValid()

	st.Value(&structA1).OldValue(&structA2).ExpectValid()

	st.Value(&structA1).OldValue(&structB).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("sp"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("ip"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("bp"), nil, "").WithOrigin("immutable"),
		field.Invalid(field.NewPath("fp"), nil, "").WithOrigin("immutable"),
	})
}
