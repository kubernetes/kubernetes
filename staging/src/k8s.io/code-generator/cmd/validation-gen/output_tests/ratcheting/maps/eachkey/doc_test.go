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

// +k8s:validation-gen=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme

// This is a test package.
package eachkey

import (
	"testing"
)

func Test_Struct(t *testing.T) {
	mkTest := func() *Struct {
		return &Struct{
			MapField: map[string]string{"x": "y"},
			MapTypedefField: map[UnvalidatedStringType]string{
				"x": "y",
			},
			MapValidatedTypedefField: map[ValidatedStringType]string{
				"x": "y",
			},
			ValidatedMapTypeField: map[string]string{
				"x": "y",
			},
		}
	}
	st := localSchemeBuilder.Test(t)
	st.Value(mkTest()).ExpectValidateFalseByPath(map[string][]string{
		"mapField":                 {"field Struct.MapField(keys)"},
		"mapTypedefField":          {"field Struct.MapTypedefField(keys)"},
		"mapValidatedTypedefField": {"ValidatedStringType", "field Struct.MapValidatedTypedefField(keys)"},
		"validatedMapTypeField":    {"field Struct.ValidatedMapTypeField(keys)", "type ValidatedMapType(keys)"},
	})

	st.Value(mkTest()).OldValue(mkTest()).ExpectValid()
}
