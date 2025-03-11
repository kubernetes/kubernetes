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

package zerodefaults

import (
	"testing"

	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero-values.
	}).ExpectRegexpsByPath(map[string][]string{
		// "stringField": optional value fields with zero defaults are just docs
		// "intField": optional value fields with zero defaults are just docs
		// "boolField": optional value fields with zero defaults are just docs
		"stringPtrField": {"Required value"},
		"intPtrField":    {"Required value"},
		"boolPtrField":   {"Required value"},
	})

	st.Value(&Struct{
		StringField:    "abc",
		StringPtrField: ptr.To(""),
		IntField:       123,
		IntPtrField:    ptr.To(0),
		BoolField:      true,
		BoolPtrField:   ptr.To(false),
	}).ExpectValid()
}
