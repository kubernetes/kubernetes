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

package zerovalueallowed

import (
	"testing"

	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Zero values are valid for value-type fields with +k8s:zeroValueAllowed.
	// Pointer fields still require a non-nil pointer (presence check).
	st.Value(&Struct{
		// IntField=0, StringField="" are valid (zero values allowed)
		// IntPtrField=nil, StringPtrField=nil should fail (pointer must be non-nil)
	}).ExpectRegexpsByPath(map[string][]string{
		"intPtrField":    {"Required value"},
		"stringPtrField": {"Required value"},
	})

	// All fields set to non-zero values: valid.
	st.Value(&Struct{
		IntField:       1,
		StringField:    "hello",
		IntPtrField:    ptr.To(42),
		StringPtrField: ptr.To("world"),
	}).ExpectValid()

	// Zero values via pointers are valid (pointer is non-nil, value is zero).
	st.Value(&Struct{
		IntField:       0,
		StringField:    "",
		IntPtrField:    ptr.To(0),
		StringPtrField: ptr.To(""),
	}).ExpectValid()
}
