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

package enum

import (
	"testing"

	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero vals
	}).ExpectRegexpsByPath(map[string][]string{
		"enum0Field": {"Unsupported value: \"\"$"},
		"enum1Field": {"Unsupported value: \"\": supported values: \"e1v1\""},
		"enum2Field": {"Unsupported value: \"\": supported values: \"e2v1\", \"e2v2\""},
	})

	st.Value(&Struct{
		Enum0Field:      "",                // no valid value exists
		Enum0PtrField:   ptr.To(Enum0("")), // no valid value exists
		Enum1Field:      E1V1,
		Enum1PtrField:   ptr.To(E1V1),
		Enum2Field:      E2V1,
		Enum2PtrField:   ptr.To(E2V1),
		NotEnumField:    "x",
		NotEnumPtrField: ptr.To(NotEnum("x")),
	}).ExpectRegexpsByPath(map[string][]string{
		"enum0Field":    {"Unsupported value: \"\"$"},
		"enum0PtrField": {"Unsupported value: \"\"$"},
	})

	st.Value(&Struct{
		Enum0Field:      "x",                // no valid value exists
		Enum0PtrField:   ptr.To(Enum0("x")), // no valid value exists
		Enum1Field:      "x",
		Enum1PtrField:   ptr.To(Enum1("x")),
		Enum2Field:      "x",
		Enum2PtrField:   ptr.To(Enum2("x")),
		NotEnumField:    "x",
		NotEnumPtrField: ptr.To(NotEnum("x")),
	}).ExpectRegexpsByPath(map[string][]string{
		"enum0Field":    {"Unsupported value: \"x\"$"},
		"enum0PtrField": {"Unsupported value: \"x\"$"},
		"enum1Field":    {"Unsupported value: \"x\": supported values: \"e1v1\""},
		"enum1PtrField": {"Unsupported value: \"x\": supported values: \"e1v1\""},
		"enum2Field":    {"Unsupported value: \"x\": supported values: \"e2v1\", \"e2v2\""},
		"enum2PtrField": {"Unsupported value: \"x\": supported values: \"e2v1\", \"e2v2\""},
	})
}
