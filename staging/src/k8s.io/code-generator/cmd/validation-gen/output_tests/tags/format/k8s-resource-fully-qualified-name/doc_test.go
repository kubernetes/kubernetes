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

package fullyqualifiedname

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestFullyQualifiedName(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		FullyQualifiedNameField:        "my-prefix/my_name",
		FullyQualifiedNamePtrField:     ptr.To("my-prefix/my_name"),
		FullyQualifiedNameTypedefField: "my-prefix/my_name",
	}).ExpectValid()

	invalidStruct := &Struct{
		FullyQualifiedNameField:        "my_name",
		FullyQualifiedNamePtrField:     ptr.To(""),
		FullyQualifiedNameTypedefField: "my-prefix/",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByOrigin().ByField(), field.ErrorList{
		field.Invalid(field.NewPath("fullyQualifiedNameField"), "my_name", "a fully qualified name must be a domain and a name separated by a slash").WithOrigin("format=k8s-resource-fully-qualified-name"),
		field.Invalid(field.NewPath("fullyQualifiedNamePtrField"), "", "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_'").WithOrigin("format=k8s-resource-fully-qualified-name"),
		field.Invalid(field.NewPath("fullyQualifiedNameTypedefField"), "my-prefix/", "name must not be empty").WithOrigin("format=k8s-resource-fully-qualified-name"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
