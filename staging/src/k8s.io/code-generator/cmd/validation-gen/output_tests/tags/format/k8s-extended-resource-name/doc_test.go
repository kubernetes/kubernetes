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

package k8sextendedresourcename

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestK8sExtendedResourceName(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&MyType{
		NameField:        "example.com/my-resource",
		NamePtrField:     ptr.To("my-domain.org/foo"),
		NameTypedefField: "example.com/another-resource",
	}).ExpectValid()

	st.Value(&MyType{
		NameField:        "example.com/my_resource",
		NamePtrField:     ptr.To("example.com/My-Resource"),
		NameTypedefField: "example.com/my.resource",
	}).ExpectValid()

	invalidStruct := &MyType{
		NameField:        "kubernetes.io/my-resource",
		NamePtrField:     ptr.To("requests.example.com/my-resource"),
		NameTypedefField: "my-resource",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("nameField"), invalidStruct.NameField, "a qualified name must not be a reserved name").WithOrigin("format=k8s-extended-resource-name"),
		field.Invalid(field.NewPath("namePtrField"), *invalidStruct.NamePtrField, "a qualified name must not have a reserved prefix").WithOrigin("format=k8s-extended-resource-name"),
		field.Invalid(field.NewPath("nameTypedefField"), invalidStruct.NameTypedefField, "a qualified name must be a valid domain prefix and a name separated by a slash").WithOrigin("format=k8s-extended-resource-name"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()

	const commonDetail = "a valid extended resource name must consist of a domain name prefix and a path segment separated by a slash, where the path segment consists of alphanumeric characters, '-', and must start and end with an alphanumeric character, and the domain name prefix is a valid DNS subdomain name, with the exception that 'requests' is a valid domain name prefix"
	invalidStruct = &MyType{
		NameField:        "example.com/my-resource-",
		NamePtrField:     ptr.To("example.com/-my-resource"),
		NameTypedefField: "example.com/",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("nameField"), invalidStruct.NameField, commonDetail).WithOrigin("format=k8s-extended-resource-name"),
		field.Invalid(field.NewPath("namePtrField"), *invalidStruct.NamePtrField, commonDetail).WithOrigin("format=k8s-extended-resource-name"),
		field.Invalid(field.NewPath("nameTypedefField"), invalidStruct.NameTypedefField, commonDetail).WithOrigin("format=k8s-extended-resource-name"),
	})

	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
