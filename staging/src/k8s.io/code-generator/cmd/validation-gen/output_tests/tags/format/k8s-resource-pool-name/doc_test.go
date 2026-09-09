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

// +output_tests
package format

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		ResourcePoolNameField:        "foo.bar",
		ResourcePoolNamePtrField:     ptr.To("foo.bar"),
		ResourcePoolNameTypedefField: "foo.bar",
	}).ExpectValid()

	st.Value(&Struct{
		ResourcePoolNameField:        "1.2.3.4",
		ResourcePoolNamePtrField:     ptr.To("1.2.3.4"),
		ResourcePoolNameTypedefField: "1.2.3.4",
	}).ExpectValid()

	invalidStruct := &Struct{
		ResourcePoolNameField:        "",
		ResourcePoolNamePtrField:     ptr.To(""),
		ResourcePoolNameTypedefField: "",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("resourcePoolNameField"), nil, "").WithOrigin("format=k8s-resource-pool-name"),
		field.Invalid(field.NewPath("resourcePoolNamePtrField"), nil, "").WithOrigin("format=k8s-resource-pool-name"),
		field.Invalid(field.NewPath("resourcePoolNameTypedefField"), nil, "").WithOrigin("format=k8s-resource-pool-name"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()

	invalidStruct = &Struct{
		ResourcePoolNameField:        "Not a ResourcePoolName",
		ResourcePoolNamePtrField:     ptr.To("Not a ResourcePoolName"),
		ResourcePoolNameTypedefField: "Not a ResourcePoolName",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("resourcePoolNameField"), nil, "").WithOrigin("format=k8s-resource-pool-name"),
		field.Invalid(field.NewPath("resourcePoolNamePtrField"), nil, "").WithOrigin("format=k8s-resource-pool-name"),
		field.Invalid(field.NewPath("resourcePoolNameTypedefField"), nil, "").WithOrigin("format=k8s-resource-pool-name"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()

	invalidStruct = &Struct{
		ResourcePoolNameField:        "a..b",
		ResourcePoolNamePtrField:     ptr.To("a..b"),
		ResourcePoolNameTypedefField: "a..b",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("resourcePoolNameField"), nil, "").WithOrigin("format=k8s-resource-pool-name"),
		field.Invalid(field.NewPath("resourcePoolNamePtrField"), nil, "").WithOrigin("format=k8s-resource-pool-name"),
		field.Invalid(field.NewPath("resourcePoolNameTypedefField"), nil, "").WithOrigin("format=k8s-resource-pool-name"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
