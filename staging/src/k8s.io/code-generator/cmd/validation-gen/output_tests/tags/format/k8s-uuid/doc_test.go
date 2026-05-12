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

package k8suuid

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestK8sUUID(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&MyType{
		UUIDField:        "123e4567-e89b-12d3-a456-426614174000",
		UUIDPtrField:     ptr.To("123e4567-e89b-12d3-a456-426614174000"),
		UUIDTypedefField: "123e4567-e89b-12d3-a456-426614174000",
	}).ExpectValid()

	invalidStruct := &MyType{
		UUIDField:        "123E4567-E89B-12D3-A456-426614174000",
		UUIDPtrField:     ptr.To("123E4567-E89B-12D3-A456-426614174000"),
		UUIDTypedefField: "123E4567-E89B-12D3-A456-426614174000",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("uuidField"), nil, "").WithOrigin("format=k8s-uuid"),
		field.Invalid(field.NewPath("uuidPtrField"), nil, "").WithOrigin("format=k8s-uuid"),
		field.Invalid(field.NewPath("uuidTypedefField"), nil, "").WithOrigin("format=k8s-uuid"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()

	invalidStruct = &MyType{
		UUIDField:        "not-a-uuid",
		UUIDPtrField:     ptr.To("not-a-uuid"),
		UUIDTypedefField: "not-a-uuid",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("uuidField"), nil, "").WithOrigin("format=k8s-uuid"),
		field.Invalid(field.NewPath("uuidPtrField"), nil, "").WithOrigin("format=k8s-uuid"),
		field.Invalid(field.NewPath("uuidTypedefField"), nil, "").WithOrigin("format=k8s-uuid"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
