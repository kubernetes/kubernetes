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

package k8sip

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestK8sIP(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&MyType{
		IPField:        "10.0.0.1",
		IPPtrField:     ptr.To("2001:db8::1"),
		IPTypedefField: "192.168.01.1",
	}).ExpectValid()

	invalidStruct := &MyType{
		IPField:        "not-an-ip",
		IPPtrField:     ptr.To("not-an-ip"),
		IPTypedefField: "not-an-ip",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("ipField"), nil, "").WithOrigin("format=k8s-ip"),
		field.Invalid(field.NewPath("ipPtrField"), nil, "").WithOrigin("format=k8s-ip"),
		field.Invalid(field.NewPath("ipTypedefField"), nil, "").WithOrigin("format=k8s-ip"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()

	invalidStruct = &MyType{
		IPField:        "256.0.0.1",
		IPPtrField:     ptr.To("256.0.0.1"),
		IPTypedefField: "256.0.0.1",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("ipField"), nil, "").WithOrigin("format=k8s-ip"),
		field.Invalid(field.NewPath("ipPtrField"), nil, "").WithOrigin("format=k8s-ip"),
		field.Invalid(field.NewPath("ipTypedefField"), nil, "").WithOrigin("format=k8s-ip"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
