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

package k8scidrv4

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestK8sCIDRv4(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid IPv4 CIDRs
	st.Value(&MyType{
		CIDRv4Field:        "192.168.1.0/24",
		CIDRv4PtrField:     ptr.To("10.0.0.0/8"),
		CIDRv4TypedefField: "172.16.0.0/12",
	}).ExpectValid()

	// Valid: interface-address form
	st.Value(&MyType{
		CIDRv4Field:        "192.168.1.5/24",
		CIDRv4PtrField:     ptr.To("10.1.2.3/8"),
		CIDRv4TypedefField: "10.1.2.3/8",
	}).ExpectValid()

	// Invalid: IPv6 CIDRs are rejected
	invalidStruct := &MyType{
		CIDRv4Field:        "2001:db8::/64",
		CIDRv4PtrField:     ptr.To("2001:db8::/64"),
		CIDRv4TypedefField: "2001:db8::/64",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("cidrv4Field"), nil, "").WithOrigin("format=k8s-cidrv4"),
		field.Invalid(field.NewPath("cidrv4PtrField"), nil, "").WithOrigin("format=k8s-cidrv4"),
		field.Invalid(field.NewPath("cidrv4TypedefField"), nil, "").WithOrigin("format=k8s-cidrv4"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()

	// Invalid: not a CIDR
	invalidStruct = &MyType{
		CIDRv4Field:        "not-a-cidr",
		CIDRv4PtrField:     ptr.To("not-a-cidr"),
		CIDRv4TypedefField: "not-a-cidr",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("cidrv4Field"), nil, "").WithOrigin("format=k8s-cidrv4"),
		field.Invalid(field.NewPath("cidrv4PtrField"), nil, "").WithOrigin("format=k8s-cidrv4"),
		field.Invalid(field.NewPath("cidrv4TypedefField"), nil, "").WithOrigin("format=k8s-cidrv4"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
