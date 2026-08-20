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

package k8scidrv6

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestK8sCIDRv6(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid IPv6 CIDRs
	st.Value(&MyType{
		CIDRv6Field:        "2001:db8::/64",
		CIDRv6PtrField:     ptr.To("::/0"),
		CIDRv6TypedefField: "::1/128",
	}).ExpectValid()

	// Valid: interface-address form for IPv6
	st.Value(&MyType{
		CIDRv6Field:        "2001:db8::1/64",
		CIDRv6PtrField:     ptr.To("fe80::1/10"),
		CIDRv6TypedefField: "2001:db8::1/64",
	}).ExpectValid()

	// Invalid: IPv4 CIDRs are rejected
	invalidStruct := &MyType{
		CIDRv6Field:        "192.168.1.0/24",
		CIDRv6PtrField:     ptr.To("192.168.1.0/24"),
		CIDRv6TypedefField: "192.168.1.0/24",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("cidrv6Field"), nil, "").WithOrigin("format=k8s-cidrv6"),
		field.Invalid(field.NewPath("cidrv6PtrField"), nil, "").WithOrigin("format=k8s-cidrv6"),
		field.Invalid(field.NewPath("cidrv6TypedefField"), nil, "").WithOrigin("format=k8s-cidrv6"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()

	// Invalid: not a CIDR
	invalidStruct = &MyType{
		CIDRv6Field:        "not-a-cidr",
		CIDRv6PtrField:     ptr.To("not-a-cidr"),
		CIDRv6TypedefField: "not-a-cidr",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("cidrv6Field"), nil, "").WithOrigin("format=k8s-cidrv6"),
		field.Invalid(field.NewPath("cidrv6PtrField"), nil, "").WithOrigin("format=k8s-cidrv6"),
		field.Invalid(field.NewPath("cidrv6TypedefField"), nil, "").WithOrigin("format=k8s-cidrv6"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
