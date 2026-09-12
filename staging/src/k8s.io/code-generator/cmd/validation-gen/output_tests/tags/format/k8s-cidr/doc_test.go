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

package k8scidr

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestK8sCIDR(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Valid IPv4 CIDR
	st.Value(&MyType{
		CIDRField:        "192.168.1.0/24",
		CIDRPtrField:     ptr.To("10.0.0.0/8"),
		CIDRTypedefField: "172.16.0.0/12",
	}).ExpectValid()

	// Valid IPv6 CIDR
	st.Value(&MyType{
		CIDRField:        "2001:db8::/64",
		CIDRPtrField:     ptr.To("::/0"),
		CIDRTypedefField: "::1/128",
	}).ExpectValid()

	// Valid: interface-address form (host bits set) - sloppy mode allows this
	st.Value(&MyType{
		CIDRField:        "192.168.1.5/24",
		CIDRPtrField:     ptr.To("10.1.2.3/8"),
		CIDRTypedefField: "2001:db8::1/64",
	}).ExpectValid()

	// Valid: leading zeros in IPv4 octets accepted (sloppy)
	st.Value(&MyType{
		CIDRField:        "010.002.003.000/24",
		CIDRPtrField:     ptr.To("010.002.003.000/24"),
		CIDRTypedefField: "010.002.003.000/24",
	}).ExpectValid()

	// Invalid: not a CIDR
	invalidStruct := &MyType{
		CIDRField:        "not-a-cidr",
		CIDRPtrField:     ptr.To("not-a-cidr"),
		CIDRTypedefField: "not-a-cidr",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("cidrField"), nil, "").WithOrigin("format=k8s-cidr"),
		field.Invalid(field.NewPath("cidrPtrField"), nil, "").WithOrigin("format=k8s-cidr"),
		field.Invalid(field.NewPath("cidrTypedefField"), nil, "").WithOrigin("format=k8s-cidr"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()

	// Invalid: IP address without prefix length
	invalidStruct = &MyType{
		CIDRField:        "192.168.1.0",
		CIDRPtrField:     ptr.To("192.168.1.0"),
		CIDRTypedefField: "192.168.1.0",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("cidrField"), nil, "").WithOrigin("format=k8s-cidr"),
		field.Invalid(field.NewPath("cidrPtrField"), nil, "").WithOrigin("format=k8s-cidr"),
		field.Invalid(field.NewPath("cidrTypedefField"), nil, "").WithOrigin("format=k8s-cidr"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
