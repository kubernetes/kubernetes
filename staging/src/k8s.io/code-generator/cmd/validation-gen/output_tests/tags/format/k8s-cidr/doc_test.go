/*
Copyright The Kubernetes Authors.

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

	st.Value(&MyType{
		CIDRField:        "10.9.8.0/24",
		CIDRPtrField:     ptr.To("2001:db8::/64"),
		CIDRTypedefField: "0.0.0.0/0",
	}).ExpectValid()

	for _, bad := range []string{
		"not-a-cidr",
		"10.9.8.0",            // no prefix length
		"010.009.008.000/24",  // leading 0s
		"::ffff:10.9.8.0/120", // IPv4-mapped IPv6
		"10.9.8.7/24",         // bits set beyond the prefix length
	} {
		invalidStruct := &MyType{
			CIDRField:        bad,
			CIDRPtrField:     ptr.To(bad),
			CIDRTypedefField: CIDRStringType(bad),
		}
		st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
			field.Invalid(field.NewPath("cidrField"), nil, "").WithOrigin("format=k8s-cidr"),
			field.Invalid(field.NewPath("cidrPtrField"), nil, "").WithOrigin("format=k8s-cidr"),
			field.Invalid(field.NewPath("cidrTypedefField"), nil, "").WithOrigin("format=k8s-cidr"),
		})
		// Values that predate strict validation must survive unchanged updates.
		st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
	}
}
