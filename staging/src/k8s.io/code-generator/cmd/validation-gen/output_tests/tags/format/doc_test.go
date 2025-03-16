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

package format

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		IPField:              "1.2.3.4",
		IPPtrField:           ptr.To("1.2.3.4"),
		DNSLabelField:        "foo-bar",
		DNSLabelPtrField:     ptr.To("foo-bar"),
		IPTypedefField:       "1.2.3.4",
		DNSLabelTypedefField: "foo-bar",
	}).ExpectValid()

	st.Value(&Struct{
		IPField:              "abcd::1234",
		IPPtrField:           ptr.To("abcd::1234"),
		DNSLabelField:        "1234",
		DNSLabelPtrField:     ptr.To("1234"),
		IPTypedefField:       "abcd::1234",
		DNSLabelTypedefField: "1234",
	}).ExpectValid()

	st.Value(&Struct{
		IPField:              "",
		IPPtrField:           ptr.To(""),
		DNSLabelField:        "",
		DNSLabelPtrField:     ptr.To(""),
		IPTypedefField:       "",
		DNSLabelTypedefField: "",
	}).ExpectInvalid(
		field.Invalid(field.NewPath("ipField"), "", "must be a valid IP address (e.g. 10.9.8.7 or 2001:db8::ffff)"),
		field.Invalid(field.NewPath("ipPtrField"), "", "must be a valid IP address (e.g. 10.9.8.7 or 2001:db8::ffff)"),
		field.Invalid(field.NewPath("dnsLabelField"), "", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
		field.Invalid(field.NewPath("dnsLabelPtrField"), "", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
		field.Invalid(field.NewPath("ipTypedefField"), "", "must be a valid IP address (e.g. 10.9.8.7 or 2001:db8::ffff)"),
		field.Invalid(field.NewPath("dnsLabelTypedefField"), "", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
	)

	st.Value(&Struct{
		IPField:              "Not an IP",
		IPPtrField:           ptr.To("Not an IP"),
		DNSLabelField:        "Not a DNS label",
		DNSLabelPtrField:     ptr.To("Not a DNS label"),
		IPTypedefField:       "Not an IP",
		DNSLabelTypedefField: "Not a DNS label",
	}).ExpectInvalid(
		field.Invalid(field.NewPath("ipField"), "Not an IP", "must be a valid IP address (e.g. 10.9.8.7 or 2001:db8::ffff)"),
		field.Invalid(field.NewPath("ipPtrField"), "Not an IP", "must be a valid IP address (e.g. 10.9.8.7 or 2001:db8::ffff)"),
		field.Invalid(field.NewPath("dnsLabelField"), "Not a DNS label", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
		field.Invalid(field.NewPath("dnsLabelPtrField"), "Not a DNS label", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
		field.Invalid(field.NewPath("ipTypedefField"), "Not an IP", "must be a valid IP address (e.g. 10.9.8.7 or 2001:db8::ffff)"),
		field.Invalid(field.NewPath("dnsLabelTypedefField"), "Not a DNS label", "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
	)
}
