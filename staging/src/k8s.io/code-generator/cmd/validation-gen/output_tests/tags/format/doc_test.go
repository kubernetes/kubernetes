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
		IPField:               "1.2.3.4",
		IPPtrField:            ptr.To("1.2.3.4"),
		IPTypedefField:        "1.2.3.4",
		ShortNameField:        "foo-bar",
		ShortNamePtrField:     ptr.To("foo-bar"),
		ShortNameTypedefField: "foo-bar",
	}).ExpectValid()

	st.Value(&Struct{
		IPField:               "abcd::1234",
		IPPtrField:            ptr.To("abcd::1234"),
		IPTypedefField:        "abcd::1234",
		ShortNameField:        "1234",
		ShortNamePtrField:     ptr.To("1234"),
		ShortNameTypedefField: "1234",
	}).ExpectValid()

	invalidStruct := &Struct{
		IPField:               "",
		IPPtrField:            ptr.To(""),
		IPTypedefField:        "",
		ShortNameField:        "",
		ShortNamePtrField:     ptr.To(""),
		ShortNameTypedefField: "",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("ipField"), nil, "").WithOrigin("format=k8s-ip-sloppy"),
		field.Invalid(field.NewPath("ipPtrField"), nil, "").WithOrigin("format=k8s-ip-sloppy"),
		field.Invalid(field.NewPath("ipTypedefField"), nil, "").WithOrigin("format=k8s-ip-sloppy"),
		field.Invalid(field.NewPath("shortNameField"), nil, "").WithOrigin("format=k8s-short-name"),
		field.Invalid(field.NewPath("shortNamePtrField"), nil, "").WithOrigin("format=k8s-short-name"),
		field.Invalid(field.NewPath("shortNameTypedefField"), nil, "").WithOrigin("format=k8s-short-name"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()

	invalidStruct = &Struct{
		IPField:               "Not an IP",
		IPPtrField:            ptr.To("Not an IP"),
		IPTypedefField:        "Not an IP",
		ShortNameField:        "Not a ShortName",
		ShortNamePtrField:     ptr.To("Not a ShortName"),
		ShortNameTypedefField: "Not a ShortName",
	}
	st.Value(invalidStruct).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("ipField"), nil, "").WithOrigin("format=k8s-ip-sloppy"),
		field.Invalid(field.NewPath("ipPtrField"), nil, "").WithOrigin("format=k8s-ip-sloppy"),
		field.Invalid(field.NewPath("ipTypedefField"), nil, "").WithOrigin("format=k8s-ip-sloppy"),
		field.Invalid(field.NewPath("shortNameField"), nil, "").WithOrigin("format=k8s-short-name"),
		field.Invalid(field.NewPath("shortNamePtrField"), nil, "").WithOrigin("format=k8s-short-name"),
		field.Invalid(field.NewPath("shortNameTypedefField"), nil, "").WithOrigin("format=k8s-short-name"),
	})
	// Test validation ratcheting
	st.Value(invalidStruct).OldValue(invalidStruct).ExpectValid()
}
