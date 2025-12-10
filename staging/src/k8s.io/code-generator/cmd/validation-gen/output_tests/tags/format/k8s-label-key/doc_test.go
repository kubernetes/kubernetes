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

package format

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	validCases := []string{
		"simple",
		"now-with-dashes",
		"1-starts-with-num",
		"1234",
		"simple.com/simple",
		"now-with-dashes.com/simple",
		"now.with.dots.com/simple",
		"now-with.dashes-and.dots.com/simple",
		"1-num.2-num.com/3-num",
		"1234.com/5678",
		"1.2.3.4/5678",
		"Uppercase_Is_OK_123",
		"example.com/Uppercase_Is_OK_123",
		"requests.storage-foo",
		strings.Repeat("a", 63),
		strings.Repeat("a", 253) + "/" + strings.Repeat("b", 63),
	}

	for _, s := range validCases {
		st.Value(&Struct{
			LabelKeyField:        s,
			LabelKeyPtrField:     ptr.To(s),
			LabelKeyTypedefField: LabelKeyStringType(s),
		}).ExpectValid()
	}

	invalidCases := []string{
		"nospecialchars%^=@",
		"cantendwithadash-",
		"-cantstartwithadash-",
		"only/one/slash",
		"Example.com/abc",
		"example_com/abc",
		"example.com/",
		"/simple",
		strings.Repeat("a", 64),
		strings.Repeat("a", 254) + "/abc",
	}

	for _, s := range invalidCases {
		st.Value(&Struct{
			LabelKeyField:        s,
			LabelKeyPtrField:     ptr.To(s),
			LabelKeyTypedefField: LabelKeyStringType(s),
		}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin(), field.ErrorList{
			field.Invalid(field.NewPath("labelKeyField"), nil, "").WithOrigin("format=k8s-label-key"),
			field.Invalid(field.NewPath("labelKeyPtrField"), nil, "").WithOrigin("format=k8s-label-key"),
			field.Invalid(field.NewPath("labelKeyTypedefField"), nil, "").WithOrigin("format=k8s-label-key"),
		})
	}
}
