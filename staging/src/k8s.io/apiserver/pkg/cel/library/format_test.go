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

package library_test

import (
	"fmt"
	"testing"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"k8s.io/apiserver/pkg/cel/library"
)

func TestFormat(t *testing.T) {
	type testcase struct {
		name               string
		expr               string
		expectValue        ref.Val
		expectedCompileErr []string
		expectedRuntimeErr string
	}

	cases := []testcase{
		{
			name:        "example_usage_dns1123Label",
			expr:        `format.dns1123Label().validate("my-label-name")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "example_usage_dns1123Subdomain",
			expr:        `format.dns1123Subdomain().validate("apiextensions.k8s.io")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "example_usage_qualifiedName",
			expr:        `format.qualifiedName().validate("apiextensions.k8s.io/v1beta1")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "example_usage_dns1123LabelPrefix",
			expr:        `format.dns1123LabelPrefix().validate("my-label-prefix-")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "example_usage_dns1123SubdomainPrefix",
			expr:        `format.dns1123SubdomainPrefix().validate("mysubdomain.prefix.-")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "example_usage_dns1035LabelPrefix",
			expr:        `format.dns1035LabelPrefix().validate("my-label-prefix-")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "example_usage_uri",
			expr:        `format.uri().validate("http://example.com")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "example_usage_uuid",
			expr:        `format.uuid().validate("123e4567-e89b-12d3-a456-426614174000")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "example_usage_byte",
			expr:        `format.byte().validate("aGVsbG8=")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "example_usage_date",
			expr:        `format.date().validate("2021-01-01")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "example_usage_datetime",
			expr:        `format.datetime().validate("2021-01-01T00:00:00Z")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "dns1123Label",
			expr:        `format.dns1123Label().validate("contains a space")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{"a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"})),
		},
		{
			name:        "dns1123Subdomain",
			expr:        `format.dns1123Subdomain().validate("contains a space")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{`a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`})),
		},
		{
			name:        "dns1035Label",
			expr:        `format.dns1035Label().validate("contains a space")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{`a DNS-1035 label must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123', regex used for validation is '[a-z]([-a-z0-9]*[a-z0-9])?')`})),
		},
		{
			name:        "qualifiedName",
			expr:        `format.qualifiedName().validate("contains a space")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{`name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`})),
		},
		{
			name:        "dns1123LabelPrefix",
			expr:        `format.dns1123LabelPrefix().validate("contains a space-")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{"a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"})),
		},
		{
			name:        "dns1123SubdomainPrefix",
			expr:        `format.dns1123SubdomainPrefix().validate("contains a space-")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{`a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`})),
		},
		{
			name:        "dns1035LabelPrefix",
			expr:        `format.dns1035LabelPrefix().validate("contains a space-")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{`a DNS-1035 label must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123', regex used for validation is '[a-z]([-a-z0-9]*[a-z0-9])?')`})),
		},
		{
			name:        "dns1123Label_Success",
			expr:        `format.dns1123Label().validate("my-label-name")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "dns1123Subdomain_Success",
			expr:        `format.dns1123Subdomain().validate("example.com")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "dns1035Label_Success",
			expr:        `format.dns1035Label().validate("my-label-name")`,
			expectValue: types.OptionalNone,
		},
		{
			name:        "qualifiedName_Success",
			expr:        `format.qualifiedName().validate("my.name")`,
			expectValue: types.OptionalNone,
		},
		{
			// byte is base64
			name:        "byte_success",
			expr:        `format.byte().validate("aGVsbG8=")`,
			expectValue: types.OptionalNone,
		},
		{
			// byte is base64
			name:        "byte_failure",
			expr:        `format.byte().validate("aGVsbG8")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{"invalid base64"})),
		},
		{
			name: "date_success",
			expr: `format.date().validate("2020-01-01")`,
			// date is a valid date
			expectValue: types.OptionalNone,
		},
		{
			name:        "date_failure",
			expr:        `format.date().validate("2020-01-32")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{"invalid date"})),
		},
		{
			name: "datetime_success",
			expr: `format.datetime().validate("2020-01-01T00:00:00Z")`,
			// datetime is a valid date
			expectValue: types.OptionalNone,
		},
		{
			name:        "datetime_failure",
			expr:        `format.datetime().validate("2020-01-32T00:00:00Z")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{"invalid datetime"})),
		},
		{
			name:        "unknown_format",
			expr:        `format.named("unknown").hasValue()`,
			expectValue: types.False,
		},
		{
			name:        "labelValue_success",
			expr:        `format.labelValue().validate("my-cool-label-Value")`,
			expectValue: types.OptionalNone,
		},
		{
			name: "labelValue_failure",
			expr: `format.labelValue().validate("my-cool-label-Value!!\n\n!!!")`,
			expectValue: types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, []string{
				"a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')",
			})),
		},
	}

	// Also test format names and comparisons of all constants
	for keyLHS := range library.ConstantFormats {
		cases = append(cases, testcase{
			name:        "lookup and comparison",
			expr:        fmt.Sprintf(`format.named("%s").hasValue()`, keyLHS),
			expectValue: types.True,
		}, testcase{
			name:        "comparison with lookup succeeds",
			expr:        fmt.Sprintf(`format.named("%s").value() == format.%s()`, keyLHS, keyLHS),
			expectValue: types.True,
		})

		for keyRHS := range library.ConstantFormats {
			if keyLHS == keyRHS {
				continue
			}
			cases = append(cases, testcase{
				name:        fmt.Sprintf("compare_%s_%s", keyLHS, keyRHS),
				expr:        fmt.Sprintf(`format.%s() == format.%s()`, keyLHS, keyRHS),
				expectValue: types.False,
			})
		}
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testQuantity(t, tc.expr, tc.expectValue, tc.expectedRuntimeErr, tc.expectedCompileErr)
		})
	}
}
