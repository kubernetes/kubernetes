/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"strings"
	"testing"
)

func TestGenerateBootstrapToken(t *testing.T) {
	token, err := GenerateBootstrapToken()
	if err != nil {
		t.Fatalf("GenerateBootstrapToken returned an unexpected error: %+v", err)
	}
	if !IsValidBootstrapToken(token) {
		t.Errorf("GenerateBootstrapToken didn't generate a valid token: %q", token)
	}
}

func TestRandBytes(t *testing.T) {
	var randTest = []int{
		0,
		1,
		2,
		3,
		100,
	}

	for _, rt := range randTest {
		actual, err := randBytes(rt)
		if err != nil {
			t.Errorf("failed randBytes: %v", err)
		}
		if len(actual) != rt {
			t.Errorf("failed randBytes:\n\texpected: %d\n\t  actual: %d\n", rt, len(actual))
		}
	}
}

func TestTokenFromIDAndSecret(t *testing.T) {
	var tests = []struct {
		id       string
		secret   string
		expected string
	}{
		{"foo", "bar", "foo.bar"}, // should use default
		{"abcdef", "abcdef0123456789", "abcdef.abcdef0123456789"},
		{"h", "b", "h.b"},
	}
	for _, rt := range tests {
		actual := TokenFromIDAndSecret(rt.id, rt.secret)
		if actual != rt.expected {
			t.Errorf(
				"failed TokenFromIDAndSecret:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual,
			)
		}
	}
}

func TestIsValidBootstrapToken(t *testing.T) {
	var tests = []struct {
		token    string
		expected bool
	}{
		{token: "", expected: false},
		{token: ".", expected: false},
		{token: "1234567890123456789012", expected: false},   // invalid parcel size
		{token: "12345.1234567890123456", expected: false},   // invalid parcel size
		{token: ".1234567890123456", expected: false},        // invalid parcel size
		{token: "123456.", expected: false},                  // invalid parcel size
		{token: "123456:1234567890.123456", expected: false}, // invalid separation
		{token: "abcdef:1234567890123456", expected: false},  // invalid separation
		{token: "Abcdef.1234567890123456", expected: false},  // invalid token id
		{token: "123456.AABBCCDDEEFFGGHH", expected: false},  // invalid token secret
		{token: "123456.AABBCCD-EEFFGGHH", expected: false},  // invalid character
		{token: "abc*ef.1234567890123456", expected: false},  // invalid character
		{token: "abcdef.1234567890123456", expected: true},
		{token: "123456.aabbccddeeffgghh", expected: true},
		{token: "ABCDEF.abcdef0123456789", expected: false},
		{token: "abcdef.abcdef0123456789", expected: true},
		{token: "123456.1234560123456789", expected: true},
	}
	for _, rt := range tests {
		actual := IsValidBootstrapToken(rt.token)
		if actual != rt.expected {
			t.Errorf(
				"failed IsValidBootstrapToken for the token %q\n\texpected: %t\n\t  actual: %t",
				rt.token,
				rt.expected,
				actual,
			)
		}
	}
}

func TestIsValidBootstrapTokenID(t *testing.T) {
	var tests = []struct {
		tokenID  string
		expected bool
	}{
		{tokenID: "", expected: false},
		{tokenID: "1234567890123456789012", expected: false},
		{tokenID: "12345", expected: false},
		{tokenID: "Abcdef", expected: false},
		{tokenID: "ABCDEF", expected: false},
		{tokenID: "abcdef.", expected: false},
		{tokenID: "abcdef", expected: true},
		{tokenID: "123456", expected: true},
	}
	for _, rt := range tests {
		actual := IsValidBootstrapTokenID(rt.tokenID)
		if actual != rt.expected {
			t.Errorf(
				"failed IsValidBootstrapTokenID for the token %q\n\texpected: %t\n\t  actual: %t",
				rt.tokenID,
				rt.expected,
				actual,
			)
		}
	}
}

func TestBootstrapTokenSecretName(t *testing.T) {
	var tests = []struct {
		tokenID  string
		expected string
	}{
		{"foo", "bootstrap-token-foo"},
		{"bar", "bootstrap-token-bar"},
		{"", "bootstrap-token-"},
		{"abcdef", "bootstrap-token-abcdef"},
	}
	for _, rt := range tests {
		actual := BootstrapTokenSecretName(rt.tokenID)
		if actual != rt.expected {
			t.Errorf(
				"failed BootstrapTokenSecretName:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual,
			)
		}
	}
}

func TestValidateBootstrapGroupName(t *testing.T) {
	tests := []struct {
		name  string
		input string
		valid bool
	}{
		{"valid", "system:bootstrappers:foo", true},
		{"valid nested", "system:bootstrappers:foo:bar:baz", true},
		{"valid with dashes and number", "system:bootstrappers:foo-bar-42", true},
		{"invalid uppercase", "system:bootstrappers:Foo", false},
		{"missing prefix", "foo", false},
		{"prefix with no body", "system:bootstrappers:", false},
		{"invalid spaces", "system:bootstrappers: ", false},
		{"invalid asterisk", "system:bootstrappers:*", false},
		{"trailing colon", "system:bootstrappers:foo:", false},
		{"trailing dash", "system:bootstrappers:foo-", false},
		{"script tags", "system:bootstrappers:<script> alert(\"scary?!\") </script>", false},
		{"too long", "system:bootstrappers:" + strings.Repeat("x", 300), false},
	}
	for _, test := range tests {
		err := ValidateBootstrapGroupName(test.input)
		if err != nil && test.valid {
			t.Errorf("test %q: ValidateBootstrapGroupName(%q) returned unexpected error: %v", test.name, test.input, err)
		}
		if err == nil && !test.valid {
			t.Errorf("test %q: ValidateBootstrapGroupName(%q) was supposed to return an error but didn't", test.name, test.input)
		}
	}
}

func TestValidateUsages(t *testing.T) {
	tests := []struct {
		name  string
		input []string
		valid bool
	}{
		{"valid of signing", []string{"signing"}, true},
		{"valid of authentication", []string{"authentication"}, true},
		{"all valid", []string{"authentication", "signing"}, true},
		{"single invalid", []string{"authentication", "foo"}, false},
		{"all invalid", []string{"foo", "bar"}, false},
	}

	for _, test := range tests {
		err := ValidateUsages(test.input)
		if err != nil && test.valid {
			t.Errorf("test %q: ValidateUsages(%v) returned unexpected error: %v", test.name, test.input, err)
		}
		if err == nil && !test.valid {
			t.Errorf("test %q: ValidateUsages(%v) was supposed to return an error but didn't", test.name, test.input)
		}
	}
}
