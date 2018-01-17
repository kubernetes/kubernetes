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
