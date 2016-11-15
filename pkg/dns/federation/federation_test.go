/*
Copyright 2016 The Kubernetes Authors.

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

package fed

import (
	"github.com/stretchr/testify/assert"

	"reflect"
	"testing"
)

func TestParseFederationsFlag(t *testing.T) {
	type TestCase struct {
		input    string
		hasError bool
		expected map[string]string
	}

	for _, testCase := range []TestCase{
		{input: "", expected: make(map[string]string)},
		{input: "a=b", expected: map[string]string{"a": "b"}},
		{input: "a=b,cc=dd", expected: map[string]string{"a": "b", "cc": "dd"}},
		{input: "abc=d.e.f", expected: map[string]string{"abc": "d.e.f"}},

		{input: "ccdd", hasError: true},
		{input: "a=b,ccdd", hasError: true},
		{input: "-", hasError: true},
		{input: "a.b.c=d.e.f", hasError: true},
	} {
		output := make(map[string]string)
		err := ParseFederationsFlag(testCase.input, output)

		if !testCase.hasError {
			assert.Nil(t, err, "unexpected err", testCase)
			assert.True(t, reflect.DeepEqual(
				testCase.expected, output), output, testCase)
		} else {
			assert.NotNil(t, err, testCase)
		}
	}
}

func TestValidateName(t *testing.T) {
	// More complete testing is done in validation.IsDNS1123Label. These
	// tests are to catch issues specific to the implementation of
	// kube-dns.
	assert.NotNil(t, ValidateName(""))
	assert.NotNil(t, ValidateName("."))
	assert.NotNil(t, ValidateName("ab.cd"))
	assert.Nil(t, ValidateName("abcd"))
}

func TestValidateDomain(t *testing.T) {
	// More complete testing is done in
	// validation.IsDNS1123Subdomain. These tests are to catch issues
	// specific to the implementation of kube-dns.
	assert.NotNil(t, ValidateDomain(""))
	assert.NotNil(t, ValidateDomain("."))
	assert.Nil(t, ValidateDomain("ab.cd"))
	assert.Nil(t, ValidateDomain("abcd"))
	assert.Nil(t, ValidateDomain("a.b.c.d"))
}
