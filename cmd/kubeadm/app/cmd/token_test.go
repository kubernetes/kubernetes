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

package cmd

import (
	"bytes"
	"regexp"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
)

const (
	TokenExpectedRegex = "^\\S{6}\\.\\S{16}\n$"
)

func TestRunGenerateToken(t *testing.T) {
	var buf bytes.Buffer

	err := RunGenerateToken(&buf)
	if err != nil {
		t.Errorf("RunGenerateToken returned an error: %v", err)
	}

	output := buf.String()

	matched, err := regexp.MatchString(TokenExpectedRegex, output)
	if err != nil {
		t.Fatalf("encountered an error while trying to match RunGenerateToken's output: %v", err)
	}
	if !matched {
		t.Errorf("RunGenerateToken's output did not match expected regex; wanted: [%s], got: [%s]", TokenExpectedRegex, output)
	}
}

func TestRunCreateToken(t *testing.T) {
	var buf bytes.Buffer
	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("get", "secrets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, nil, errors.NewNotFound(v1.Resource("secrets"), "foo")
	})

	testCases := []struct {
		name          string
		token         string
		usages        []string
		extraGroups   []string
		expectedError bool
	}{
		{
			name:          "valid: empty token",
			token:         "",
			usages:        []string{"signing", "authentication"},
			extraGroups:   []string{"system:bootstrappers:foo"},
			expectedError: false,
		},
		{
			name:          "valid: non-empty token",
			token:         "abcdef.1234567890123456",
			usages:        []string{"signing", "authentication"},
			extraGroups:   []string{"system:bootstrappers:foo"},
			expectedError: false,
		},
		{
			name:          "valid: no extraGroups",
			token:         "abcdef.1234567890123456",
			usages:        []string{"signing", "authentication"},
			extraGroups:   []string{},
			expectedError: false,
		},
		{
			name:          "invalid: incorrect token",
			token:         "123456.AABBCCDDEEFFGGHH",
			usages:        []string{"signing", "authentication"},
			extraGroups:   []string{},
			expectedError: true,
		},
		{
			name:          "invalid: incorrect extraGroups",
			token:         "abcdef.1234567890123456",
			usages:        []string{"signing", "authentication"},
			extraGroups:   []string{"foo"},
			expectedError: true,
		},
		{
			name:          "invalid: specifying --groups when --usages doesn't include authentication",
			token:         "abcdef.1234567890123456",
			usages:        []string{"signing"},
			extraGroups:   []string{"foo"},
			expectedError: true,
		},
		{
			name:          "invalid: partially incorrect usages",
			token:         "abcdef.1234567890123456",
			usages:        []string{"foo", "authentication"},
			extraGroups:   []string{"system:bootstrappers:foo"},
			expectedError: true,
		},
		{
			name:          "invalid: all incorrect usages",
			token:         "abcdef.1234567890123456",
			usages:        []string{"foo", "bar"},
			extraGroups:   []string{"system:bootstrappers:foo"},
			expectedError: true,
		},
	}
	for _, tc := range testCases {
		err := RunCreateToken(&buf, fakeClient, tc.token, 0, tc.usages, tc.extraGroups, "", false, "")
		if (err != nil) != tc.expectedError {
			t.Errorf("Test case %s: RunCreateToken expected error: %v, saw: %v", tc.name, tc.expectedError, (err != nil))
		}
	}
}
