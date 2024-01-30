/*
Copyright 2018 The Kubernetes Authors.

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

package main

import (
	"os"
	"testing"
)

func setEnvVar(t *testing.T, env, val string, exists bool) {
	if exists {
		t.Setenv(env, val)
	} else if prev, ok := os.LookupEnv(env); ok {
		t.Cleanup(func() { os.Setenv(env, prev) })

		if err := os.Unsetenv(env); err != nil {
			t.Errorf("couldn't unset env %s: %v", env, err)
		}
	}
}

func TestFallbackToEnv(t *testing.T) {
	testCases := []struct {
		desc          string
		env           string
		value         string
		valueSet      bool
		expectedValue string
		expectedError bool
	}{
		{
			desc:          "value unset",
			env:           "FOO",
			valueSet:      false,
			expectedValue: "",
			expectedError: true,
		},
		{
			desc:          "value set empty",
			env:           "FOO",
			value:         "",
			valueSet:      true,
			expectedValue: "",
			expectedError: true,
		},
		{
			desc:          "value set",
			env:           "FOO",
			value:         "foo",
			valueSet:      true,
			expectedValue: "foo",
			expectedError: false,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			setEnvVar(t, test.env, test.value, test.valueSet)
			value, err := fallbackToEnv("some-flag", test.env)
			if test.expectedError {
				if err == nil {
					t.Errorf("expected error, got: %v", err)
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if value != test.expectedValue {
					t.Errorf("unexpected result: %s, expected: %s", value, test.expectedValue)
				}
			}
		})
	}
}

func TestFallbackToEnvWithDefault(t *testing.T) {
	testCases := []struct {
		desc          string
		env           string
		value         string
		valueSet      bool
		defaultValue  string
		expectedValue string
		expectedError bool
	}{
		{
			desc:          "value unset",
			env:           "FOO",
			valueSet:      false,
			defaultValue:  "default",
			expectedValue: "default",
		},
		{
			desc:          "value set empty",
			env:           "FOO",
			value:         "",
			valueSet:      true,
			defaultValue:  "default",
			expectedValue: "default",
		},
		{
			desc:          "value set",
			env:           "FOO",
			value:         "foo",
			valueSet:      true,
			defaultValue:  "default",
			expectedValue: "foo",
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			setEnvVar(t, test.env, test.value, test.valueSet)
			value := fallbackToEnvWithDefault("some-flag", test.env, test.defaultValue)
			if value != test.expectedValue {
				t.Errorf("unexpected result: %s, expected: %s", value, test.expectedValue)
			}
		})
	}
}
