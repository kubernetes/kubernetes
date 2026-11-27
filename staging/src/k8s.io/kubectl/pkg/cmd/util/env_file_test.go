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
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// Test the cases of proccessEnvFileLine that can be run without touching the
// environment.
func Test_processEnvFileLine(t *testing.T) {
	testCases := []struct {
		name          string
		line          []byte
		currentLine   int
		expectedKey   string
		expectedValue string
		expectedErr   string
	}{
		{"the utf8bom is trimmed on the first line",
			append(utf8bom, 'a', '=', 'c'), 0, "a", "c", ""},

		{"the utf8bom is NOT trimmed on the second line",
			append(utf8bom, 'a', '=', 'c'), 1, "", "", "not a valid key name"},

		{"no key is returned on a comment line",
			[]byte{' ', '#', 'c'}, 0, "", "", ""},

		{"no key is returned on a blank line",
			[]byte{' ', ' ', '\t'}, 0, "", "", ""},

		{"key is returned even with no value",
			[]byte{' ', 'x', '='}, 0, "x", "", ""},
	}
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			key, value, err := processEnvFileLine(tt.line, `filename`, tt.currentLine)
			t.Logf("Testing that %s.", tt.name)
			if tt.expectedKey != key {
				t.Errorf("\texpected key %q, received %q", tt.expectedKey, key)
			}
			if tt.expectedValue != value {
				t.Errorf("\texpected value %q, received %q", tt.expectedValue, value)
			}
			if len(tt.expectedErr) == 0 {
				if err != nil {
					t.Errorf("\tunexpected err %v", err)
				}
			} else {
				if !strings.Contains(err.Error(), tt.expectedErr) {
					t.Errorf("\terr %v doesn't match expected %q", err, tt.expectedErr)
				}
			}
		})
	}
}

// proccessEnvFileLine needs to fetch the value from the environment if no
// equals sign is provided.
// For example:
//
//	my_key1=alpha
//	my_key2=beta
//	my_key3
//
// In this file, my_key3 must be fetched from the environment.
// Test this capability.
func Test_processEnvFileLine_readEnvironment(t *testing.T) {
	const realKey = "k8s_test_env_file_key"
	const realValue = `my_value`

	t.Setenv(realKey, `my_value`)

	key, value, err := processEnvFileLine([]byte(realKey), `filename`, 3)
	if err != nil {
		t.Fatal(err)
	}
	if key != realKey {
		t.Errorf(`expected key %q, received %q`, realKey, key)
	}
	if value != realValue {
		t.Errorf(`expected value %q, received %q`, realValue, value)
	}
}

func TestAddFromEnvFile(t *testing.T) {
	testCases := []struct {
		name        string
		content     string
		expected    map[string]string
		expectedErr string
	}{
		{
			name:     "single line values",
			content:  "A=1\nB=2\n",
			expected: map[string]string{"A": "1", "B": "2"},
		},
		{
			name: "multi-line value between single line values",
			content: `A=1
CERT="""
line 1
line 2
"""
B=2
`,
			expected: map[string]string{
				"A":    "1",
				"CERT": "line 1\nline 2",
				"B":    "2",
			},
		},
		{
			name: "lines inside a multi-line value are not parsed",
			content: `KEY="""
# not a comment

NOT_A_KEY=value
"""
`,
			expected: map[string]string{"KEY": "# not a comment\n\nNOT_A_KEY=value"},
		},
		{
			name:     "a lone double quote is part of the value",
			content:  "PASSWORD=\"*2KHjkdf!@kdlq\nOTHER=value\n",
			expected: map[string]string{"PASSWORD": "\"*2KHjkdf!@kdlq", "OTHER": "value"},
		},
		{
			name: "unterminated multi-line value",
			content: `A=1
KEY="""
line 1
`,
			expectedErr: `unterminated multi-line value for key "KEY" starting at line 2`,
		},
		{
			name:        "invalid utf8 inside a multi-line value",
			content:     "KEY=\"\"\"\nline 1\n\xff\n\"\"\"\n",
			expectedErr: "invalid utf8 bytes at line 3",
		},
	}
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			filePath := filepath.Join(t.TempDir(), "env")
			if err := os.WriteFile(filePath, []byte(tt.content), 0o600); err != nil {
				t.Fatal(err)
			}

			got := map[string]string{}
			err := AddFromEnvFile(filePath, func(key, value string) error {
				got[key] = value
				return nil
			})
			if len(tt.expectedErr) > 0 {
				if err == nil || !strings.Contains(err.Error(), tt.expectedErr) {
					t.Fatalf("err %v doesn't match expected %q", err, tt.expectedErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err %v", err)
			}
			if !reflect.DeepEqual(tt.expected, got) {
				t.Errorf("expected %q, received %q", tt.expected, got)
			}
		})
	}
}
