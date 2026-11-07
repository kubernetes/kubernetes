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

// AddFromEnvFile must not report success when bufio.Scanner stops early: a line longer than the
// 64KiB token limit otherwise silently truncates the ConfigMap or Secret and kubectl exits 0.
func TestAddFromEnvFile_oversizedLineIsAnError(t *testing.T) {
	const key = "KEY"
	dir := t.TempDir()
	path := filepath.Join(dir, "big.env")
	content := key + "=" + strings.Repeat("a", 70*1024) + "\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	got := map[string]string{}
	err := AddFromEnvFile(path, func(k, v string) error {
		got[k] = v
		return nil
	})
	if err == nil {
		t.Fatalf("expected an error for a value beyond the scanner token limit, got nil (collected %d keys)", len(got))
	}
	if _, ok := got[key]; ok {
		t.Errorf("the truncated value must not be reported as a key/value pair")
	}
}

// A file that stays within the token limit keeps working.
func TestAddFromEnvFile_withinTokenLimit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ok.env")
	value := strings.Repeat("a", 60*1024)
	if err := os.WriteFile(path, []byte("A=1\nB="+value+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	got := map[string]string{}
	if err := AddFromEnvFile(path, func(k, v string) error {
		got[k] = v
		return nil
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got["A"] != "1" || got["B"] != value {
		t.Errorf("unexpected values: A=%q len(B)=%d", got["A"], len(got["B"]))
	}
}
