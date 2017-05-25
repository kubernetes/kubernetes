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

package kubectl

import (
	"strings"
	"testing"
)

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

		{"a key is returned even with no value next to the equals sign",
			[]byte{' ', 'x', '='}, 0, "x", "", ""},

		{"a key is returned even with no value and no equals sign",
			[]byte{' ', 'x', '1', '='}, 0, "x1", "", ""},
	}
	for _, c := range testCases {
		key, value, err := proccessEnvFileLine(c.line, `filename`, c.currentLine)
		t.Logf("Testing that %s.", c.name)
		if c.expectedKey != key {
			t.Errorf("\texpected key %q, received %q", c.expectedKey, key)
		}
		if c.expectedValue != value {
			t.Errorf("\texpected value %q, received %q", c.expectedValue, value)
		}
		if len(c.expectedErr) == 0 {
			if err != nil {
				t.Errorf("\tunexpected err %v", err)
			}
		} else {
			if !strings.Contains(err.Error(), c.expectedErr) {
				t.Errorf("\terr %v doesn't match expected %q", err, c.expectedErr)
			}
		}
	}
}
