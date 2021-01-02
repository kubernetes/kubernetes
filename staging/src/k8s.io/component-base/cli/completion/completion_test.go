/*
Copyright 2021 The Kubernetes Authors.

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

package completion

import (
	"bytes"
	"testing"

	"github.com/spf13/cobra"
)

func TestGetSupportedShells(t *testing.T) {
	shells := GetSupportedShells()
	if len(shells) == 0 {
		t.Error("No supported shells")
	}
}

func TestRunCompletion(t *testing.T) {
	var out bytes.Buffer
	type TestCase struct {
		name          string
		args          []string
		expectedError bool
	}

	testCases := []TestCase{
		{
			name:          "invalid: unsupported shell name",
			args:          []string{"unsupported"},
			expectedError: true,
		},
	}

	// Test all supported shells.
	for _, shell := range GetSupportedShells() {
		test := TestCase{
			name: "valid: test shell " + shell,
			args: []string{shell},
		}
		testCases = append(testCases, test)
	}

	// use dummy cobra commands
	parentCmd := &cobra.Command{}
	cmd := &cobra.Command{}
	parentCmd.AddCommand(cmd)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if err := RunCompletionForShell(&out, "", cmd, tc.args[0]); (err != nil) != tc.expectedError {
				t.Errorf("Test case %q: TestRunCompletion expected error: %v, saw: %v", tc.name, tc.expectedError, (err != nil))
			}
		})
	}
}
