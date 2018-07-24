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

package cmd

import (
	"bytes"
	"testing"

	"github.com/spf13/cobra"
)

const shellsError = "Unexpected empty completion shells list"

func TestNewCmdCompletion(t *testing.T) {
	var out bytes.Buffer
	shells := GetSupportedShells()
	if len(shells) == 0 {
		t.Errorf(shellsError)
	}
	// test NewCmdCompletion with a valid shell.
	// use a dummy parent command as NewCmdCompletion needs it.
	parentCmd := &cobra.Command{}
	args := []string{"completion", shells[0]}
	parentCmd.SetArgs(args)
	cmd := NewCmdCompletion(&out, "")
	parentCmd.AddCommand(cmd)
	if err := parentCmd.Execute(); err != nil {
		t.Errorf("Cannot exectute NewCmdCompletion: %v", err)
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
			name:          "invalid: missing argument",
			args:          []string{},
			expectedError: true,
		},
		{
			name:          "invalid: too many arguments",
			args:          []string{"", ""},
			expectedError: true,
		},
		{
			name:          "invalid: unsupported shell name",
			args:          []string{"unsupported"},
			expectedError: true,
		},
	}

	// test all supported shells
	shells := GetSupportedShells()
	if len(shells) == 0 {
		t.Errorf(shellsError)
	}
	for _, shell := range shells {
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
		if err := RunCompletion(&out, "", cmd, tc.args); (err != nil) != tc.expectedError {
			t.Errorf("Test case %q: TestRunCompletion expected error: %v, saw: %v", tc.name, tc.expectedError, (err != nil))
		}
	}
}
