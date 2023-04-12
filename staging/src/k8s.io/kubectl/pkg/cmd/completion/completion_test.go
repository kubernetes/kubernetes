/*
Copyright 2022 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericiooptions"
)

func TestBashCompletions(t *testing.T) {
	testCases := []struct {
		name          string
		args          []string
		expectedError string
	}{
		{
			name: "bash",
			args: []string{"bash"},
		},
		{
			name: "zsh",
			args: []string{"zsh"},
		},
		{
			name: "fish",
			args: []string{"fish"},
		},
		{
			name: "powershell",
			args: []string{"powershell"},
		},
		{
			name: "no args",
			args: []string{},
			expectedError: `Shell not specified.
See 'kubectl completion -h' for help and examples`,
		},
		{
			name: "too many args",
			args: []string{"bash", "zsh"},
			expectedError: `Too many arguments. Expected only the shell type.
See 'kubectl completion -h' for help and examples`,
		},
		{
			name: "unsupported shell",
			args: []string{"foo"},
			expectedError: `Unsupported shell type "foo".
See 'kubectl completion -h' for help and examples`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(tt *testing.T) {
			_, _, out, _ := genericiooptions.NewTestIOStreams()
			parentCmd := &cobra.Command{
				Use: "kubectl",
			}
			cmd := NewCmdCompletion(out, defaultBoilerPlate)
			parentCmd.AddCommand(cmd)
			err := RunCompletion(out, defaultBoilerPlate, cmd, tc.args)
			if tc.expectedError == "" {
				if err != nil {
					tt.Fatalf("Unexpected error: %v", err)
				}
				if out.Len() == 0 {
					tt.Fatalf("Output was not written")
				}
				if !strings.Contains(out.String(), defaultBoilerPlate) {
					tt.Fatalf("Output does not contain boilerplate:\n%s", out.String())
				}
			} else {
				if err == nil {
					tt.Fatalf("An error was expected but no error was returned")
				}
				if err.Error() != tc.expectedError {
					tt.Fatalf("Unexpected error: %v\nexpected: %v\n", err, tc.expectedError)
				}
			}
		})
	}
}
