/*
Copyright The Kubernetes Authors.

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

package plugin

import (
	"fmt"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/cobra"
)

func TestGetPluginCompletions(t *testing.T) {
	tests := []struct {
		name      string
		output    string
		want      []string
		directive cobra.ShellCompDirective
	}{
		{
			name:      "LF",
			output:    "first\tfirst description\nsecond \n:4\n",
			want:      []string{"first\tfirst description", "second "},
			directive: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:      "CRLF",
			output:    "first\tfirst description\r\nsecond \r\n:4\r\n",
			want:      []string{"first\tfirst description", "second "},
			directive: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:      "CRLF with empty lines",
			output:    "\r\nfirst\r\n\r\n:4\r\n\r\n",
			want:      []string{"first"},
			directive: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:   "CRLF without directive",
			output: "first\r\nsecond\r\n",
			want:   []string{"first", "second"},
		},
		{
			name:   "CRLF with invalid directive",
			output: "first\r\n:invalid\r\n",
			want:   []string{"first", ":invalid"},
		},
		{
			name:      "directive without trailing newline",
			output:    "first\r\n:4",
			want:      []string{"first"},
			directive: cobra.ShellCompDirectiveNoFileComp,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("KUBECTL_TEST_PLUGIN_COMPLETION_OUTPUT", tt.output)
			t.Setenv("KUBECTL_TEST_PLUGIN_COMPLETION_PROCESS", "1")
			// Use the test binary as a plugin so this also works without a Unix shell.
			completions, directive := getPluginCompletions(os.Args[0], []string{"-test.run=^TestPluginCompletionProcess$"}, os.Environ())
			if diff := cmp.Diff(tt.want, completions); diff != "" {
				t.Errorf("unexpected completions (-want +got):\n%s", diff)
			}
			if directive != tt.directive {
				t.Errorf("expected directive %v, got %v", tt.directive, directive)
			}
		})
	}
}

func TestPluginCompletionProcess(t *testing.T) {
	if os.Getenv("KUBECTL_TEST_PLUGIN_COMPLETION_PROCESS") != "1" {
		return
	}
	fmt.Print(os.Getenv("KUBECTL_TEST_PLUGIN_COMPLETION_OUTPUT"))
	os.Exit(0)
}
