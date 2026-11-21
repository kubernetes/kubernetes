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
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/spf13/cobra"
)

func TestSetupPluginCompletion(t *testing.T) {
	tests := []struct {
		name         string
		args         []string
		plugins      []string
		helpers      []string
		wantCommands []string
	}{
		{name: "known plugin", args: []string{"foo", ""}, plugins: []string{"foo", "unrelated"}, helpers: []string{"foo"}, wantCommands: []string{"foo"}},
		{name: "plugin argument", args: []string{"foo", "arg", ""}, plugins: []string{"foo", "unrelated"}, helpers: []string{"foo"}, wantCommands: []string{"foo"}},
		{name: "plugin flag", args: []string{"foo", "--flag", "value", ""}, plugins: []string{"foo", "unrelated"}, helpers: []string{"foo"}, wantCommands: []string{"foo"}},
		{name: "hyphenated plugin", args: []string{"foo-bar", ""}, plugins: []string{"foo_bar", "unrelated"}, helpers: []string{"foo_bar"}, wantCommands: []string{"foo-bar"}},
		{name: "nested plugin", args: []string{"foo", "bar", ""}, plugins: []string{"foo", "foo-bar", "unrelated"}, helpers: []string{"foo", "foo-bar"}, wantCommands: []string{"foo", "foo bar"}},
		{name: "no helper preserves discovery", args: []string{"foo", ""}, plugins: []string{"foo", "foo-bar", "unrelated"}, wantCommands: []string{"foo", "foo bar", "unrelated"}},
		{name: "nested plugin without helper does not use parent helper", args: []string{"foo", "bar", ""}, plugins: []string{"foo", "foo-bar", "unrelated"}, helpers: []string{"foo"}, wantCommands: []string{"foo", "foo bar", "unrelated"}},
		{name: "top level preserves discovery", args: []string{""}, plugins: []string{"foo", "unrelated"}, helpers: []string{"foo"}, wantCommands: []string{"foo", "unrelated"}},
		{name: "partial name preserves discovery", args: []string{"fo"}, plugins: []string{"foo", "unrelated"}, helpers: []string{"foo"}, wantCommands: []string{"foo", "unrelated"}},
		{name: "built in", args: []string{"get", ""}, plugins: []string{"foo", "get"}, helpers: []string{"get"}},
		{name: "global flag", args: []string{"--namespace", "default", "foo", ""}, plugins: []string{"foo"}, helpers: []string{"foo"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			t.Setenv("PATH", dir)
			// ListPlugins currently treats Windows executable extensions as part
			// of the command name, so keep the discovery cases platform-neutral.
			if runtime.GOOS == "windows" {
				t.Skip("plugin discovery does not strip Windows executable extensions")
			}
			for _, name := range tt.plugins {
				if err := os.WriteFile(filepath.Join(dir, "kubectl-"+name), nil, 0700); err != nil {
					t.Fatal(err)
				}
			}
			for _, name := range tt.helpers {
				if err := os.WriteFile(filepath.Join(dir, "kubectl_complete-"+name), nil, 0700); err != nil {
					t.Fatal(err)
				}
			}
			root := &cobra.Command{Use: "kubectl"}
			root.AddCommand(&cobra.Command{Use: "get"})
			SetupPluginCompletion(root, tt.args)
			var got []string
			var collect func(*cobra.Command, string)
			collect = func(parent *cobra.Command, prefix string) {
				for _, child := range parent.Commands() {
					if child.Name() == "get" {
						continue
					}
					name := prefix + child.Name()
					got = append(got, name)
					collect(child, name+" ")
				}
			}
			collect(root, "")
			if !slices.Equal(got, tt.wantCommands) {
				t.Fatalf("registered commands = %v, want %v", got, tt.wantCommands)
			}
		})
	}
}
