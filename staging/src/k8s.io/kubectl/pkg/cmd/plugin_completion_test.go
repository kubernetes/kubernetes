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

package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"k8s.io/cli-runtime/pkg/genericiooptions"
)

func TestPluginArgumentCompletion(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test completion helpers use a shell script")
	}
	first, second := t.TempDir(), t.TempDir()
	t.Setenv("PATH", first+string(os.PathListSeparator)+second)
	for _, dir := range []string{first, second} {
		for _, name := range []string{"fast", "fast-nested", "dash_name"} {
			if dir == first && name == "fast-nested" {
				continue
			}
			if err := os.WriteFile(filepath.Join(dir, "kubectl-"+name), []byte("#!/bin/sh\nexit 1\n"), 0700); err != nil {
				t.Fatal(err)
			}
			prefix := "first"
			if dir == second {
				prefix = "second"
			}
			script := fmt.Sprintf("#!/bin/sh\nprintf '%%s\n' '%s-%s' \"$@\" ':4'\n", prefix, name)
			if err := os.WriteFile(filepath.Join(dir, "kubectl_complete-"+name), []byte(script), 0700); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, tt := range []struct {
		name string
		args []string
		want string
	}{
		{name: "arguments", args: []string{"__complete", "fast", "one", "two"}, want: "first-fast\none\ntwo\n:4\n"},
		{name: "empty argument", args: []string{"__complete", "fast", ""}, want: "first-fast\n:4\n"},
		{name: "flags", args: []string{"__complete", "fast", "--flag", "value", ""}, want: "first-fast\n--flag\nvalue\n:4\n"},
		{name: "nested", args: []string{"__complete", "fast", "nested", "arg"}, want: "second-fast-nested\narg\n:4\n"},
		{name: "hyphen", args: []string{"__complete", "dash-name", "arg"}, want: "first-dash_name\narg\n:4\n"},
		{name: "no descriptions", args: []string{"__completeNoDesc", "fast", ""}, want: "first-fast\n:4\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			streams, _, out, errOut := genericiooptions.NewTestIOStreams()
			root := NewKubectlCommand(KubectlOptions{IOStreams: streams})
			root.SetOut(out)
			root.SetErr(errOut)
			root.SetArgs(tt.args)
			if err := root.Execute(); err != nil {
				t.Fatalf("completion failed: %v (%s)", err, errOut)
			}
			if out.String() != tt.want {
				t.Fatalf("completion = %q, want %q", out.String(), tt.want)
			}
		})
	}
}

func TestPluginHelpDiscoveryIsLazy(t *testing.T) {
	for _, usage := range []bool{false, true} {
		t.Run(fmt.Sprintf("usage=%t", usage), func(t *testing.T) {
			dir := t.TempDir()
			t.Setenv("PATH", dir)
			streams, _, out, errOut := genericiooptions.NewTestIOStreams()
			root := NewKubectlCommand(KubectlOptions{IOStreams: streams})
			root.SetOut(out)
			root.SetErr(errOut)
			if err := os.WriteFile(filepath.Join(dir, "kubectl-lateplugin"), nil, 0700); err != nil {
				t.Fatal(err)
			}
			for range 2 {
				if usage {
					if err := root.Usage(); err != nil {
						t.Fatal(err)
					}
				} else {
					if err := root.Help(); err != nil {
						t.Fatal(err)
					}
				}
			}
			output := out.String() + errOut.String()
			if strings.Count(output, "Subcommands provided by plugins:") != 2 || strings.Count(output, "lateplugin") != 4 {
				t.Fatalf("plugin missing or duplicated in help: %s", output)
			}
		})
	}
}

func TestPluginCompletionWithoutDirectoryReadAccess(t *testing.T) {
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("requires Unix directory permissions for a non-root user")
	}
	dir := t.TempDir()
	t.Setenv("PATH", dir)
	for name, script := range map[string]string{
		"kubectl-fast":          "#!/bin/sh\nexit 0\n",
		"kubectl_complete-fast": "#!/bin/sh\nprintf 'choice\n:4\n'\n",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(script), 0700); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Chmod(dir, 0111); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(dir, 0700) })
	if _, err := os.ReadDir(dir); err == nil {
		t.Skip("directory remains readable")
	}
	streams, _, out, errOut := genericiooptions.NewTestIOStreams()
	root := NewKubectlCommand(KubectlOptions{IOStreams: streams})
	root.SetOut(out)
	root.SetErr(errOut)
	root.SetArgs([]string{"__complete", "fast", ""})
	if err := root.Execute(); err != nil {
		t.Fatal(err)
	}
	if out.String() != "choice\n:4\n" {
		t.Fatalf("completion = %q, stderr = %q", out.String(), errOut.String())
	}
}
