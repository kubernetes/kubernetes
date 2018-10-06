/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func TestNormalizationFuncGlobalExistence(t *testing.T) {
	// This test can be safely deleted when we will not support multiple flag formats
	root := NewKubectlCommand(os.Stdin, os.Stdout, os.Stderr)

	if root.Parent() != nil {
		t.Fatal("We expect the root command to be returned")
	}
	if root.GlobalNormalizationFunc() == nil {
		t.Fatal("We expect that root command has a global normalization function")
	}

	if reflect.ValueOf(root.GlobalNormalizationFunc()).Pointer() != reflect.ValueOf(root.Flags().GetNormalizeFunc()).Pointer() {
		t.Fatal("root command seems to have a wrong normalization function")
	}

	sub := root
	for sub.HasSubCommands() {
		sub = sub.Commands()[0]
	}

	// In case of failure of this test check this PR: spf13/cobra#110
	if reflect.ValueOf(sub.Flags().GetNormalizeFunc()).Pointer() != reflect.ValueOf(root.Flags().GetNormalizeFunc()).Pointer() {
		t.Fatal("child and root commands should have the same normalization functions")
	}
}

func Test_deprecatedAlias(t *testing.T) {
	var correctCommandCalled bool
	makeCobraCommand := func() *cobra.Command {
		cobraCmd := new(cobra.Command)
		cobraCmd.Use = "print five lines"
		cobraCmd.Run = func(*cobra.Command, []string) {
			correctCommandCalled = true
		}
		return cobraCmd
	}

	original := makeCobraCommand()
	alias := deprecatedAlias("echo", makeCobraCommand())

	if len(alias.Deprecated) == 0 {
		t.Error("deprecatedAlias should always have a non-empty .Deprecated")
	}
	if !strings.Contains(alias.Deprecated, "print") {
		t.Error("deprecatedAlias should give the name of the new function in its .Deprecated field")
	}
	if !alias.Hidden {
		t.Error("deprecatedAlias should never have .Hidden == false (deprecated aliases should be hidden)")
	}

	if alias.Name() != "echo" {
		t.Errorf("deprecatedAlias has name %q, expected %q",
			alias.Name(), "echo")
	}
	if original.Name() != "print" {
		t.Errorf("original command has name %q, expected %q",
			original.Name(), "print")
	}

	buffer := new(bytes.Buffer)
	alias.SetOutput(buffer)
	alias.Execute()
	str := buffer.String()
	if !strings.Contains(str, "deprecated") || !strings.Contains(str, "print") {
		t.Errorf("deprecation warning %q does not include enough information", str)
	}

	// It would be nice to test to see that original.Run == alias.Run
	// Unfortunately Golang does not allow comparing functions. I could do
	// this with reflect, but that's technically invoking undefined
	// behavior. Best we can do is make sure that the function is called.
	if !correctCommandCalled {
		t.Errorf("original function doesn't appear to have been called by alias")
	}
}

func TestKubectlCommandHandlesPlugins(t *testing.T) {
	tests := []struct {
		name             string
		args             []string
		expectPlugin     string
		expectPluginArgs []string
		expectError      string
	}{
		{
			name:             "test that normal commands are able to be executed, when no plugin overshadows them",
			args:             []string{"kubectl", "get", "foo"},
			expectPlugin:     "",
			expectPluginArgs: []string{},
		},
		{
			name:             "test that a plugin executable is found based on command args",
			args:             []string{"kubectl", "foo", "--bar"},
			expectPlugin:     "plugin/testdata/kubectl-foo",
			expectPluginArgs: []string{"foo", "--bar"},
		},
		{
			name: "test that a plugin does not execute over an existing command by the same name",
			args: []string{"kubectl", "version"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pluginsHandler := &testPluginHandler{
				pluginsDirectory: "plugin/testdata",
			}
			_, in, out, errOut := genericclioptions.NewTestIOStreams()

			cmdutil.BehaviorOnFatal(func(str string, code int) {
				errOut.Write([]byte(str))
			})

			root := NewDefaultKubectlCommandWithArgs(pluginsHandler, test.args, in, out, errOut)
			if err := root.Execute(); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if pluginsHandler.err != nil && pluginsHandler.err.Error() != test.expectError {
				t.Fatalf("unexpected error: expected %q to occur, but got %q", test.expectError, pluginsHandler.err)
			}

			if pluginsHandler.executedPlugin != test.expectPlugin {
				t.Fatalf("unexpected plugin execution: expedcted %q, got %q", test.expectPlugin, pluginsHandler.executedPlugin)
			}

			if len(pluginsHandler.withArgs) != len(test.expectPluginArgs) {
				t.Fatalf("unexpected plugin execution args: expedcted %q, got %q", test.expectPluginArgs, pluginsHandler.withArgs)
			}
		})
	}
}

type testPluginHandler struct {
	pluginsDirectory string

	// execution results
	executedPlugin string
	withArgs       []string
	withEnv        []string

	err error
}

func (h *testPluginHandler) Lookup(filename string) (string, error) {
	dir, err := os.Stat(h.pluginsDirectory)
	if err != nil {
		h.err = err
		return "", err
	}

	if !dir.IsDir() {
		h.err = fmt.Errorf("expected %q to be a directory", h.pluginsDirectory)
		return "", h.err
	}

	plugins, err := ioutil.ReadDir(h.pluginsDirectory)
	if err != nil {
		h.err = err
		return "", err
	}

	for _, p := range plugins {
		if p.Name() == filename {
			return fmt.Sprintf("%s/%s", h.pluginsDirectory, p.Name()), nil
		}
	}

	h.err = fmt.Errorf("unable to find a plugin executable %q", filename)
	return "", h.err
}

func (h *testPluginHandler) Execute(executablePath string, cmdArgs, env []string) error {
	h.executedPlugin = executablePath
	h.withArgs = cmdArgs
	h.withEnv = env
	return nil
}
