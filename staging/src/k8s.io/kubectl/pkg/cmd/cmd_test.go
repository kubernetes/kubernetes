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
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/kubectl/pkg/cmd/plugin"
)

func TestNormalizationFuncGlobalExistence(t *testing.T) {
	// This test can be safely deleted when we will not support multiple flag formats
	root := NewKubectlCommand(KubectlOptions{IOStreams: genericiooptions.IOStreams{In: os.Stdin, Out: os.Stdout, ErrOut: os.Stderr}})

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

func TestKubectlSubcommandShadowPlugin(t *testing.T) {
	tests := []struct {
		name              string
		args              []string
		expectPlugin      string
		expectPluginArgs  []string
		expectLookupError string
	}{
		{
			name:             "test that a plugin executable is found based on command args when builtin subcommand does not exist",
			args:             []string{"kubectl", "create", "foo", "--bar", "--bar2", "--namespace", "test-namespace"},
			expectPlugin:     "plugin/testdata/kubectl-create-foo",
			expectPluginArgs: []string{"--bar", "--bar2", "--namespace", "test-namespace"},
		},
		{
			name:              "test that a plugin executable is not found based on command args when also builtin subcommand does not exist",
			args:              []string{"kubectl", "create", "foo2", "--bar", "--bar2", "--namespace", "test-namespace"},
			expectLookupError: "unable to find a plugin executable \"kubectl-create-foo2\"",
		},
		{
			name:             "test that normal commands are able to be executed, when builtin subcommand exists",
			args:             []string{"kubectl", "create", "job", "foo", "--image=busybox", "--dry-run=client", "--namespace", "test-namespace"},
			expectPlugin:     "",
			expectPluginArgs: []string{},
		},
		// rest of the tests are copied from TestKubectlCommandHandlesPlugins function,
		// just to retest them also when feature is enabled.
		{
			name:             "test that normal commands are able to be executed, when no plugin overshadows them",
			args:             []string{"kubectl", "config", "get-clusters"},
			expectPlugin:     "",
			expectPluginArgs: []string{},
		},
		{
			name:             "test that a plugin executable is found based on command args",
			args:             []string{"kubectl", "foo", "--bar"},
			expectPlugin:     "plugin/testdata/kubectl-foo",
			expectPluginArgs: []string{"--bar"},
		},
		{
			name: "test that a plugin does not execute over an existing command by the same name",
			args: []string{"kubectl", "version", "--client=true"},
		},
		{
			name: "test that a plugin does not execute over Cobra's help command",
			args: []string{"kubectl", "help"},
		},
		{
			name: "test that a plugin does not execute over Cobra's __complete command",
			args: []string{"kubectl", cobra.ShellCompRequestCmd, "de"},
		},
		{
			name: "test that a plugin does not execute over Cobra's __completeNoDesc command",
			args: []string{"kubectl", cobra.ShellCompNoDescRequestCmd, "de"},
		},
		{
			name: "test that a flag does not break Cobra's help command",
			args: []string{"kubectl", "--kubeconfig=/path/to/kubeconfig", "help"},
		},
		{
			name: "test that a flag does not break Cobra's __complete command",
			args: []string{"kubectl", "--kubeconfig=/path/to/kubeconfig", cobra.ShellCompRequestCmd},
		},
		{
			name: "test that a flag does not break Cobra's __completeNoDesc command",
			args: []string{"kubectl", "--kubeconfig=/path/to/kubeconfig", cobra.ShellCompNoDescRequestCmd},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pluginsHandler := &testPluginHandler{
				pluginsDirectory: "plugin/testdata",
				validPrefixes:    plugin.ValidPluginFilenamePrefixes,
			}
			ioStreams, _, _, _ := genericiooptions.NewTestIOStreams()
			root := NewDefaultKubectlCommandWithArgs(KubectlOptions{PluginHandler: pluginsHandler, Arguments: test.args, IOStreams: ioStreams})
			// original plugin handler (DefaultPluginHandler) is implemented by exec call so no additional actions are expected on the cobra command if we activate the plugin flow
			if !pluginsHandler.lookedup && !pluginsHandler.executed {
				// args must be set, otherwise Execute will use os.Args (args used for starting the test) and test.args would not be passed
				// to the command which might invoke only "kubectl" without any additional args and give false positives
				root.SetArgs(test.args[1:])
				// Important note! Incorrect command or command failing validation might just call os.Exit(1) which would interrupt execution of the test
				if err := root.Execute(); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}

			if (pluginsHandler.lookupErr != nil && pluginsHandler.lookupErr.Error() != test.expectLookupError) ||
				(pluginsHandler.lookupErr == nil && len(test.expectLookupError) > 0) {
				t.Fatalf("unexpected error: expected %q to occur, but got %q", test.expectLookupError, pluginsHandler.lookupErr)
			}

			if pluginsHandler.lookedup && !pluginsHandler.executed && len(test.expectLookupError) == 0 {
				// we have to fail here, because we have found the plugin, but not executed the plugin, nor the command (this would normally result in an error: unknown command)
				t.Fatalf("expected plugin execution, but did not occur")
			}

			if pluginsHandler.executedPlugin != test.expectPlugin {
				t.Fatalf("unexpected plugin execution: expected %q, got %q", test.expectPlugin, pluginsHandler.executedPlugin)
			}

			if pluginsHandler.executed && len(test.expectPlugin) == 0 {
				t.Fatalf("unexpected plugin execution: expected no plugin, got %q", pluginsHandler.executedPlugin)
			}

			if !cmp.Equal(pluginsHandler.withArgs, test.expectPluginArgs, cmpopts.EquateEmpty()) {
				t.Fatalf("unexpected plugin execution args: expected %q, got %q", test.expectPluginArgs, pluginsHandler.withArgs)
			}
		})
	}
}

func TestKubectlCommandHandlesPlugins(t *testing.T) {
	tests := []struct {
		name              string
		args              []string
		expectPlugin      string
		expectPluginArgs  []string
		expectLookupError string
	}{
		{
			name:             "test that normal commands are able to be executed, when no plugin overshadows them",
			args:             []string{"kubectl", "config", "get-clusters"},
			expectPlugin:     "",
			expectPluginArgs: []string{},
		},
		{
			name:             "test that normal commands are able to be executed, when no plugin overshadows them and shadowing feature is not enabled",
			args:             []string{"kubectl", "create", "job", "foo", "--image=busybox", "--dry-run=client"},
			expectPlugin:     "",
			expectPluginArgs: []string{},
		},
		{
			name:             "test that a plugin executable is found based on command args",
			args:             []string{"kubectl", "foo", "--bar"},
			expectPlugin:     "plugin/testdata/kubectl-foo",
			expectPluginArgs: []string{"--bar"},
		},
		{
			name:             "test that a plugin executable is found based on command args with positional argument",
			args:             []string{"kubectl", "foo", "positional", "--bar"},
			expectPlugin:     "plugin/testdata/kubectl-foo",
			expectPluginArgs: []string{"positional", "--bar"},
		},
		{
			name:             "test that an allowed subcommand plugin executable is found based on command args with positional argument",
			args:             []string{"kubectl", "create", "foo", "positional", "--bar"},
			expectPlugin:     "plugin/testdata/kubectl-create-foo",
			expectPluginArgs: []string{"positional", "--bar"},
		},
		{
			name: "test that a plugin does not execute over an existing command by the same name",
			args: []string{"kubectl", "version", "--client=true"},
		},
		// The following tests make sure that commands added by Cobra cannot be shadowed by a plugin
		// See https://github.com/kubernetes/kubectl/issues/1116
		{
			name: "test that a plugin does not execute over Cobra's help command",
			args: []string{"kubectl", "help"},
		},
		{
			name: "test that a plugin does not execute over Cobra's __complete command",
			args: []string{"kubectl", cobra.ShellCompRequestCmd, "de"},
		},
		{
			name: "test that a plugin does not execute over Cobra's __completeNoDesc command",
			args: []string{"kubectl", cobra.ShellCompNoDescRequestCmd, "de"},
		},
		// The following tests make sure that commands added by Cobra cannot be shadowed by a plugin
		// even when a flag is specified first.  This can happen when using aliases.
		// See https://github.com/kubernetes/kubectl/issues/1119
		{
			name: "test that a flag does not break Cobra's help command",
			args: []string{"kubectl", "--kubeconfig=/path/to/kubeconfig", "help"},
		},
		{
			name: "test that a flag does not break Cobra's __complete command",
			args: []string{"kubectl", "--kubeconfig=/path/to/kubeconfig", cobra.ShellCompRequestCmd},
		},
		{
			name: "test that a flag does not break Cobra's __completeNoDesc command",
			args: []string{"kubectl", "--kubeconfig=/path/to/kubeconfig", cobra.ShellCompNoDescRequestCmd},
		},
		// As for the previous tests, an alias could add a flag without using the = form.
		// We don't support this case as parsing the flags becomes quite complicated (flags
		// that take a value, versus flags that don't)
		// {
		// 	name: "test that a flag with a space does not break Cobra's help command",
		// 	args: []string{"kubectl", "--kubeconfig", "/path/to/kubeconfig", "help"},
		// },
		// {
		// 	name: "test that a flag with a space does not break Cobra's __complete command",
		// 	args: []string{"kubectl", "--kubeconfig", "/path/to/kubeconfig", cobra.ShellCompRequestCmd},
		// },
		// {
		// 	name: "test that a flag with a space does not break Cobra's __completeNoDesc command",
		// 	args: []string{"kubectl", "--kubeconfig", "/path/to/kubeconfig", cobra.ShellCompNoDescRequestCmd},
		// },
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pluginsHandler := &testPluginHandler{
				pluginsDirectory: "plugin/testdata",
				validPrefixes:    plugin.ValidPluginFilenamePrefixes,
			}
			ioStreams, _, _, _ := genericiooptions.NewTestIOStreams()
			root := NewDefaultKubectlCommandWithArgs(KubectlOptions{PluginHandler: pluginsHandler, Arguments: test.args, IOStreams: ioStreams})
			// original plugin handler (DefaultPluginHandler) is implemented by exec call so no additional actions are expected on the cobra command if we activate the plugin flow
			if !pluginsHandler.lookedup && !pluginsHandler.executed {
				// args must be set, otherwise Execute will use os.Args (args used for starting the test) and test.args would not be passed
				// to the command which might invoke only "kubectl" without any additional args and give false positives
				root.SetArgs(test.args[1:])
				// Important note! Incorrect command or command failing validation might just call os.Exit(1) which would interrupt execution of the test
				if err := root.Execute(); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}

			if (pluginsHandler.lookupErr != nil && pluginsHandler.lookupErr.Error() != test.expectLookupError) ||
				(pluginsHandler.lookupErr == nil && len(test.expectLookupError) > 0) {
				t.Fatalf("unexpected error: expected %q to occur, but got %q", test.expectLookupError, pluginsHandler.lookupErr)
			}

			if pluginsHandler.lookedup && !pluginsHandler.executed && len(test.expectLookupError) == 0 {
				// we have to fail here, because we have found the plugin, but not executed the plugin, nor the command (this would normally result in an error: unknown command)
				t.Fatalf("expected plugin execution, but did not occur")
			}

			if pluginsHandler.executedPlugin != test.expectPlugin {
				t.Fatalf("unexpected plugin execution: expected %q, got %q", test.expectPlugin, pluginsHandler.executedPlugin)
			}

			if pluginsHandler.executed && len(test.expectPlugin) == 0 {
				t.Fatalf("unexpected plugin execution: expected no plugin, got %q", pluginsHandler.executedPlugin)
			}

			if !cmp.Equal(pluginsHandler.withArgs, test.expectPluginArgs, cmpopts.EquateEmpty()) {
				t.Fatalf("unexpected plugin execution args: expected %q, got %q", test.expectPluginArgs, pluginsHandler.withArgs)
			}
		})
	}
}

type testPluginHandler struct {
	pluginsDirectory string
	validPrefixes    []string

	// lookup results
	lookedup  bool
	lookupErr error

	// execution results
	executed       bool
	executedPlugin string
	withArgs       []string
	withEnv        []string
}

func (h *testPluginHandler) Lookup(filename string) (string, bool) {
	h.lookedup = true

	dir, err := os.Stat(h.pluginsDirectory)
	if err != nil {
		h.lookupErr = err
		return "", false
	}

	if !dir.IsDir() {
		h.lookupErr = fmt.Errorf("expected %q to be a directory", h.pluginsDirectory)
		return "", false
	}

	plugins, err := os.ReadDir(h.pluginsDirectory)
	if err != nil {
		h.lookupErr = err
		return "", false
	}

	filenameWithSuportedPrefix := ""
	for _, prefix := range h.validPrefixes {
		for _, p := range plugins {
			filenameWithSuportedPrefix = fmt.Sprintf("%s-%s", prefix, filename)
			if p.Name() == filenameWithSuportedPrefix {
				h.lookupErr = nil
				return fmt.Sprintf("%s/%s", h.pluginsDirectory, p.Name()), true
			}
		}
	}

	h.lookupErr = fmt.Errorf("unable to find a plugin executable %q", filenameWithSuportedPrefix)
	return "", false
}

func (h *testPluginHandler) Execute(executablePath string, cmdArgs, env []string) error {
	h.executed = true
	h.executedPlugin = executablePath
	h.withArgs = cmdArgs
	h.withEnv = env
	return nil
}

func TestKubectlCommandHeadersHooks(t *testing.T) {
	tests := map[string]struct {
		envVar    string
		addsHooks bool
	}{
		"empty environment variable; hooks added": {
			envVar:    "",
			addsHooks: true,
		},
		"random env var value; hooks added": {
			envVar:    "foo",
			addsHooks: true,
		},
		"true env var value; hooks added": {
			envVar:    "true",
			addsHooks: true,
		},
		"false env var value; hooks NOT added": {
			envVar:    "false",
			addsHooks: false,
		},
		"zero env var value; hooks NOT added": {
			envVar:    "0",
			addsHooks: false,
		},
	}

	for name, testCase := range tests {
		t.Run(name, func(t *testing.T) {
			cmds := &cobra.Command{}
			kubeConfigFlags := genericclioptions.NewConfigFlags(true).WithDeprecatedPasswordFlag()
			if kubeConfigFlags.WrapConfigFn != nil {
				t.Fatal("expected initial nil WrapConfigFn")
			}
			t.Setenv(kubectlCmdHeaders, testCase.envVar)
			addCmdHeaderHooks(cmds, kubeConfigFlags)
			// Valdidate whether the hooks were added.
			if testCase.addsHooks && kubeConfigFlags.WrapConfigFn == nil {
				t.Error("after adding kubectl command header, expecting non-nil WrapConfigFn")
			}
			if !testCase.addsHooks && kubeConfigFlags.WrapConfigFn != nil {
				t.Error("env var feature gate should have blocked setting WrapConfigFn")
			}
		})
	}
}
