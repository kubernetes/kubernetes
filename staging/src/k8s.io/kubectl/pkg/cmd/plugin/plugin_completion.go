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

package plugin

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

func GetPluginCommandGroup(kubectl *cobra.Command) templates.CommandGroup {
	// Find root level
	return templates.CommandGroup{
		Message:  i18n.T("Subcommands provided by plugins:"),
		Commands: registerPluginCommands(kubectl, false),
	}
}

// SetupPluginCompletion adds a Cobra command to the command tree for each
// plugin.  This is only done when performing shell completion that relate
// to plugins.
func SetupPluginCompletion(cmd *cobra.Command, args []string) {
	kubectl := cmd.Root()
	if len(args) > 0 {
		if strings.HasPrefix(args[0], "-") {
			// Plugins are not supported if the first argument is a flag,
			// so no need to add them in that case.
			return
		}

		if len(args) == 1 {
			// We are completing a subcommand at the first level so
			// we should include all plugins names.
			registerPluginCommands(kubectl, true)
			return
		}

		// We have more than one argument.
		// Check if we know the first level subcommand.
		// If we don't it could be a plugin and we'll need to add
		// the plugin commands for completion to work.
		found := false
		for _, subCmd := range kubectl.Commands() {
			if args[0] == subCmd.Name() {
				found = true
				break
			}
		}

		if !found {
			// We don't know the subcommand for which completion
			// is being called: it could be a plugin.
			//
			// When using a plugin, the kubectl global flags are not supported.
			// Therefore, when doing completion, we need to remove these flags
			// to avoid them being included in the completion choices.
			// This must be done *before* adding the plugin commands so that
			// when creating those plugin commands, the flags don't exist.
			kubectl.ResetFlags()
			cobra.CompDebugln("Cleared global flags for plugin completion", true)

			registerPluginCommands(kubectl, true)
		}
	}
}

// registerPluginCommand allows adding Cobra command to the command tree or extracting them for usage in
// e.g. the help function or for registering the completion function
func registerPluginCommands(kubectl *cobra.Command, list bool) (cmds []*cobra.Command) {
	userDefinedCommands := []*cobra.Command{}

	streams := genericclioptions.IOStreams{
		In:     &bytes.Buffer{},
		Out:    io.Discard,
		ErrOut: io.Discard,
	}

	o := &PluginListOptions{IOStreams: streams}
	o.Complete(kubectl)
	plugins, _ := o.ListPlugins()

	for _, plugin := range plugins {
		plugin = filepath.Base(plugin)
		args := []string{}

		// Plugins are named "kubectl-<name>" or with more - such as
		// "kubectl-<name>-<subcmd1>..."
		rawPluginArgs := strings.Split(plugin, "-")[1:]
		pluginArgs := rawPluginArgs[:1]
		if list {
			pluginArgs = rawPluginArgs
		}

		// Iterate through all segments, for kubectl-my_plugin-sub_cmd, we will end up with
		// two iterations: one for my_plugin and one for sub_cmd.
		for _, arg := range pluginArgs {
			// Underscores (_) in plugin's filename are replaced with dashes(-)
			// e.g. foo_bar -> foo-bar
			args = append(args, strings.ReplaceAll(arg, "_", "-"))
		}

		// In order to avoid that the same plugin command is added more than once,
		// find the lowest command given args from the root command
		parentCmd, remainingArgs, _ := kubectl.Find(args)
		if parentCmd == nil {
			parentCmd = kubectl
		}

		for _, remainingArg := range remainingArgs {
			cmd := &cobra.Command{
				Use: remainingArg,
				// Add a description that will be shown with completion choices.
				// Make each one different by including the plugin name to avoid
				// all plugins being grouped in a single line during completion for zsh.
				Short:              fmt.Sprintf(i18n.T("The command %s is a plugin installed by the user"), remainingArg),
				DisableFlagParsing: true,
				// Allow plugins to provide their own completion choices
				ValidArgsFunction: pluginCompletion,
				// A Run is required for it to be a valid command
				Run: func(cmd *cobra.Command, args []string) {},
			}
			// Add the plugin command to the list of user defined commands
			userDefinedCommands = append(userDefinedCommands, cmd)

			if list {
				parentCmd.AddCommand(cmd)
				parentCmd = cmd
			}
		}
	}

	return userDefinedCommands
}

// pluginCompletion deals with shell completion beyond the plugin name, it allows to complete
// plugin arguments and flags.
// It will look on $PATH for a specific executable file that will provide completions
// for the plugin in question.
//
// When called, this completion executable should print the completion choices to stdout.
// The arguments passed to the executable file will be the arguments for the plugin currently
// on the command-line.  For example, if a user types:
//
//	kubectl myplugin arg1 arg2 a<TAB>
//
// the completion executable will be called with arguments: "arg1" "arg2" "a".
// And if a user types:
//
//	kubectl myplugin arg1 arg2 <TAB>
//
// the completion executable will be called with arguments: "arg1" "arg2" "".  Notice the empty
// last argument which indicates that a new word should be completed but that the user has not
// typed anything for it yet.
//
// Kubectl's plugin completion logic supports Cobra's ShellCompDirective system.  This means a plugin
// can optionally print :<value of a shell completion directive> as its very last line to provide
// directives to the shell on how to perform completion.  If this directive is not present, the
// cobra.ShellCompDirectiveDefault will be used. Please see Cobra's documentation for more details:
// https://github.com/spf13/cobra/blob/master/shell_completions.md#dynamic-completion-of-nouns
//
// The completion executable should be named kubectl_complete-<plugin>.  For example, for a plugin
// named kubectl-get_all, the completion file should be named kubectl_complete-get_all.  The completion
// executable must have executable permissions set on it and must be on $PATH.
func pluginCompletion(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	// Recreate the plugin name from the commandPath
	pluginName := strings.ReplaceAll(strings.ReplaceAll(cmd.CommandPath(), "-", "_"), " ", "-")

	path, found := lookupCompletionExec(pluginName)
	if !found {
		cobra.CompDebugln(fmt.Sprintf("Plugin %s does not provide a matching completion executable", pluginName), true)
		return nil, cobra.ShellCompDirectiveDefault
	}

	args = append(args, toComplete)
	cobra.CompDebugln(fmt.Sprintf("About to call: %s %s", path, strings.Join(args, " ")), true)
	return getPluginCompletions(path, args, os.Environ())
}

// lookupCompletionExec will look for the existence of an executable
// that can provide completion for the given plugin name.
// The first filepath to match is returned, or a boolean false if
// such an executable is not found.
func lookupCompletionExec(pluginName string) (string, bool) {
	// Convert the plugin name into the plugin completion name by inserting "_complete" before the first -.
	// For example, convert kubectl-get_all to kubectl_complete-get_all
	pluginCompExec := strings.Replace(pluginName, "-", "_complete-", 1)
	cobra.CompDebugln(fmt.Sprintf("About to look for: %s", pluginCompExec), true)
	path, err := exec.LookPath(pluginCompExec)
	if err != nil || len(path) == 0 {
		return "", false
	}
	return path, true
}

// getPluginCompletions receives an executable's filepath, a slice
// of arguments, and a slice of environment variables
// to relay to the executable.
// The executable is responsible for printing the completions of the
// plugin for the current set of arguments.
func getPluginCompletions(executablePath string, cmdArgs, environment []string) ([]string, cobra.ShellCompDirective) {
	buf := new(bytes.Buffer)

	prog := exec.Command(executablePath, cmdArgs...)
	prog.Stdin = os.Stdin
	prog.Stdout = buf
	prog.Stderr = os.Stderr
	prog.Env = environment

	var comps []string
	directive := cobra.ShellCompDirectiveDefault
	if err := prog.Run(); err == nil {
		for _, comp := range strings.Split(buf.String(), "\n") {
			// Remove any empty lines
			if len(comp) > 0 {
				comps = append(comps, comp)
			}
		}

		// Check if the last line of output is of the form :<integer>, which
		// indicates a Cobra ShellCompDirective.  We do this for plugins
		// that use Cobra or the ones that wish to use this directive to
		// communicate a special behavior for the shell.
		if len(comps) > 0 {
			lastLine := comps[len(comps)-1]
			if len(lastLine) > 1 && lastLine[0] == ':' {
				if strInt, err := strconv.Atoi(lastLine[1:]); err == nil {
					directive = cobra.ShellCompDirective(strInt)
					comps = comps[:len(comps)-1]
				}
			}
		}
	}
	return comps, directive
}
