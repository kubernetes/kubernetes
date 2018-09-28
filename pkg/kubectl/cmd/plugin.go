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

package cmd

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	plugin_long = templates.LongDesc(`
		Provides utilities for interacting with plugins.

		Plugins provide extended functionality that is not part of the major command-line distribution.
		Please refer to the documentation and examples for more information about how write your own plugins.`)

	plugin_list_long = templates.LongDesc(`
		List all available plugin files on a user's PATH.

		Available plugin files are those that are:
		- executable
		- anywhere on the user's PATH
		- begin with "kubectl-"
`)
)

func NewCmdPlugin(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use: "plugin [flags]",
		DisableFlagsInUseLine: true,
		Short: i18n.T("Provides utilities for interacting with plugins."),
		Long:  plugin_long,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.DefaultSubCommandRun(streams.ErrOut)(cmd, args)
		},
	}

	cmd.AddCommand(NewCmdPluginList(f, streams))
	return cmd
}

type PluginListOptions struct {
	Verifier PathVerifier
	NameOnly bool

	genericclioptions.IOStreams
}

// NewCmdPluginList provides a way to list all plugin executables visible to kubectl
func NewCmdPluginList(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := &PluginListOptions{
		IOStreams: streams,
	}

	cmd := &cobra.Command{
		Use:   "list",
		Short: "list all visible plugin executables on a user's PATH",
		Long:  plugin_list_long,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(cmd))
			cmdutil.CheckErr(o.Run())
		},
	}

	cmd.Flags().BoolVar(&o.NameOnly, "name-only", o.NameOnly, "If true, display only the binary name of each plugin, rather than its full path")
	return cmd
}

func (o *PluginListOptions) Complete(cmd *cobra.Command) error {
	o.Verifier = &CommandOverrideVerifier{
		root:        cmd.Root(),
		seenPlugins: make(map[string]string, 0),
	}
	return nil
}

func (o *PluginListOptions) Run() error {
	path := "PATH"
	if runtime.GOOS == "windows" {
		path = "path"
	}

	pluginsFound := false
	isFirstFile := true
	pluginWarnings := 0
	paths := sets.NewString(filepath.SplitList(os.Getenv(path))...)
	for _, dir := range paths.List() {
		files, err := ioutil.ReadDir(dir)
		if err != nil {
			continue
		}

		for _, f := range files {
			if f.IsDir() {
				continue
			}
			if !strings.HasPrefix(f.Name(), "kubectl-") {
				continue
			}

			if isFirstFile {
				fmt.Fprintf(o.ErrOut, "The following kubectl-compatible plugins are available:\n\n")
				pluginsFound = true
				isFirstFile = false
			}

			pluginPath := f.Name()
			if !o.NameOnly {
				pluginPath = filepath.Join(dir, pluginPath)
			}

			fmt.Fprintf(o.Out, "%s\n", pluginPath)
			if errs := o.Verifier.Verify(filepath.Join(dir, f.Name())); len(errs) != 0 {
				for _, err := range errs {
					fmt.Fprintf(o.ErrOut, "  - %s\n", err)
					pluginWarnings++
				}
			}
		}
	}

	if !pluginsFound {
		return fmt.Errorf("error: unable to find any kubectl plugins in your PATH")
	}

	if pluginWarnings > 0 {
		fmt.Fprintln(o.ErrOut)
		if pluginWarnings == 1 {
			return fmt.Errorf("one plugin warning was found")
		}
		return fmt.Errorf("%v plugin warnings were found", pluginWarnings)
	}

	return nil
}

// pathVerifier receives a path and determines if it is valid or not
type PathVerifier interface {
	// Verify determines if a given path is valid
	Verify(path string) []error
}

type CommandOverrideVerifier struct {
	root        *cobra.Command
	seenPlugins map[string]string
}

// Verify implements PathVerifier and determines if a given path
// is valid depending on whether or not it overwrites an existing
// kubectl command path, or a previously seen plugin.
func (v *CommandOverrideVerifier) Verify(path string) []error {
	if v.root == nil {
		return []error{fmt.Errorf("unable to verify path with nil root")}
	}

	// extract the plugin binary name
	segs := strings.Split(path, "/")
	binName := segs[len(segs)-1]

	cmdPath := strings.Split(binName, "-")
	if len(cmdPath) > 1 {
		// the first argument is always "kubectl" for a plugin binary
		cmdPath = cmdPath[1:]
	}

	errors := []error{}

	if isExec, err := isExecutable(path); err == nil && !isExec {
		errors = append(errors, fmt.Errorf("warning: %s identified as a kubectl plugin, but it is not executable", path))
	} else if err != nil {
		errors = append(errors, fmt.Errorf("error: unable to identify %s as an executable file: %v", path, err))
	}

	if existingPath, ok := v.seenPlugins[binName]; ok {
		errors = append(errors, fmt.Errorf("warning: %s is overshadowed by a similarly named plugin: %s", path, existingPath))
	} else {
		v.seenPlugins[binName] = path
	}

	if cmd, _, err := v.root.Find(cmdPath); err == nil {
		errors = append(errors, fmt.Errorf("warning: %s overwrites existing command: %q", binName, cmd.CommandPath()))
	}

	return errors
}

func isExecutable(fullPath string) (bool, error) {
	info, err := os.Stat(fullPath)
	if err != nil {
		return false, err
	}

	if runtime.GOOS == "windows" {
		if strings.HasSuffix(info.Name(), ".exe") {
			return true, nil
		}
		return false, nil
	}

	if m := info.Mode(); !m.IsDir() && m&0111 != 0 {
		return true, nil
	}

	return false, nil
}
