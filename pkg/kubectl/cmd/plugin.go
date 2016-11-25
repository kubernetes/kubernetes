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
	"io"
	"os"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/plugins"
)

// NewCmdForPlugin creates a command capable of running the provided plugin.
func NewCmdForPlugin(plugin *plugins.Plugin, runner plugins.PluginRunner, in io.Reader, out, errout io.Writer) *cobra.Command {
	if !plugin.IsValid() {
		return nil
	}

	return &cobra.Command{
		Use:     plugin.Name,
		Short:   plugin.ShortDesc,
		Long:    templates.LongDesc(plugin.LongDesc),
		Example: templates.Examples(plugin.Example),
		Run: func(cmd *cobra.Command, args []string) {
			ctx := plugins.RunningContext{
				In:         in,
				Out:        out,
				ErrOut:     errout,
				Args:       args,
				Env:        os.Environ(),
				WorkingDir: plugin.Dir,
			}
			if err := runner.Run(plugin, ctx); err != nil {
				cmdutil.CheckErr(err)
			}
		},
	}
}
