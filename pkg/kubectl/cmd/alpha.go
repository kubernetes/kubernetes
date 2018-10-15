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
	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
)

// NewCmdAlpha creates a command that acts as an alternate root command for features in alpha
func NewCmdAlpha(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "alpha",
		Short: i18n.T("Commands for features in alpha"),
		Long:  templates.LongDesc(i18n.T("These commands correspond to alpha features that are not enabled in Kubernetes clusters by default.")),
	}

	// Alpha commands should be added here. As features graduate from alpha they should move
	// from here to the CommandGroups defined by NewKubeletCommand() in cmd.go.
	//cmd.AddCommand(NewCmdDebug(f, in, out, err))

	// NewKubeletCommand() will hide the alpha command if it has no subcommands. Overriding
	// the help function ensures a reasonable message if someone types the hidden command anyway.
	if !cmd.HasSubCommands() {
		cmd.SetHelpFunc(func(*cobra.Command, []string) {
			cmd.Println(i18n.T("No alpha commands are available in this version of kubectl"))
		})
	}

	return cmd
}
