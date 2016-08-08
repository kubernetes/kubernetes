/*
Copyright 2016 The Kubernetes Authors.

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
	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	"io"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

// TopOptions contains all the options for running the top cli command.
type TopOptions struct{}

var (
	topLong = dedent.Dedent(`
		Display Resource (CPU/Memory/Storage) usage.

		The top command allows you to see the resource consumption of the nodes or pods.
		It downloads the usage metrics of a given resource via the Resource Metrics API.`)
)

func NewCmdTop(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &TopOptions{}

	cmd := &cobra.Command{
		Use:   "top",
		Short: "Display Resource (CPU/Memory/Storage) usage",
		Long:  topLong,
		Run: func(cmd *cobra.Command, args []string) {
			if err := options.RunTop(f, cmd, args, out); err != nil {
				cmdutil.CheckErr(err)
			}
		},
	}
	cmdutil.AddInclude3rdPartyFlags(cmd)

	// create subcommands
	cmd.AddCommand(NewCmdTopNode(f, out))
	cmd.AddCommand(NewCmdTopPod(f, out))
	return cmd
}

// RunTop implements all the necessary functionality for top.
func (o TopOptions) RunTop(f *cmdutil.Factory, cmd *cobra.Command, args []string, out io.Writer) error {
	return cmdutil.UsageError(cmd, "Required resource not specified.")
}
