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
	"errors"
	"io"

	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/metricsutil"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
)

// TopNodeOptions contains all the options for running the top-node cli command.
type TopNodeOptions struct {
	ResourceName string
	Selector     string
	Client       *metricsutil.HeapsterMetricsClient
	Printer      *metricsutil.TopCmdPrinter
}

var (
	topNodeLong = dedent.Dedent(`
		Display Resource (CPU/Memory/Storage) usage of nodes.

		The top-node command allows you to see the resource consumption of nodes.`)

	topNodeExample = dedent.Dedent(`
		  # Show metrics for all nodes
		  kubectl top node

		  # Show metrics for a given node
		  kubectl top node NODE_NAME`)
)

func NewCmdTopNode(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &TopNodeOptions{}

	cmd := &cobra.Command{
		Use:     "node [NAME | -l label]",
		Short:   "Display Resource (CPU/Memory/Storage) usage of nodes",
		Long:    topNodeLong,
		Example: topNodeExample,
		Run: func(cmd *cobra.Command, args []string) {
			if err := options.Complete(f, cmd, args, out); err != nil {
				cmdutil.CheckErr(err)
			}
			if err := options.Validate(); err != nil {
				cmdutil.CheckErr(cmdutil.UsageError(cmd, err.Error()))
			}
			if err := options.RunTopNode(); err != nil {
				cmdutil.CheckErr(err)
			}
		},
		Aliases: []string{"nodes"},
	}
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on")
	return cmd
}

func (o *TopNodeOptions) Complete(f *cmdutil.Factory, cmd *cobra.Command, args []string, out io.Writer) error {
	var err error
	if len(args) == 1 {
		o.ResourceName = args[0]
	} else if len(args) > 1 {
		return cmdutil.UsageError(cmd, cmd.Use)
	}

	cli, err := f.Client()
	if err != nil {
		return err
	}
	o.Client = metricsutil.DefaultHeapsterMetricsClient(cli)
	o.Printer = metricsutil.NewTopCmdPrinter(out)
	return nil
}

func (o *TopNodeOptions) Validate() error {
	if len(o.ResourceName) > 0 && len(o.Selector) > 0 {
		return errors.New("only one of NAME or --selector can be provided")
	}
	return nil
}

func (o TopNodeOptions) RunTopNode() error {
	metrics, err := o.Client.GetNodeMetrics(o.ResourceName, o.Selector)
	if err != nil {
		return err
	}
	return o.Printer.PrintNodeMetrics(metrics)
}
