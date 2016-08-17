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

// TopPodOptions contains all the options for running the top-pod cli command.
type TopPodOptions struct {
	ResourceName    string
	AllNamespaces   bool
	PrintContainers bool
	Selector        string
	Namespace       string
	Client          *metricsutil.HeapsterMetricsClient
	Printer         *metricsutil.TopCmdPrinter
}

var (
	topPodLong = dedent.Dedent(`
		Display Resource (CPU/Memory/Storage) usage of pods.

		The 'top pod' command allows you to see the resource consumption of pods.`)

	topPodExample = dedent.Dedent(`
		  # Show metrics for all pods in the default namespace
		  kubectl top pod

		  # Show metrics for all pods in the given namespace
		  kubectl top pod --namespace=NAMESPACE

		  # Show metrics for a given pod and its containers
		  kubectl top pod POD_NAME --containers

		  # Show metrics for the pods defined by label name=myLabel
		  kubectl top pod -l name=myLabel`)
)

func NewCmdTopPod(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &TopPodOptions{}

	cmd := &cobra.Command{
		Use:     "pod [NAME | -l label]",
		Short:   "Display Resource (CPU/Memory/Storage) usage of pods",
		Long:    topPodLong,
		Example: topPodExample,
		Run: func(cmd *cobra.Command, args []string) {
			if err := options.Complete(f, cmd, args, out); err != nil {
				cmdutil.CheckErr(err)
			}
			if err := options.RunTopPod(); err != nil {
				cmdutil.CheckErr(err)
			}
		},
		Aliases: []string{"pods"},
	}
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().BoolVar(&options.PrintContainers, "containers", false, "If present, print usage of containers within a pod.")
	cmd.Flags().BoolVar(&options.AllNamespaces, "all-namespaces", false, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	return cmd
}

// Complete completes all the required options for top.
func (o *TopPodOptions) Complete(f *cmdutil.Factory, cmd *cobra.Command, args []string, out io.Writer) error {
	var err error
	if len(args) == 1 {
		o.ResourceName = args[0]
	} else if len(args) > 1 {
		return cmdutil.UsageError(cmd, cmd.Use)
	}

	o.Namespace, _, err = f.DefaultNamespace()
	if err != nil {
		return err
	}
	cli, err := f.Client()
	if err != nil {
		return err
	}
	o.Client = metricsutil.DefaultHeapsterMetricsClient(cli)
	o.Printer = metricsutil.NewTopCmdPrinter(out)
	return nil
}

func (o *TopPodOptions) Validate() error {
	if len(o.ResourceName) > 0 && len(o.Selector) > 0 {
		return errors.New("only one of NAME or --selector can be provided")
	}
	return nil
}

// RunTop implements all the necessary functionality for top.
func (o TopPodOptions) RunTopPod() error {
	metrics, err := o.Client.GetPodMetrics(o.Namespace, o.ResourceName, o.AllNamespaces, o.Selector)
	if err != nil {
		return err
	}
	return o.Printer.PrintPodMetrics(metrics, o.PrintContainers, o.AllNamespaces)
}
