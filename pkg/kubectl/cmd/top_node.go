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

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/metricsutil"
)

// TopNodeOptions contains all the options for running the top-node cli command.
type TopNodeOptions struct {
	ResourceName    string
	Selector        string
	NodeClient      coreclient.NodesGetter
	HeapsterOptions HeapsterTopOptions
	Client          *metricsutil.HeapsterMetricsClient
	Printer         *metricsutil.TopCmdPrinter
}

type HeapsterTopOptions struct {
	Namespace string
	Service   string
	Scheme    string
	Port      string
}

func (o *HeapsterTopOptions) Bind(flags *pflag.FlagSet) {
	flags.StringVar(&o.Namespace, "heapster-namespace", metricsutil.DefaultHeapsterNamespace, "Namespace Heapster service is located in")
	flags.StringVar(&o.Service, "heapster-service", metricsutil.DefaultHeapsterService, "Name of Heapster service")
	flags.StringVar(&o.Scheme, "heapster-scheme", metricsutil.DefaultHeapsterScheme, "Scheme (http or https) to connect to Heapster as")
	flags.StringVar(&o.Port, "heapster-port", metricsutil.DefaultHeapsterPort, "Port name in service to use")
}

var (
	topNodeLong = templates.LongDesc(`
		Display Resource (CPU/Memory/Storage) usage of nodes.

		The top-node command allows you to see the resource consumption of nodes.`)

	topNodeExample = templates.Examples(`
		  # Show metrics for all nodes
		  kubectl top node

		  # Show metrics for a given node
		  kubectl top node NODE_NAME`)
)

func NewCmdTopNode(f cmdutil.Factory, out io.Writer) *cobra.Command {
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
		Aliases: []string{"nodes", "no"},
	}
	cmd.Flags().StringVarP(&options.Selector, "selector", "l", "", "Selector (label query) to filter on, supports '=', '==', and '!='.")
	options.HeapsterOptions.Bind(cmd.Flags())
	return cmd
}

func (o *TopNodeOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string, out io.Writer) error {
	var err error
	if len(args) == 1 {
		o.ResourceName = args[0]
	} else if len(args) > 1 {
		return cmdutil.UsageError(cmd, cmd.Use)
	}

	clientset, err := f.ClientSet()
	if err != nil {
		return err
	}
	o.NodeClient = clientset.Core()
	o.Client = metricsutil.NewHeapsterMetricsClient(clientset.Core(), o.HeapsterOptions.Namespace, o.HeapsterOptions.Scheme, o.HeapsterOptions.Service, o.HeapsterOptions.Port)
	o.Printer = metricsutil.NewTopCmdPrinter(out)
	return nil
}

func (o *TopNodeOptions) Validate() error {
	if len(o.ResourceName) > 0 && len(o.Selector) > 0 {
		return errors.New("only one of NAME or --selector can be provided")
	}
	if len(o.Selector) > 0 {
		_, err := labels.Parse(o.Selector)
		if err != nil {
			return err
		}
	}
	return nil
}

func (o TopNodeOptions) RunTopNode() error {
	var err error
	selector := labels.Everything()
	if len(o.Selector) > 0 {
		selector, err = labels.Parse(o.Selector)
		if err != nil {
			return err
		}
	}
	metrics, err := o.Client.GetNodeMetrics(o.ResourceName, selector)
	if err != nil {
		return err
	}
	if len(metrics) == 0 {
		return errors.New("metrics not available yet")
	}

	var nodes []api.Node
	if len(o.ResourceName) > 0 {
		node, err := o.NodeClient.Nodes().Get(o.ResourceName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		nodes = append(nodes, *node)
	} else {
		nodeList, err := o.NodeClient.Nodes().List(api.ListOptions{
			LabelSelector: selector,
		})
		if err != nil {
			return err
		}
		nodes = append(nodes, nodeList.Items...)
	}

	allocatable := make(map[string]api.ResourceList)

	for _, n := range nodes {
		allocatable[n.Name] = n.Status.Allocatable
	}

	return o.Printer.PrintNodeMetrics(metrics, allocatable)
}
