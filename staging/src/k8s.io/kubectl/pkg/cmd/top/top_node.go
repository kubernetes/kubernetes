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

package top

import (
	"context"
	"errors"
	"fmt"

	"github.com/spf13/cobra"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/discovery"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/metricsutil"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
	metricsV1api "k8s.io/metrics/pkg/apis/metrics/v1"
	metricsV1beta1api "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsclientset "k8s.io/metrics/pkg/client/clientset/versioned"
)

// TopNodeFlags directly reflect the information that CLI is gathering via flags.
type TopNodeFlags struct {
	Selector           string
	SortBy             string
	NoHeaders          bool
	UseProtocolBuffers bool
	ShowCapacity       bool
	ShowSwap           bool

	genericiooptions.IOStreams
}

// TopNodeOptions contains all the options for running the top-node cli command.
type TopNodeOptions struct {
	ResourceName string
	Selector     string
	SortBy       string
	NoHeaders    bool
	ShowCapacity bool
	ShowSwap     bool

	NodeClient      corev1client.CoreV1Interface
	Printer         *metricsutil.TopCmdPrinter
	DiscoveryClient discovery.DiscoveryInterface
	MetricsClient   metricsclientset.Interface

	genericiooptions.IOStreams
}

var (
	topNodeLong = templates.LongDesc(i18n.T(`
		Display resource (CPU/memory) usage of nodes.

		The top-node command allows you to see the resource consumption of nodes.`))

	topNodeExample = templates.Examples(i18n.T(`
		  # Show metrics for all nodes
		  kubectl top node

		  # Show metrics for a given node
		  kubectl top node NODE_NAME`))
)

func NewTopNodeFlags(streams genericiooptions.IOStreams) *TopNodeFlags {
	return &TopNodeFlags{
		IOStreams:          streams,
		UseProtocolBuffers: true,
	}
}

func NewCmdTopNode(f cmdutil.Factory, flags *TopNodeFlags, streams genericiooptions.IOStreams) *cobra.Command {
	if flags == nil {
		flags = NewTopNodeFlags(streams)
	}

	cmd := &cobra.Command{
		Use:                   "node [NAME | -l label]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Display resource (CPU/memory) usage of nodes"),
		Long:                  topNodeLong,
		Example:               topNodeExample,
		ValidArgsFunction:     completion.ResourceNameCompletionFunc(f, "node"),
		Run: func(cmd *cobra.Command, args []string) {
			o, err := flags.ToOptions(f, args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunTopNode())
		},
		Aliases: []string{"nodes", "no"},
	}
	flags.AddFlags(cmd)

	return cmd
}

// AddFlags registers flags for a cli
func (flags *TopNodeFlags) AddFlags(cmd *cobra.Command) {
	cmdutil.AddLabelSelectorFlagVar(cmd, &flags.Selector)
	cmd.Flags().StringVar(&flags.SortBy, "sort-by", flags.SortBy, "If non-empty, sort nodes list using specified field. The field can be either 'cpu' or 'memory'.")
	cmd.Flags().BoolVar(&flags.NoHeaders, "no-headers", flags.NoHeaders, "If present, print output without headers")
	cmd.Flags().BoolVar(&flags.UseProtocolBuffers, "use-protocol-buffers", flags.UseProtocolBuffers, "Enables using protocol-buffers to access Metrics API.")
	cmd.Flags().BoolVar(&flags.ShowCapacity, "show-capacity", flags.ShowCapacity, "Print node resources based on Capacity instead of Allocatable(default) of the nodes.")
	cmd.Flags().BoolVar(&flags.ShowSwap, "show-swap", flags.ShowSwap, "Print node resources related to swap memory.")
}

// ToOptions converts from CLI inputs to runtime inputs
func (flags *TopNodeFlags) ToOptions(f cmdutil.Factory, args []string) (*TopNodeOptions, error) {
	o := &TopNodeOptions{
		Selector:     flags.Selector,
		SortBy:       flags.SortBy,
		NoHeaders:    flags.NoHeaders,
		ShowCapacity: flags.ShowCapacity,
		ShowSwap:     flags.ShowSwap,
		IOStreams:    flags.IOStreams,
	}

	if len(args) == 1 {
		o.ResourceName = args[0]
	} else if len(args) > 1 {
		return nil, fmt.Errorf("unexpected arguments: %v", args[1:])
	}

	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return nil, err
	}

	o.DiscoveryClient = clientset.DiscoveryClient

	config, err := f.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	if flags.UseProtocolBuffers {
		config.ContentType = "application/vnd.kubernetes.protobuf"
	}
	o.MetricsClient, err = metricsclientset.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	o.NodeClient = clientset.CoreV1()

	o.Printer = metricsutil.NewTopCmdPrinter(o.Out, o.ShowSwap)
	return o, nil
}

func (o *TopNodeOptions) Validate() error {
	if len(o.SortBy) > 0 {
		if o.SortBy != sortByCPU && o.SortBy != sortByMemory {
			return errors.New("--sort-by accepts only cpu or memory")
		}
	}
	if len(o.ResourceName) > 0 && len(o.Selector) > 0 {
		return errors.New("only one of NAME or --selector can be provided")
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

	apiGroups, err := o.DiscoveryClient.ServerGroups()
	if err != nil {
		return err
	}

	metricsAPIAvailable := SupportedMetricsAPIVersionAvailable(apiGroups)

	if metricsAPIAvailable == "" {
		return errors.New("Metrics API not available")
	}

	metrics, err := getNodeMetricsFromMetricsAPI(o.MetricsClient, o.ResourceName, selector, metricsAPIAvailable)
	if err != nil {
		return err
	}

	if len(metrics.Items) == 0 {
		return errors.New("metrics not available yet")
	}

	var nodes []v1.Node
	if len(o.ResourceName) > 0 {
		node, err := o.NodeClient.Nodes().Get(context.TODO(), o.ResourceName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		nodes = append(nodes, *node)
	} else {
		nodeList, err := o.NodeClient.Nodes().List(context.TODO(), metav1.ListOptions{
			LabelSelector: selector.String(),
		})
		if err != nil {
			return err
		}
		nodes = append(nodes, nodeList.Items...)
	}

	availableResources := make(map[string]v1.ResourceList)

	for _, n := range nodes {
		if !o.ShowCapacity {
			availableResources[n.Name] = n.Status.Allocatable
		} else {
			availableResources[n.Name] = n.Status.Capacity
		}

		if n.Status.NodeInfo.Swap != nil && n.Status.NodeInfo.Swap.Capacity != nil {
			swapCapacity := *n.Status.NodeInfo.Swap.Capacity
			availableResources[n.Name]["swap"] = *resource.NewQuantity(swapCapacity, resource.BinarySI)
		} else {
			o.Printer.RegisterMissingResource(n.Name, metricsutil.ResourceSwap)
		}

	}

	return o.Printer.PrintNodeMetrics(metrics.Items, availableResources, o.NoHeaders, o.SortBy)
}

func getNodeMetricsFromMetricsAPI(metricsClient metricsclientset.Interface, resourceName string, selector labels.Selector, metricsVersion string) (*metricsapi.NodeMetricsList, error) {
	var err error
	if metricsVersion == "v1" {
		versionedMetrics := &metricsV1api.NodeMetricsList{}
		mc := metricsClient.MetricsV1()
		nm := mc.NodeMetricses()
		if resourceName != "" {
			m, err := nm.Get(context.TODO(), resourceName, metav1.GetOptions{})
			if err != nil {
				return nil, err
			}
			versionedMetrics.Items = []metricsV1api.NodeMetrics{*m}
		} else {
			versionedMetrics, err = nm.List(context.TODO(), metav1.ListOptions{LabelSelector: selector.String()})
			if err != nil {
				return nil, err
			}
		}
		metrics := &metricsapi.NodeMetricsList{}
		err = metricsV1api.Convert_v1_NodeMetricsList_To_metrics_NodeMetricsList(versionedMetrics, metrics, nil)
		if err != nil {
			return nil, err
		}
		return metrics, nil
	}
	// fallback to metric v1beta1
	versionedMetrics := &metricsV1beta1api.NodeMetricsList{}
	mc := metricsClient.MetricsV1beta1()
	nm := mc.NodeMetricses()
	if resourceName != "" {
		m, err := nm.Get(context.TODO(), resourceName, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		versionedMetrics.Items = []metricsV1beta1api.NodeMetrics{*m}
	} else {
		versionedMetrics, err = nm.List(context.TODO(), metav1.ListOptions{LabelSelector: selector.String()})
		if err != nil {
			return nil, err
		}
	}
	metrics := &metricsapi.NodeMetricsList{}
	err = metricsV1beta1api.Convert_v1beta1_NodeMetricsList_To_metrics_NodeMetricsList(versionedMetrics, metrics, nil)
	if err != nil {
		return nil, err
	}
	return metrics, nil
}
