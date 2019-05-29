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
	"errors"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/discovery"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/metricsutil"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
	"k8s.io/kubernetes/pkg/kubectl/util/templates"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
	metricsV1beta1api "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsclientset "k8s.io/metrics/pkg/client/clientset/versioned"
)

// TopNodeOptions contains all the options for running the top-node cli command.
type TopNodeOptions struct {
	ResourceName    string
	Selector        string
	NoHeaders       bool
	NodeClient      corev1client.CoreV1Interface
	HeapsterOptions HeapsterTopOptions
	Client          *metricsutil.HeapsterMetricsClient
	Printer         *metricsutil.TopCmdPrinter
	DiscoveryClient discovery.DiscoveryInterface
	MetricsClient   metricsclientset.Interface

	genericclioptions.IOStreams
}

type HeapsterTopOptions struct {
	Namespace string
	Service   string
	Scheme    string
	Port      string
}

func (o *HeapsterTopOptions) Bind(flags *pflag.FlagSet) {
	if len(o.Namespace) == 0 {
		o.Namespace = metricsutil.DefaultHeapsterNamespace
	}
	if len(o.Service) == 0 {
		o.Service = metricsutil.DefaultHeapsterService
	}
	if len(o.Scheme) == 0 {
		o.Scheme = metricsutil.DefaultHeapsterScheme
	}
	if len(o.Port) == 0 {
		o.Port = metricsutil.DefaultHeapsterPort
	}

	flags.StringVar(&o.Namespace, "heapster-namespace", o.Namespace, "Namespace Heapster service is located in")
	flags.StringVar(&o.Service, "heapster-service", o.Service, "Name of Heapster service")
	flags.StringVar(&o.Scheme, "heapster-scheme", o.Scheme, "Scheme (http or https) to connect to Heapster as")
	flags.StringVar(&o.Port, "heapster-port", o.Port, "Port name in service to use")
}

var (
	topNodeLong = templates.LongDesc(i18n.T(`
		Display Resource (CPU/Memory/Storage) usage of nodes.

		The top-node command allows you to see the resource consumption of nodes.`))

	topNodeExample = templates.Examples(i18n.T(`
		  # Show metrics for all nodes
		  kubectl top node

		  # Show metrics for a given node
		  kubectl top node NODE_NAME`))
)

func NewCmdTopNode(f cmdutil.Factory, o *TopNodeOptions, streams genericclioptions.IOStreams) *cobra.Command {
	if o == nil {
		o = &TopNodeOptions{
			IOStreams: streams,
		}
	}

	cmd := &cobra.Command{
		Use:                   "node [NAME | -l label]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Display Resource (CPU/Memory/Storage) usage of nodes"),
		Long:                  topNodeLong,
		Example:               topNodeExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunTopNode())
		},
		Aliases: []string{"nodes", "no"},
	}
	cmd.Flags().StringVarP(&o.Selector, "selector", "l", o.Selector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().BoolVar(&o.NoHeaders, "no-headers", o.NoHeaders, "If present, print output without headers")

	o.HeapsterOptions.Bind(cmd.Flags())
	return cmd
}

func (o *TopNodeOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(args) == 1 {
		o.ResourceName = args[0]
	} else if len(args) > 1 {
		return cmdutil.UsageErrorf(cmd, "%s", cmd.Use)
	}

	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}

	o.DiscoveryClient = clientset.DiscoveryClient

	config, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	o.MetricsClient, err = metricsclientset.NewForConfig(config)
	if err != nil {
		return err
	}

	o.NodeClient = clientset.CoreV1()
	o.Client = metricsutil.NewHeapsterMetricsClient(clientset.CoreV1(), o.HeapsterOptions.Namespace, o.HeapsterOptions.Scheme, o.HeapsterOptions.Service, o.HeapsterOptions.Port)

	o.Printer = metricsutil.NewTopCmdPrinter(o.Out)
	return nil
}

func (o *TopNodeOptions) Validate() error {
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

	metrics := &metricsapi.NodeMetricsList{}
	if metricsAPIAvailable {
		metrics, err = getNodeMetricsFromMetricsAPI(o.MetricsClient, o.ResourceName, selector)
		if err != nil {
			return err
		}
	} else {
		metrics, err = o.Client.GetNodeMetrics(o.ResourceName, selector.String())
		if err != nil {
			return err
		}
	}

	if len(metrics.Items) == 0 {
		return errors.New("metrics not available yet")
	}

	var nodes []v1.Node
	if len(o.ResourceName) > 0 {
		node, err := o.NodeClient.Nodes().Get(o.ResourceName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		nodes = append(nodes, *node)
	} else {
		nodeList, err := o.NodeClient.Nodes().List(metav1.ListOptions{
			LabelSelector: selector.String(),
		})
		if err != nil {
			return err
		}
		nodes = append(nodes, nodeList.Items...)
	}

	allocatable := make(map[string]v1.ResourceList)

	for _, n := range nodes {
		allocatable[n.Name] = n.Status.Allocatable
	}

	return o.Printer.PrintNodeMetrics(metrics.Items, allocatable, o.NoHeaders)
}

func getNodeMetricsFromMetricsAPI(metricsClient metricsclientset.Interface, resourceName string, selector labels.Selector) (*metricsapi.NodeMetricsList, error) {
	var err error
	versionedMetrics := &metricsV1beta1api.NodeMetricsList{}
	mc := metricsClient.MetricsV1beta1()
	nm := mc.NodeMetricses()
	if resourceName != "" {
		m, err := nm.Get(resourceName, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		versionedMetrics.Items = []metricsV1beta1api.NodeMetrics{*m}
	} else {
		versionedMetrics, err = nm.List(metav1.ListOptions{LabelSelector: selector.String()})
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
