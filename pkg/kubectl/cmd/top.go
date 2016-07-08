/*
Copyright 2014 The Kubernetes Authors.

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
	"io"
	"encoding/json"
	"strings"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"
	//"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	//"k8s.io/kubernetes/pkg/kubectl/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
	//"k8s.io/kubernetes/pkg/runtime"
	//utilerrors "k8s.io/kubernetes/pkg/util/errors"
	//"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/pkg/api/v1"
	//"github.com/vmware/govmomi/object"
	//"k8s.io/kubernetes/pkg/api"
	//"k8s.io/kubernetes/third_party/golang/go/doc/testdata"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"errors"
)

// TopOptions is the start of the data required to perform the operation. As new fields are added, add them here instead of
// referencing the cmd.Flags()
type TopOptions struct {
}

var (
	topLong = dedent.Dedent(`
		Display Resource (CPU/Memory/Storage) usage of nodes or pods.

		The top command allows you to see the resource consumption of the nodes or pods.
		It downloads the usage metrics of a given resource (node/pod) via the Resource Metrics API.
		`)

	topExample = dedent.Dedent(`
		  # Show metrics for all nodes in the default namespace
		  kubectl top node

		  # Show metrics for a given pod in the default namespace
		  kubectl top pod POD_NAME

		  # Show metrics for the pods defined by the selector query
		  kubectl top pod --selector="key: value"`)
)

var HandledResources []unversioned.GroupKind = []unversioned.GroupKind{
	api.Kind("Pod"),
	api.Kind("Node"),
};

var GetResourcesHandledByTop = func() []string {
	keys := make([]string, 0)
	for _, k := range HandledResources {
		resource := strings.ToLower(k.Kind)
		keys = append(keys, resource)
	}
	return keys
}

func NewCmdTop(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	options := &TopOptions{}

	// retrieve a list of handled resources as valid args
	validArgs := GetResourcesHandledByTop()
	argAliases := kubectl.ResourceAliases(validArgs)

	cmd := &cobra.Command{
		Use:     "top TYPE [NAME] [flags]",
		Short:   "Display Resource (CPU/Memory/Storage) usage of nodes or pods",
		Long:    topLong,
		Example: topExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunTop(f, out, cmd, args, options)
			cmdutil.CheckErr(err)
		},
		ValidArgs:  validArgs,
		ArgAliases: argAliases,
	}
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	cmd.Flags().Bool("all-namespaces", false, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

var GetResourceKind = func(resourceType string) (unversioned.GroupKind, error) {
	// TODO
	switch resourceType {
	case "node", "nodes":
		return api.Kind("Node"), nil
	case "pod", "pods":
		return api.Kind("Pod"), nil
	}
	return unversioned.GroupKind{}, errors.New("Unknown resource requested.")
}

func RunTop(f *cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string, options *TopOptions) error {

	selector := cmdutil.GetFlagString(cmd, "selector")
	//allNamespaces := cmdutil.GetFlagBool(cmd, "all-namespaces")

	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}

	if len(args) == 0 {
		fmt.Fprint(out, "You must specify the type of resource to get.")
		return cmdutil.UsageError(cmd, "Required resource not specified.")
	}

	cli, err := f.Client()
	if err != nil {
		return err
	}
	resType, err := GetResourceKind(args[0])
	if err != nil {
		return err
	}
	params := map[string]string{"labelSelector": selector}



	switch resType {
	case api.Kind("Node"):
		PrintNodeMetrics(out, DefaultHeapsterMetricsClient(cli), params, "")
	case api.Kind("Pod"):
		PrintPodMetrics(out, DefaultHeapsterMetricsClient(cli), params, cmdNamespace, "")
	}

	return nil
}

func GetNodeMetricsUrl(nodeName string) string {
	return fmt.Sprintf("%s/nodes/%s", MetricsRoot, nodeName)
}

var MeasuredResources = map[v1.ResourceName]string {
	v1.ResourceCPU: "CPU",
	v1.ResourceMemory: "Memory",
	v1.ResourceStorage: "Storage",
}

type ResourceMetrics struct{
	metrics	map[v1.ResourceName]resource.Quantity
}

func PrintNodeMetrics(out io.Writer, cli *HeapsterMetricsClient, params map[string]string, nodeName string) error {
	resultRaw, err := GetMetrics(cli, GetNodeMetricsUrl(nodeName), params)
	if err != nil {
		return err
	}

	metrics := make([]metrics_api.NodeMetrics, 0)
	err = json.Unmarshal(resultRaw, &metrics)
	if err != nil {
		fmt.Errorf("failed to unmarshall heapster response: %v", err)
		return err
	}

	w := kubectl.GetNewTabWriter(out)
	defer w.Flush()
	fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", "NAME", "CPU", "MEMORY", "STORAGE", "TIMESTAMP")

	for _, m := range metrics {
		nodeMetrics := GetResourceMetrics(&m.Usage)
		cpu := nodeMetrics.metrics[v1.ResourceCPU]
		mem := nodeMetrics.metrics[v1.ResourceMemory]
		storage := nodeMetrics.metrics[v1.ResourceStorage]
		fmt.Fprintf(w, "%s\t%vm\t%v Mi\t%v Mi\t%s\n", m.Name, cpu.MilliValue(),
			mem.Value() / (1024*1024), storage.Value() / (1024*1024), m.Timestamp)
	}
	return nil
}

func GetPodMetricsUrl(namespace string, name string) string {
	return fmt.Sprintf("%s/namespaces/%s/pods/%s", MetricsRoot, namespace, name)
}

func GetResourceMetrics(usage *v1.ResourceList) *ResourceMetrics {
	resMetrics := &ResourceMetrics{metrics: make(map[v1.ResourceName]resource.Quantity)}
	for resource := range MeasuredResources {
		resQuantity, found := usage[resource]
		if !found {
			fmt.Errorf("no %v metrics available", strings.ToLower(MeasuredResources[resource]))
		}
		resMetrics.metrics[resource] = resQuantity
	}
	return resMetrics
}

func PrintPodMetrics(out io.Writer, cli *HeapsterMetricsClient, params map[string]string, namespace string, podName string) error {
	resultRaw, err := GetMetrics(cli, GetPodMetricsUrl(namespace, podName), params)
	if err != nil {
		return err
	}

	metrics := make([]metrics_api.PodMetrics, 0)
	err = json.Unmarshal(resultRaw, &metrics)
	if err != nil {
		fmt.Errorf("failed to unmarshall heapster response: %v", err)
		return err
	}

	w := kubectl.GetNewTabWriter(out)
	defer w.Flush()
	fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n", "NAMESPACE", "NAME", "CPU", "MEMORY", "STORAGE", "TIMESTAMP")

	// TODO:
	printContainers := true
	//podMetrics := &ResourceMetrics{metrics: make(map[v1.ResourceName]resource.Quantity)}

	for _, m := range metrics {
		sumCpu, _ := resource.ParseQuantity("0")
		sumMem, _ := resource.ParseQuantity("0")
		sumStorage, _ := resource.ParseQuantity("0")
		for _, c := range m.Containers {
			containerMetrics := GetResourceMetrics(c)
			cpu := containerMetrics.metrics[v1.ResourceCPU]
			mem := containerMetrics.metrics[v1.ResourceMemory]
			storage := containerMetrics.metrics[v1.ResourceStorage]
			sumCpu.Add(cpu)
			sumMem.Add(mem)
			sumStorage.Add(storage)
			if printContainers {
				fmt.Fprint(w, "%s\t%s\t%sm\t%s Mi\t%s Mi\t%s\n", m.Namespace, c.Name,
					cpu.MilliValue(), mem.Value() / (1024*1024), storage.Value() / (1024*1024))
			}
		}
		fmt.Fprintf(w, "%s\t%s\t%sm\t%s Mi\t%s Mi\t%s\n", m.Namespace, m.Name,
			sumCpu.MilliValue(), sumMem.Value() / (1024*1024), sumStorage.Value() / (1024*1024), m.Timestamp)
	}
	return nil
}

func GetMetrics(cli *HeapsterMetricsClient, path string, params map[string]string) ([]byte, error) {
	return cli.client.Services(cli.heapsterNamespace).
		ProxyGet(cli.heapsterScheme, cli.heapsterService, cli.heapsterPort, path, params).
		DoRaw()
}

const (
	MetricsRoot = "/apis/metrics/v1alpha1/"

	DefaultHeapsterNamespace = "kube-system"
	DefaultHeapsterScheme = "http"
	DefaultHeapsterService = "heapster"
	DefaultHeapsterPort = "" // use the first exposed port on the service
)

type HeapsterMetricsClient struct {
	client		  *client.Client
	heapsterNamespace string
	heapsterScheme 	  string
	heapsterService   string
	heapsterPort      string
}

// NewHeapsterMetricsClient returns a new instance of Heapster-based implementation of MetricsClient interface.
func NewHeapsterMetricsClient(client *client.Client, namespace, scheme, service, port string) *HeapsterMetricsClient {
	return &HeapsterMetricsClient{
		client:            client,
		heapsterNamespace: namespace,
		heapsterScheme:    scheme,
		heapsterService:   service,
		heapsterPort:      port,
	}
}

func DefaultHeapsterMetricsClient(client *client.Client) *HeapsterMetricsClient {
	return NewHeapsterMetricsClient(client, DefaultHeapsterNamespace, DefaultHeapsterScheme, DefaultHeapsterService, DefaultHeapsterPort)
}