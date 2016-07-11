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

package kubectl

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
)

func PrintMetrics(out io.Writer, cli *client.Client, cmdNamespace string, allNamespaces bool,
		  printContainers bool, params map[string]string, args []string) error {

	resType, err := GetResourceKind(args[0])
	if err != nil {
		return err
	}

	name := ""
	if len(args) > 1 {
		name = args[1]
	}

	switch resType {
	case api.Kind("Node"):
		return PrintNodeMetrics(out, DefaultHeapsterMetricsClient(cli), allNamespaces, params, name)
	case api.Kind("Pod"):
		return PrintPodMetrics(out, DefaultHeapsterMetricsClient(cli), allNamespaces, printContainers, params, cmdNamespace, name)
	}
	return errors.New("Resource not supported.")
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

func PrintNodeMetrics(out io.Writer, cli *HeapsterMetricsClient, allNamespaces bool, params map[string]string, nodeName string) error {
	resultRaw, err := GetHeapsterMetrics(cli, NodeMetricsUrl(nodeName), params)
	if err != nil {
		return err
	}
	w := GetNewTabWriter(out)
	defer w.Flush()

	metrics := make([]metrics_api.NodeMetrics, 1)
	if nodeName == "" {
		err = json.Unmarshal(resultRaw, &metrics)
	} else {
		err = json.Unmarshal(resultRaw, &metrics[0])
	}
	if err != nil {
		fmt.Errorf("failed to unmarshall heapster response: %v", err)
		return err
	}

	fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", "NAME", "CPU", "MEMORY", "STORAGE", "TIMESTAMP")
	for _, m := range metrics {
		fmt.Fprintf(w, "%s", m.Name)
		PrintAllResourceUsages(w, CreateResourceMetrics(m.Usage))
		fmt.Fprintf(w, "\t%v\n", m.Timestamp)
	}
	return nil
}

func PrintPodMetrics(out io.Writer, cli *HeapsterMetricsClient, allNamespaces bool, printContainers bool,
		     params map[string]string, namespace string, podName string) error {
	resultRaw, err := GetHeapsterMetrics(cli, PodMetricsUrl(namespace, podName), params)
	if err != nil {
		return err
	}
	w := GetNewTabWriter(out)
	defer w.Flush()

	metrics := make([]metrics_api.PodMetrics, 1)
	if podName == "" {
		err = json.Unmarshal(resultRaw, &metrics)
	} else {
		err = json.Unmarshal(resultRaw, &metrics[0])
	}
	if err != nil {
		fmt.Errorf("Failed to unmarshall heapster response: %v", err)
		return err
	}

	fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n", "NAMESPACE", "NAME", "CPU", "MEMORY", "STORAGE", "TIMESTAMP")
	for _, m := range metrics {
		PrintSinglePodMetrics(w, &m, printContainers)
	}
	return nil
}

func PrintSinglePodMetrics(out io.Writer, m *metrics_api.PodMetrics, printContainers bool) {
	podMetrics := &ResourceMetrics{metrics: make(map[v1.ResourceName]resource.Quantity)}

	containers := make(map[string]*ResourceMetrics)
	for _, res := range MeasuredResources {
		podMetrics.metrics[res], _ = resource.ParseQuantity("0")
	}
	for _, c := range m.Containers {
		containers[c.Name] = CreateResourceMetrics(c.Usage)
		for _, res := range MeasuredResources {
			quantity := podMetrics.metrics[res]
			quantity.Add(c.Usage[res])
			podMetrics.metrics[res] = quantity
		}
	}

	fmt.Fprintf(out, "%s\t%s", m.Namespace, m.Name)
	PrintAllResourceUsages(out, podMetrics)
	fmt.Fprintf(out, "\t%v\n", m.Timestamp)

	if printContainers {
		//fmt.Fprintf(out, "Containers:")
		for contName := range containers {
			fmt.Fprintf(out, "%s\t%s", "", contName)
			PrintAllResourceUsages(out, containers[contName])
			fmt.Fprintf(out, "\n")
		}
		fmt.Fprintf(out, "\t\t\t\t\t\n")
	}
}

func PrintSingleResourceUsage(out io.Writer, resourceType v1.ResourceName, quantity resource.Quantity) {
	switch resourceType {
	case v1.ResourceCPU:
		fmt.Fprintf(out, "%vm", quantity.MilliValue())
	case v1.ResourceMemory, v1.ResourceStorage:
		fmt.Fprintf(out, "%v Mi", quantity.Value() / (1024 * 1024))
	default:
		fmt.Fprintf(out, "%v", quantity.Value())
	}
}

func PrintAllResourceUsages(out io.Writer, metrics *ResourceMetrics) {
	for _, res := range MeasuredResources {
		quantity := metrics.metrics[res]
		fmt.Fprintf(out, "\t")
		PrintSingleResourceUsage(out, res, quantity)
	}
}

func PodMetricsUrl(namespace string, name string) string {
	return fmt.Sprintf("%s/namespaces/%s/pods/%s", MetricsRoot, namespace, name)
}

func NodeMetricsUrl(name string) string {
	return fmt.Sprintf("%s/nodes/%s", MetricsRoot, name)
}

type ResourceMetrics struct {
	metrics map[v1.ResourceName]resource.Quantity
}

func CreateResourceMetrics(usage v1.ResourceList) *ResourceMetrics {
	resMetrics := &ResourceMetrics{metrics: make(map[v1.ResourceName]resource.Quantity)}
	for _, resource := range MeasuredResources {
		resQuantity, found := usage[resource]
		if !found {
			fmt.Errorf("no %v metrics available", strings.ToLower(string(resource)))
		}
		resMetrics.metrics[resource] = resQuantity
	}
	return resMetrics
}

var MeasuredResources = []v1.ResourceName {
	v1.ResourceCPU,
	v1.ResourceMemory,
	v1.ResourceStorage,
}

func GetHeapsterMetrics(cli *HeapsterMetricsClient, path string, params map[string]string) ([]byte, error) {
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
	client            *client.Client
	heapsterNamespace string
	heapsterScheme    string
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
