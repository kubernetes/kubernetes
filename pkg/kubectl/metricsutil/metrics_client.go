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

package metricsutil

import (
	"encoding/json"
	"fmt"

	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

const (
	MetricsRoot              = "/apis/metrics/v1alpha1/"
	DefaultHeapsterNamespace = "kube-system"
	DefaultHeapsterScheme    = "http"
	DefaultHeapsterService   = "heapster"
	DefaultHeapsterPort      = "" // use the first exposed port on the service
)

type HeapsterMetricsClient struct {
	Client            *client.Client
	HeapsterNamespace string
	HeapsterScheme    string
	HeapsterService   string
	HeapsterPort      string
}

func NewHeapsterMetricsClient(client *client.Client, namespace, scheme, service, port string) *HeapsterMetricsClient {
	return &HeapsterMetricsClient{
		Client:            client,
		HeapsterNamespace: namespace,
		HeapsterScheme:    scheme,
		HeapsterService:   service,
		HeapsterPort:      port,
	}
}

func DefaultHeapsterMetricsClient(client *client.Client) *HeapsterMetricsClient {
	return NewHeapsterMetricsClient(client, DefaultHeapsterNamespace, DefaultHeapsterScheme, DefaultHeapsterService, DefaultHeapsterPort)
}

func PodMetricsUrl(namespace string, name string) string {
	return fmt.Sprintf("%s/namespaces/%s/pods/%s", MetricsRoot, namespace, name)
}

func NodeMetricsUrl(name string) string {
	return fmt.Sprintf("%s/nodes/%s", MetricsRoot, name)
}

func (cli *HeapsterMetricsClient) GetNodeMetrics(nodeName string, selector string) ([]metrics_api.NodeMetrics, error) {
	params := map[string]string{"labelSelector": selector}
	resultRaw, err := GetHeapsterMetrics(cli, NodeMetricsUrl(nodeName), params)
	if err != nil {
		return []metrics_api.NodeMetrics{}, err
	}
	metrics := make([]metrics_api.NodeMetrics, 0)
	if nodeName == "" {
		err = json.Unmarshal(resultRaw, &metrics)
		if err != nil {
			return []metrics_api.NodeMetrics{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
		}
	} else {
		var singleMetric metrics_api.NodeMetrics
		err = json.Unmarshal(resultRaw, &singleMetric)
		if err != nil {
			return []metrics_api.NodeMetrics{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
		}
		metrics = append(metrics, singleMetric)
	}
	return metrics, nil
}

func (cli *HeapsterMetricsClient) GetPodMetrics(namespace string, podName string, allNamespaces bool, selector string) ([]metrics_api.PodMetrics, error) {
	// TODO: extend Master Metrics API with getting pods from all namespaces
	// instead of aggregating the results here
	namespaces := make([]string, 0)
	if allNamespaces {
		list, err := cli.Client.Namespaces().List(api.ListOptions{})
		if err != nil {
			return []metrics_api.PodMetrics{}, err
		}
		for _, ns := range list.Items {
			namespaces = append(namespaces, ns.Name)
		}
	} else {
		namespaces = append(namespaces, namespace)
	}

	params := map[string]string{"labelSelector": selector}
	allMetrics := make([]metrics_api.PodMetrics, 0)
	for _, ns := range namespaces {
		resultRaw, err := GetHeapsterMetrics(cli, PodMetricsUrl(ns, podName), params)
		if err != nil {
			return []metrics_api.PodMetrics{}, err
		}
		if podName == "" {
			metrics := make([]metrics_api.PodMetrics, 0)
			err = json.Unmarshal(resultRaw, &metrics)
			if err != nil {
				return []metrics_api.PodMetrics{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
			}
			allMetrics = append(allMetrics, metrics...)
		} else {
			var singleMetric metrics_api.PodMetrics
			err = json.Unmarshal(resultRaw, &singleMetric)
			if err != nil {
				return []metrics_api.PodMetrics{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
			}
			allMetrics = append(allMetrics, singleMetric)
		}
	}
	return allMetrics, nil
}

func GetHeapsterMetrics(cli *HeapsterMetricsClient, path string, params map[string]string) ([]byte, error) {
	return cli.Client.Services(cli.HeapsterNamespace).
		ProxyGet(cli.HeapsterScheme, cli.HeapsterService, cli.HeapsterPort, path, params).
		DoRaw()
}
