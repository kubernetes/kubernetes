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
	"fmt"
	"encoding/json"

	client "k8s.io/kubernetes/pkg/client/unversioned"
	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
)

const (
	MetricsRoot = "/apis/metrics/v1alpha1/"
	DefaultHeapsterNamespace = "kube-system"
	DefaultHeapsterScheme = "http"
	DefaultHeapsterService = "heapster"
	DefaultHeapsterPort = "" // use the first exposed port on the service
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

func (cli *HeapsterMetricsClient) GetNodeMetrics(nodeName string, params map[string]string) ([]metrics_api.NodeMetrics, error) {
	resultRaw, err := GetHeapsterMetrics(cli, NodeMetricsUrl(nodeName), params)
	if err != nil {
		return []metrics_api.NodeMetrics{}, err
	}

	metrics := make([]metrics_api.NodeMetrics, 0)
	if nodeName == "" {
		err = json.Unmarshal(resultRaw, &metrics)
		if err != nil {
			fmt.Errorf("failed to unmarshall heapster response: %v", err)
			return []metrics_api.NodeMetrics{}, err
		}
	} else {
		var singleMetric metrics_api.NodeMetrics
		err = json.Unmarshal(resultRaw, &singleMetric)
		if err != nil {
			fmt.Errorf("failed to unmarshall heapster response: %v", err)
			return []metrics_api.NodeMetrics{}, err
		}
		metrics = append(metrics, singleMetric)
	}
	return metrics, nil
}

func (cli *HeapsterMetricsClient) GetPodMetrics(namespace string, podName string, params map[string]string) ([]metrics_api.PodMetrics, error) {
	resultRaw, err := GetHeapsterMetrics(cli, PodMetricsUrl(namespace, podName), params)
	if err != nil {
		return []metrics_api.PodMetrics{}, err
	}

	metrics := make([]metrics_api.PodMetrics, 0)
	if podName == "" {
		err = json.Unmarshal(resultRaw, &metrics)
		if err != nil {
			fmt.Errorf("failed to unmarshall heapster response: %v", err)
			return []metrics_api.PodMetrics{}, err
		}
	} else {
		var singleMetric metrics_api.PodMetrics
		err = json.Unmarshal(resultRaw, &singleMetric)
		if err != nil {
			fmt.Errorf("failed to unmarshall heapster response: %v", err)
			return []metrics_api.PodMetrics{}, err
		}
		metrics = append(metrics, singleMetric)
	}
	return metrics, nil
}

func GetHeapsterMetrics(cli *HeapsterMetricsClient, path string, params map[string]string) ([]byte, error) {
	return cli.Client.Services(cli.HeapsterNamespace).
		ProxyGet(cli.HeapsterScheme, cli.HeapsterService, cli.HeapsterPort, path, params).
		DoRaw()
}
