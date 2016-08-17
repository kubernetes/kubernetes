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
	"errors"
	"fmt"

	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

const (
	DefaultHeapsterNamespace = "kube-system"
	DefaultHeapsterScheme    = "http"
	DefaultHeapsterService   = "heapster"
	DefaultHeapsterPort      = "" // use the first exposed port on the service
)

var (
	prefix       = "/apis"
	groupVersion = fmt.Sprintf("%s/%s", metricsGv.Group, metricsGv.Version)
	MetricsRoot  = fmt.Sprintf("%s/%s", prefix, groupVersion)

	// TODO: get this from metrics api once it's finished
	metricsGv = unversioned.GroupVersion{Group: "metrics", Version: "v1alpha1"}
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

func PodMetricsUrl(namespace string, name string) (string, error) {
	errs := validation.ValidateNamespaceName(namespace, false)
	if len(errs) > 0 {
		message := fmt.Sprintf("invalid namespace: %s - %v", namespace, errs)
		return "", errors.New(message)
	}
	if len(name) > 0 {
		errs = validation.ValidatePodName(name, false)
		if len(errs) > 0 {
			message := fmt.Sprintf("invalid pod name: %s - %v", name, errs)
			return "", errors.New(message)
		}
	}
	return fmt.Sprintf("%s/namespaces/%s/pods/%s", MetricsRoot, namespace, name), nil
}

func NodeMetricsUrl(name string) (string, error) {
	if len(name) > 0 {
		errs := validation.ValidateNodeName(name, false)
		if len(errs) > 0 {
			message := fmt.Sprintf("invalid node name: %s - %v", name, errs)
			return "", errors.New(message)
		}
	}
	return fmt.Sprintf("%s/nodes/%s", MetricsRoot, name), nil
}

func (cli *HeapsterMetricsClient) GetNodeMetrics(nodeName string, selector string) ([]metrics_api.NodeMetrics, error) {
	params := map[string]string{"labelSelector": selector}
	path, err := NodeMetricsUrl(nodeName)
	if err != nil {
		return []metrics_api.NodeMetrics{}, err
	}
	resultRaw, err := GetHeapsterMetrics(cli, path, params)
	if err != nil {
		return []metrics_api.NodeMetrics{}, err
	}
	metrics := make([]metrics_api.NodeMetrics, 0)
	if len(nodeName) == 0 {
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
		path, err := PodMetricsUrl(ns, podName)
		if err != nil {
			return []metrics_api.PodMetrics{}, err
		}
		resultRaw, err := GetHeapsterMetrics(cli, path, params)
		if err != nil {
			return []metrics_api.PodMetrics{}, err
		}
		if len(podName) == 0 {
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
