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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/validation"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1alpha1"
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
	metricsRoot  = fmt.Sprintf("%s/%s", prefix, groupVersion)

	// TODO: get this from metrics api once it's finished
	metricsGv = schema.GroupVersion{Group: "metrics", Version: "v1alpha1"}
)

type HeapsterMetricsClient struct {
	SVCClient         coreclient.ServicesGetter
	HeapsterNamespace string
	HeapsterScheme    string
	HeapsterService   string
	HeapsterPort      string
}

func NewHeapsterMetricsClient(svcClient coreclient.ServicesGetter, namespace, scheme, service, port string) *HeapsterMetricsClient {
	return &HeapsterMetricsClient{
		SVCClient:         svcClient,
		HeapsterNamespace: namespace,
		HeapsterScheme:    scheme,
		HeapsterService:   service,
		HeapsterPort:      port,
	}
}

func DefaultHeapsterMetricsClient(svcClient coreclient.ServicesGetter) *HeapsterMetricsClient {
	return NewHeapsterMetricsClient(svcClient, DefaultHeapsterNamespace, DefaultHeapsterScheme, DefaultHeapsterService, DefaultHeapsterPort)
}

func podMetricsUrl(namespace string, name string) (string, error) {
	if namespace == metav1.NamespaceAll {
		return fmt.Sprintf("%s/pods", metricsRoot), nil
	}
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
	return fmt.Sprintf("%s/namespaces/%s/pods/%s", metricsRoot, namespace, name), nil
}

func nodeMetricsUrl(name string) (string, error) {
	if len(name) > 0 {
		errs := validation.ValidateNodeName(name, false)
		if len(errs) > 0 {
			message := fmt.Sprintf("invalid node name: %s - %v", name, errs)
			return "", errors.New(message)
		}
	}
	return fmt.Sprintf("%s/nodes/%s", metricsRoot, name), nil
}

func (cli *HeapsterMetricsClient) GetNodeMetrics(nodeName string, selector labels.Selector) ([]metricsapi.NodeMetrics, error) {
	params := map[string]string{"labelSelector": selector.String()}
	path, err := nodeMetricsUrl(nodeName)
	if err != nil {
		return []metricsapi.NodeMetrics{}, err
	}
	resultRaw, err := GetHeapsterMetrics(cli, path, params)
	if err != nil {
		return []metricsapi.NodeMetrics{}, err
	}
	metrics := make([]metricsapi.NodeMetrics, 0)
	if len(nodeName) == 0 {
		metricsList := metricsapi.NodeMetricsList{}
		err = json.Unmarshal(resultRaw, &metricsList)
		if err != nil {
			return []metricsapi.NodeMetrics{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
		}
		metrics = append(metrics, metricsList.Items...)
	} else {
		var singleMetric metricsapi.NodeMetrics
		err = json.Unmarshal(resultRaw, &singleMetric)
		if err != nil {
			return []metricsapi.NodeMetrics{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
		}
		metrics = append(metrics, singleMetric)
	}
	return metrics, nil
}

func (cli *HeapsterMetricsClient) GetPodMetrics(namespace string, podName string, allNamespaces bool, selector labels.Selector) ([]metricsapi.PodMetrics, error) {
	if allNamespaces {
		namespace = metav1.NamespaceAll
	}
	path, err := podMetricsUrl(namespace, podName)
	if err != nil {
		return []metricsapi.PodMetrics{}, err
	}

	params := map[string]string{"labelSelector": selector.String()}
	allMetrics := make([]metricsapi.PodMetrics, 0)

	resultRaw, err := GetHeapsterMetrics(cli, path, params)
	if err != nil {
		return []metricsapi.PodMetrics{}, err
	}
	if len(podName) == 0 {
		metrics := metricsapi.PodMetricsList{}
		err = json.Unmarshal(resultRaw, &metrics)
		if err != nil {
			return []metricsapi.PodMetrics{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
		}
		allMetrics = append(allMetrics, metrics.Items...)
	} else {
		var singleMetric metricsapi.PodMetrics
		err = json.Unmarshal(resultRaw, &singleMetric)
		if err != nil {
			return []metricsapi.PodMetrics{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
		}
		allMetrics = append(allMetrics, singleMetric)
	}
	return allMetrics, nil
}

func GetHeapsterMetrics(cli *HeapsterMetricsClient, path string, params map[string]string) ([]byte, error) {
	return cli.SVCClient.Services(cli.HeapsterNamespace).
		ProxyGet(cli.HeapsterScheme, cli.HeapsterService, cli.HeapsterPort, path, params).
		DoRaw()
}
