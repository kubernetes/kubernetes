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
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
	metricsv1alpha1api "k8s.io/metrics/pkg/apis/metrics/v1alpha1"
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
	SVCClient         corev1client.ServicesGetter
	HeapsterNamespace string
	HeapsterScheme    string
	HeapsterService   string
	HeapsterPort      string
}

func NewHeapsterMetricsClient(svcClient corev1client.ServicesGetter, namespace, scheme, service, port string) *HeapsterMetricsClient {
	return &HeapsterMetricsClient{
		SVCClient:         svcClient,
		HeapsterNamespace: namespace,
		HeapsterScheme:    scheme,
		HeapsterService:   service,
		HeapsterPort:      port,
	}
}

func podMetricsURL(namespace string, name string) (string, error) {
	if namespace == metav1.NamespaceAll {
		return fmt.Sprintf("%s/pods", metricsRoot), nil
	}
	errs := validation.ValidateNamespaceName(namespace, false)
	if len(errs) > 0 {
		message := fmt.Sprintf("invalid namespace: %s - %v", namespace, errs)
		return "", errors.New(message)
	}
	if len(name) > 0 {
		errs = validation.NameIsDNSSubdomain(name, false)
		if len(errs) > 0 {
			message := fmt.Sprintf("invalid pod name: %s - %v", name, errs)
			return "", errors.New(message)
		}
	}
	return fmt.Sprintf("%s/namespaces/%s/pods/%s", metricsRoot, namespace, name), nil
}

func nodeMetricsURL(name string) (string, error) {
	if len(name) > 0 {
		errs := validation.NameIsDNSSubdomain(name, false)
		if len(errs) > 0 {
			message := fmt.Sprintf("invalid node name: %s - %v", name, errs)
			return "", errors.New(message)
		}
	}
	return fmt.Sprintf("%s/nodes/%s", metricsRoot, name), nil
}

func (cli *HeapsterMetricsClient) GetNodeMetrics(nodeName string, selector string) (*metricsapi.NodeMetricsList, error) {
	params := map[string]string{"labelSelector": selector}
	path, err := nodeMetricsURL(nodeName)
	if err != nil {
		return nil, err
	}
	resultRaw, err := GetHeapsterMetrics(cli, path, params)
	if err != nil {
		return nil, err
	}
	versionedMetrics := metricsv1alpha1api.NodeMetricsList{}
	if len(nodeName) == 0 {
		err = json.Unmarshal(resultRaw, &versionedMetrics)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshall heapster response: %v", err)
		}
	} else {
		var singleMetric metricsv1alpha1api.NodeMetrics
		err = json.Unmarshal(resultRaw, &singleMetric)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshall heapster response: %v", err)
		}
		versionedMetrics.Items = []metricsv1alpha1api.NodeMetrics{singleMetric}
	}
	metrics := &metricsapi.NodeMetricsList{}
	err = metricsv1alpha1api.Convert_v1alpha1_NodeMetricsList_To_metrics_NodeMetricsList(&versionedMetrics, metrics, nil)
	if err != nil {
		return nil, err
	}
	return metrics, nil
}

func (cli *HeapsterMetricsClient) GetPodMetrics(namespace string, podName string, allNamespaces bool, selector labels.Selector) (*metricsapi.PodMetricsList, error) {
	if allNamespaces {
		namespace = metav1.NamespaceAll
	}
	path, err := podMetricsURL(namespace, podName)
	if err != nil {
		return nil, err
	}

	params := map[string]string{"labelSelector": selector.String()}
	versionedMetrics := metricsv1alpha1api.PodMetricsList{}

	resultRaw, err := GetHeapsterMetrics(cli, path, params)
	if err != nil {
		return nil, err
	}
	if len(podName) == 0 {
		err = json.Unmarshal(resultRaw, &versionedMetrics)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshall heapster response: %v", err)
		}
	} else {
		var singleMetric metricsv1alpha1api.PodMetrics
		err = json.Unmarshal(resultRaw, &singleMetric)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshall heapster response: %v", err)
		}
		versionedMetrics.Items = []metricsv1alpha1api.PodMetrics{singleMetric}
	}
	metrics := &metricsapi.PodMetricsList{}
	err = metricsv1alpha1api.Convert_v1alpha1_PodMetricsList_To_metrics_PodMetricsList(&versionedMetrics, metrics, nil)
	if err != nil {
		return nil, err
	}
	return metrics, nil
}

func GetHeapsterMetrics(cli *HeapsterMetricsClient, path string, params map[string]string) ([]byte, error) {
	return cli.SVCClient.Services(cli.HeapsterNamespace).
		ProxyGet(cli.HeapsterScheme, cli.HeapsterService, cli.HeapsterPort, path, params).
		DoRaw(context.TODO())
}
