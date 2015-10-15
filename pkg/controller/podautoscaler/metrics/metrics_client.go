/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package metrics

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"

	heapster "k8s.io/heapster/api/v1/types"
)

const (
	heapsterNamespace = "kube-system"
	heapsterService   = "heapster"
)

var heapsterQueryStart = -5 * time.Minute

// An interface for getting metrics for pods.
type MetricsClient interface {
	GetCPUUtilization(namespace string, selector map[string]string) (*extensions.Utilization, error)
	// Gets average resource consumption for pods under the given selector.
	GetResourceConsumption(
		resourceName api.ResourceName,
		namespace string,
		selector map[string]string) (*extensions.ResourceConsumption, error)
}

// Aggregates results into ResourceConsumption. Also returns number of
// pods included in the aggregation.
type metricAggregator func(heapster.MetricResultList) (extensions.ResourceConsumption, int)

type metricDefinition struct {
	name       string
	aggregator metricAggregator
}

// Heapster-based implementation of MetricsClient
type HeapsterMetricsClient struct {
	client              client.Interface
	resourceDefinitions map[api.ResourceName]metricDefinition
}

var heapsterMetricDefinitions = map[api.ResourceName]metricDefinition{
	api.ResourceCPU: {"cpu-usage",
		func(metrics heapster.MetricResultList) (extensions.ResourceConsumption, int) {
			sum, count := calculateSumFromLatestSample(metrics)
			value := "0"
			if count > 0 {
				// assumes that cpu usage is in millis
				value = fmt.Sprintf("%dm", sum/uint64(count))
			}
			return extensions.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse(value)}, count
		}},
	api.ResourceMemory: {"memory-usage",
		func(metrics heapster.MetricResultList) (extensions.ResourceConsumption, int) {
			sum, count := calculateSumFromLatestSample(metrics)
			value := int64(0)
			if count > 0 {
				value = int64(sum) / int64(count)
			}
			return extensions.ResourceConsumption{Resource: api.ResourceMemory, Quantity: *resource.NewQuantity(value, resource.DecimalSI)}, count
		}},
}

func NewHeapsterMetricsClient(client client.Interface) *HeapsterMetricsClient {
	return &HeapsterMetricsClient{
		client:              client,
		resourceDefinitions: heapsterMetricDefinitions,
	}
}

func (h *HeapsterMetricsClient) GetCPUUtilization(namespace string, selector map[string]string) (*extensions.Utilization, error) {
	usage, err := h.GetResourceConsumption(api.ResourceCPU, namespace, selector)
	if err != nil {
		return nil, fmt.Errorf("failed to get CPU usage: %v", err)
	}

	// TODO: We are reading pod list twice - getting usage and here.
	podList, err := h.client.Pods(namespace).
		List(labels.SelectorFromSet(labels.Set(selector)), fields.Everything())
	if err != nil {
		return nil, fmt.Errorf("failed to get pod list: %v", err)
	}
	request := resource.MustParse("0")
	for _, pod := range podList.Items {
		for _, container := range pod.Spec.Containers {
			containerRequest := container.Resources.Requests[api.ResourceCPU]
			if containerRequest.Amount != nil {
				request.Add(containerRequest)
			}
		}
	}
	if request.Cmp(resource.MustParse("0")) == 0 {
		return nil, fmt.Errorf("Pods do not have requested CPU")
	}

	utilization := new(extensions.Utilization)
	*utilization = extensions.Utilization(float64(usage.Quantity.MilliValue()) / float64(request.MilliValue()))
	return utilization, nil
}

func (h *HeapsterMetricsClient) GetResourceConsumption(resourceName api.ResourceName, namespace string, selector map[string]string) (*extensions.ResourceConsumption, error) {
	podList, err := h.client.Pods(namespace).
		List(labels.SelectorFromSet(labels.Set(selector)), fields.Everything())

	if err != nil {
		return nil, fmt.Errorf("failed to get pod list: %v", err)
	}
	podNames := []string{}
	for _, pod := range podList.Items {
		podNames = append(podNames, pod.Name)
	}
	return h.getForPods(resourceName, namespace, podNames)
}

func (h *HeapsterMetricsClient) getForPods(resourceName api.ResourceName, namespace string, podNames []string) (*extensions.ResourceConsumption, error) {
	metricSpec, metricDefined := h.resourceDefinitions[resourceName]
	if !metricDefined {
		return nil, fmt.Errorf("heapster metric not defined for %v", resourceName)
	}
	now := time.Now()

	startTime := now.Add(heapsterQueryStart)
	metricPath := fmt.Sprintf("/api/v1/model/namespaces/%s/pod-list/%s/metrics/%s",
		namespace,
		strings.Join(podNames, ","),
		metricSpec.name)

	resultRaw, err := h.client.Services(heapsterNamespace).
		ProxyGet(heapsterService, metricPath, map[string]string{"start": startTime.Format(time.RFC3339)}).
		DoRaw()

	if err != nil {
		return nil, fmt.Errorf("failed to get pods metrics: %v", err)
	}

	var metrics heapster.MetricResultList
	err = json.Unmarshal(resultRaw, &metrics)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshall heapster response: %v", err)
	}

	glog.Infof("Metrics available: %s", string(resultRaw))

	currentConsumption, count := metricSpec.aggregator(metrics)
	if count != len(podNames) {
		return nil, fmt.Errorf("metrics obtained for %d/%d of pods", count, len(podNames))
	}

	return &currentConsumption, nil
}

func calculateSumFromLatestSample(metrics heapster.MetricResultList) (uint64, int) {
	sum := uint64(0)
	count := 0
	for _, metrics := range metrics.Items {
		var newest *heapster.MetricPoint
		newest = nil
		for i, metricPoint := range metrics.Metrics {
			if newest == nil || newest.Timestamp.Before(metricPoint.Timestamp) {
				newest = &metrics.Metrics[i]
			}
		}
		if newest != nil {
			sum += newest.Value
			count++
		}
	}
	return sum, count
}
