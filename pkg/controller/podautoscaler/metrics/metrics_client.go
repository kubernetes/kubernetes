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

// MetricsClient is an interface for getting metrics for pods.
type MetricsClient interface {
	// GetCPUUtilization returns the average utilization over all pods represented as a percent of requested CPU
	// (e.g. 70 means that an average pod uses 70% of the requested CPU)
	// and the time of generation of the oldest of utilization reports for pods.
	GetCPUUtilization(namespace string, selector map[string]string) (*int, time.Time, error)
}

// ResourceConsumption specifies consumption of a particular resource.
type ResourceConsumption struct {
	Resource api.ResourceName
	Quantity resource.Quantity
}

// Aggregates results into ResourceConsumption. Also returns number of pods included in the aggregation.
type metricAggregator func(heapster.MetricResultList) (ResourceConsumption, int, time.Time)

type metricDefinition struct {
	name       string
	aggregator metricAggregator
}

// HeapsterMetricsClient is Heapster-based implementation of MetricsClient
type HeapsterMetricsClient struct {
	client              client.Interface
	resourceDefinitions map[api.ResourceName]metricDefinition
}

var heapsterMetricDefinitions = map[api.ResourceName]metricDefinition{
	api.ResourceCPU: {"cpu-usage",
		func(metrics heapster.MetricResultList) (ResourceConsumption, int, time.Time) {
			sum, count, timestamp := calculateSumFromLatestSample(metrics)
			value := "0"
			if count > 0 {
				// assumes that cpu usage is in millis
				value = fmt.Sprintf("%dm", sum/uint64(count))
			}
			return ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse(value)}, count, timestamp
		}},
	api.ResourceMemory: {"memory-usage",
		func(metrics heapster.MetricResultList) (ResourceConsumption, int, time.Time) {
			sum, count, timestamp := calculateSumFromLatestSample(metrics)
			value := int64(0)
			if count > 0 {
				value = int64(sum) / int64(count)
			}
			return ResourceConsumption{Resource: api.ResourceMemory, Quantity: *resource.NewQuantity(value, resource.DecimalSI)}, count, timestamp
		}},
}

// NewHeapsterMetricsClient returns a new instance of Heapster-based implementation of MetricsClient interface.
func NewHeapsterMetricsClient(client client.Interface) *HeapsterMetricsClient {
	return &HeapsterMetricsClient{
		client:              client,
		resourceDefinitions: heapsterMetricDefinitions,
	}
}

func (h *HeapsterMetricsClient) GetCPUUtilization(namespace string, selector map[string]string) (*int, time.Time, error) {
	consumption, request, timestamp, err := h.GetResourceConsumptionAndRequest(api.ResourceCPU, namespace, selector)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to get CPU consumption and request: %v", err)
	}
	utilization := new(int)
	*utilization = int(float64(consumption.Quantity.MilliValue()) / float64(request.MilliValue()) * 100)
	return utilization, timestamp, nil
}

func (h *HeapsterMetricsClient) GetResourceConsumptionAndRequest(resourceName api.ResourceName, namespace string, selector map[string]string) (consumption *ResourceConsumption, request *resource.Quantity, timestamp time.Time, err error) {
	podList, err := h.client.Pods(namespace).
		List(labels.SelectorFromSet(labels.Set(selector)), fields.Everything())

	if err != nil {
		return nil, nil, time.Time{}, fmt.Errorf("failed to get pod list: %v", err)
	}
	podNames := []string{}
	sum := resource.MustParse("0")
	missing := false
	for _, pod := range podList.Items {
		podNames = append(podNames, pod.Name)
		for _, container := range pod.Spec.Containers {
			containerRequest := container.Resources.Requests[resourceName]
			if containerRequest.Amount != nil {
				sum.Add(containerRequest)
			} else {
				missing = true
			}
		}
	}
	if missing || sum.Cmp(resource.MustParse("0")) == 0 {
		return nil, nil, time.Time{}, fmt.Errorf("some pods do not have request for %s", resourceName)
	}
	glog.Infof("Sum of %s requested: %v", resourceName, sum)
	avg := resource.MustParse(fmt.Sprintf("%dm", sum.MilliValue()/int64(len(podList.Items))))
	request = &avg
	consumption, timestamp, err = h.getForPods(resourceName, namespace, podNames)
	if err != nil {
		return nil, nil, time.Time{}, err
	}
	return consumption, request, timestamp, nil
}

func (h *HeapsterMetricsClient) getForPods(resourceName api.ResourceName, namespace string, podNames []string) (*ResourceConsumption, time.Time, error) {
	metricSpec, metricDefined := h.resourceDefinitions[resourceName]
	if !metricDefined {
		return nil, time.Time{}, fmt.Errorf("heapster metric not defined for %v", resourceName)
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
		return nil, time.Time{}, fmt.Errorf("failed to get pods metrics: %v", err)
	}

	var metrics heapster.MetricResultList
	err = json.Unmarshal(resultRaw, &metrics)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
	}

	glog.Infof("Metrics available: %s", string(resultRaw))

	currentConsumption, count, timestamp := metricSpec.aggregator(metrics)
	if count != len(podNames) {
		return nil, time.Time{}, fmt.Errorf("metrics obtained for %d/%d of pods", count, len(podNames))
	}

	return &currentConsumption, timestamp, nil
}

func calculateSumFromLatestSample(metrics heapster.MetricResultList) (sum uint64, count int, timestamp time.Time) {
	sum = uint64(0)
	count = 0
	timestamp = time.Time{}
	var oldest *time.Time // creation time of the oldest of used samples across pods
	oldest = nil
	for _, metrics := range metrics.Items {
		var newest *heapster.MetricPoint // creation time of the newest sample for pod
		newest = nil
		for i, metricPoint := range metrics.Metrics {
			if newest == nil || newest.Timestamp.Before(metricPoint.Timestamp) {
				newest = &metrics.Metrics[i]
			}
		}
		if newest != nil {
			if oldest == nil || newest.Timestamp.Before(*oldest) {
				oldest = &newest.Timestamp
			}
			sum += newest.Value
			count++
		}
	}
	if oldest != nil {
		timestamp = *oldest
	}
	return sum, count, timestamp
}
