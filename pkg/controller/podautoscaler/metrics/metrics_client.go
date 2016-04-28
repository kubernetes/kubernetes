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
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/labels"

	heapster "k8s.io/heapster/metrics/api/v1/types"
)

const (
	DefaultHeapsterNamespace = "kube-system"
	DefaultHeapsterScheme    = "http"
	DefaultHeapsterService   = "heapster"
	DefaultHeapsterPort      = "" // use the first exposed port on the service
)

var heapsterQueryStart = -5 * time.Minute

// MetricsClient is an interface for getting metrics for pods.
type MetricsClient interface {
	// GetCPUUtilization returns the average utilization over all pods represented as a percent of requested CPU
	// (e.g. 70 means that an average pod uses 70% of the requested CPU)
	// and the time of generation of the oldest of utilization reports for pods.
	GetCPUUtilization(namespace string, selector labels.Selector) (*int, time.Time, error)

	// GetCustomMetric returns the average value of the given custom metrics from the
	// pods picked using the namespace and selector passed as arguments.
	GetCustomMetric(customMetricName string, namespace string, selector labels.Selector) (*float64, time.Time, error)
}

type intAndFloat struct {
	intValue   int64
	floatValue float64
}

// Aggregates results into ResourceConsumption. Also returns number of pods included in the aggregation.
type metricAggregator func(heapster.MetricResultList) (intAndFloat, int, time.Time)

type metricDefinition struct {
	name       string
	aggregator metricAggregator
}

// HeapsterMetricsClient is Heapster-based implementation of MetricsClient
type HeapsterMetricsClient struct {
	client            clientset.Interface
	heapsterNamespace string
	heapsterScheme    string
	heapsterService   string
	heapsterPort      string
}

var averageFunction = func(metrics heapster.MetricResultList) (intAndFloat, int, time.Time) {
	sum, count, timestamp := calculateSumFromTimeSample(metrics, time.Minute)
	result := intAndFloat{0, 0}
	if count > 0 {
		result.intValue = sum.intValue / int64(count)
		result.floatValue = sum.floatValue / float64(count)
	}
	return result, count, timestamp
}

var heapsterCpuUsageMetricDefinition = metricDefinition{"cpu-usage", averageFunction}

func getHeapsterCustomMetricDefinition(metricName string) metricDefinition {
	return metricDefinition{"custom/" + metricName, averageFunction}
}

// NewHeapsterMetricsClient returns a new instance of Heapster-based implementation of MetricsClient interface.
func NewHeapsterMetricsClient(client clientset.Interface, namespace, scheme, service, port string) *HeapsterMetricsClient {
	return &HeapsterMetricsClient{
		client:            client,
		heapsterNamespace: namespace,
		heapsterScheme:    scheme,
		heapsterService:   service,
		heapsterPort:      port,
	}
}

func (h *HeapsterMetricsClient) GetCPUUtilization(namespace string, selector labels.Selector) (*int, time.Time, error) {
	avgConsumption, avgRequest, timestamp, err := h.GetCpuConsumptionAndRequestInMillis(namespace, selector)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to get CPU consumption and request: %v", err)
	}
	utilization := int((avgConsumption * 100) / avgRequest)
	return &utilization, timestamp, nil
}

func (h *HeapsterMetricsClient) GetCpuConsumptionAndRequestInMillis(namespace string, selector labels.Selector) (avgConsumption int64,
	avgRequest int64, timestamp time.Time, err error) {

	podList, err := h.client.Core().Pods(namespace).
		List(api.ListOptions{LabelSelector: selector})

	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("failed to get pod list: %v", err)
	}
	podNames := []string{}
	requestSum := int64(0)
	missing := false
	for _, pod := range podList.Items {
		if pod.Status.Phase == api.PodPending {
			// Skip pending pods.
			continue
		}

		podNames = append(podNames, pod.Name)
		for _, container := range pod.Spec.Containers {
			containerRequest := container.Resources.Requests[api.ResourceCPU]
			if containerRequest.Amount != nil {
				requestSum += containerRequest.MilliValue()
			} else {
				missing = true
			}
		}
	}
	if len(podNames) == 0 && len(podList.Items) > 0 {
		return 0, 0, time.Time{}, fmt.Errorf("no running pods")
	}
	if missing || requestSum == 0 {
		return 0, 0, time.Time{}, fmt.Errorf("some pods do not have request for cpu")
	}
	glog.V(4).Infof("%s %s - sum of CPU requested: %d", namespace, selector, requestSum)
	requestAvg := requestSum / int64(len(podList.Items))
	// Consumption is already averaged and in millis.
	consumption, timestamp, err := h.getForPods(heapsterCpuUsageMetricDefinition, namespace, podNames)
	if err != nil {
		return 0, 0, time.Time{}, err
	}
	return consumption.intValue, requestAvg, timestamp, nil
}

// GetCustomMetric returns the average value of the given custom metric from the
// pods picked using the namespace and selector passed as arguments.
func (h *HeapsterMetricsClient) GetCustomMetric(customMetricName string, namespace string, selector labels.Selector) (*float64, time.Time, error) {
	metricSpec := getHeapsterCustomMetricDefinition(customMetricName)

	podList, err := h.client.Core().Pods(namespace).List(api.ListOptions{LabelSelector: selector})

	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to get pod list: %v", err)
	}
	podNames := []string{}
	for _, pod := range podList.Items {
		if pod.Status.Phase == api.PodPending {
			// Skip pending pods.
			continue
		}
		podNames = append(podNames, pod.Name)
	}
	if len(podNames) == 0 && len(podList.Items) > 0 {
		return nil, time.Time{}, fmt.Errorf("no running pods")
	}

	value, timestamp, err := h.getForPods(metricSpec, namespace, podNames)
	if err != nil {
		return nil, time.Time{}, err
	}
	return &value.floatValue, timestamp, nil
}

func (h *HeapsterMetricsClient) getForPods(metricSpec metricDefinition, namespace string, podNames []string) (*intAndFloat, time.Time, error) {

	now := time.Now()

	startTime := now.Add(heapsterQueryStart)
	metricPath := fmt.Sprintf("/api/v1/model/namespaces/%s/pod-list/%s/metrics/%s",
		namespace,
		strings.Join(podNames, ","),
		metricSpec.name)

	resultRaw, err := h.client.Core().Services(h.heapsterNamespace).
		ProxyGet(h.heapsterScheme, h.heapsterService, h.heapsterPort, metricPath, map[string]string{"start": startTime.Format(time.RFC3339)}).
		DoRaw()

	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to get pods metrics: %v", err)
	}

	var metrics heapster.MetricResultList
	err = json.Unmarshal(resultRaw, &metrics)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
	}

	glog.V(4).Infof("Heapster metrics result: %s", string(resultRaw))

	sum, count, timestamp := metricSpec.aggregator(metrics)
	if count != len(podNames) {
		return nil, time.Time{}, fmt.Errorf("metrics obtained for %d/%d of pods", count, len(podNames))
	}

	return &sum, timestamp, nil
}

func calculateSumFromTimeSample(metrics heapster.MetricResultList, duration time.Duration) (sum intAndFloat, count int, timestamp time.Time) {
	sum = intAndFloat{0, 0}
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
			intervalSum := intAndFloat{0, 0}
			intSumCount := 0
			floatSumCount := 0
			for _, metricPoint := range metrics.Metrics {
				if metricPoint.Timestamp.Add(duration).After(newest.Timestamp) {
					intervalSum.intValue += int64(metricPoint.Value)
					intSumCount++
					if metricPoint.FloatValue != nil {
						intervalSum.floatValue += *metricPoint.FloatValue
						floatSumCount++
					}
				}
			}
			if newest.FloatValue == nil {
				if intSumCount > 0 {
					sum.intValue += int64(intervalSum.intValue / int64(intSumCount))
					sum.floatValue += float64(intervalSum.intValue / int64(intSumCount))
				}
			} else {
				if floatSumCount > 0 {
					sum.intValue += int64(intervalSum.floatValue / float64(floatSumCount))
					sum.floatValue += intervalSum.floatValue / float64(floatSumCount)
				}
			}
			count++
		}
	}
	if oldest != nil {
		timestamp = *oldest
	}
	return sum, count, timestamp
}
