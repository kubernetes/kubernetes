/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/sets"

	heapster "k8s.io/heapster/metrics/api/v1/types"
	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
)

const (
	DefaultHeapsterNamespace = "kube-system"
	DefaultHeapsterScheme    = "http"
	DefaultHeapsterService   = "heapster"
	DefaultHeapsterPort      = "" // use the first exposed port on the service
)

var heapsterQueryStart = -5 * time.Minute

// UtilizationInfo contains extra metadata about the returned metric values
type UtilizationInfo struct {
	// ReadyPodsCount is the total number of pods factored into the computation of the returned metric value
	ReadyPodsCount int

	// UnreadyPodsCount is the total number of running pods not factored into the computation of the returned metric value (due to being unready)
	UnreadyPodsCount int

	// OldestTimestamp is the time of generation of the oldest of the utilization reports for the pods
	OldestTimestamp time.Time

	// Utiliziation is the metric utilization for the requested metric (may be CPU as a percentage of request,
	// or a raw custom metric value)
	Utilization float64
}

// MetricsClient is an interface for getting metrics for pods.
type MetricsClient interface {
	// GetCPUUtilization returns the average utilization over all pods represented as a percent of requested CPU
	// (e.g. 70 means that an average pod uses 70% of the requested CPU), as well as associated information about
	// the computation.
	GetCPUUtilization(namespace string, selector labels.Selector) (*UtilizationInfo, error)

	// GetCustomMetric returns the average value of the given custom metrics from the
	// pods picked using the namespace and selector passed as arguments, as well as associated information about
	// the computation.
	GetCustomMetric(customMetricName string, namespace string, selector labels.Selector) (*UtilizationInfo, error)
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

func (h *HeapsterMetricsClient) GetCPUUtilization(namespace string, selector labels.Selector) (*UtilizationInfo, error) {
	avgConsumption, avgRequest, info, err := h.GetCpuConsumptionAndRequestInMillis(namespace, selector)
	if err != nil {
		return nil, fmt.Errorf("failed to get CPU consumption and request: %v", err)
	}
	info.Utilization = float64((avgConsumption * 100) / avgRequest)

	return info, nil
}

func (h *HeapsterMetricsClient) GetCpuConsumptionAndRequestInMillis(namespace string, selector labels.Selector) (avgConsumption int64,
	avgRequest int64, utilizationInfo *UtilizationInfo, err error) {

	podList, err := h.client.Core().Pods(namespace).
		List(api.ListOptions{LabelSelector: selector})

	if err != nil {
		return 0, 0, nil, fmt.Errorf("failed to get pod list: %v", err)
	}
	readyPods := sets.NewString()
	unreadyPods := sets.NewString()
	requestSum := int64(0)
	missing := false
	for _, pod := range podList.Items {
		if pod.Status.Phase != api.PodRunning {
			// Count only running pods.
			continue
		}

		if !api.IsPodReady(&pod) {
			// don't perform the normal logic on unready pods
			unreadyPods.Insert(pod.Name)
			continue
		}

		readyPods.Insert(pod.Name)

		for _, container := range pod.Spec.Containers {
			if containerRequest, ok := container.Resources.Requests[api.ResourceCPU]; ok {
				requestSum += containerRequest.MilliValue()
			} else {
				missing = true
			}
		}
	}
	if readyPods.Len() == 0 {
		return 0, 0, nil, fmt.Errorf("no running and ready pods")
	}
	if missing || requestSum == 0 {
		return 0, 0, nil, fmt.Errorf("some pods do not have request for cpu")
	}
	glog.V(4).Infof("%s %s - sum of CPU requested: %d", namespace, selector, requestSum)
	requestAvg := requestSum / int64(readyPods.Len())
	// Consumption is already averaged and in millis.
	consumption, timestamp, err := h.getCpuUtilizationForPods(namespace, selector, readyPods, unreadyPods)
	if err != nil {
		return 0, 0, nil, err
	}

	info := &UtilizationInfo{
		ReadyPodsCount:   readyPods.Len(),
		UnreadyPodsCount: unreadyPods.Len(),
		OldestTimestamp:  timestamp,
	}

	return consumption, requestAvg, info, nil
}

func (h *HeapsterMetricsClient) getCpuUtilizationForPods(namespace string, selector labels.Selector, readyPods sets.String, unreadyPods sets.String) (int64, time.Time, error) {
	metricPath := fmt.Sprintf("/apis/metrics/v1alpha1/namespaces/%s/pods", namespace)
	params := map[string]string{"labelSelector": selector.String()}

	resultRaw, err := h.client.Core().Services(h.heapsterNamespace).
		ProxyGet(h.heapsterScheme, h.heapsterService, h.heapsterPort, metricPath, params).
		DoRaw()
	if err != nil {
		return 0, time.Time{}, fmt.Errorf("failed to get pods metrics: %v", err)
	}

	glog.V(4).Infof("Heapster metrics result: %s", string(resultRaw))

	metrics := metrics_api.PodMetricsList{}
	err = json.Unmarshal(resultRaw, &metrics)
	if err != nil {
		return 0, time.Time{}, fmt.Errorf("failed to unmarshall heapster response: %v", err)
	}

	sum := int64(0)
	count := int64(0)
	for _, m := range metrics.Items {
		if readyPods.Has(m.Name) {
			for _, c := range m.Containers {
				cpu, found := c.Usage[v1.ResourceCPU]
				if !found {
					return 0, time.Time{}, fmt.Errorf("no cpu for container %v in pod %v/%v", c.Name, namespace, m.Name)
				}
				sum += cpu.MilliValue()
			}

			count++
		}
	}

	// we got the wrong number of metrics (we're probably missing some), so report an error
	if int(count) != readyPods.Len() {
		hint := ""
		if int(count) < readyPods.Len() {
			present := sets.NewString()
			for _, m := range metrics.Items {
				present.Insert(m.Name)
			}
			for expected := range readyPods {
				if !present.Has(expected) {
					hint = fmt.Sprintf(", sample missing pod: %s/%s", namespace, expected)
					break
				}
			}
		}
		return 0, time.Time{}, fmt.Errorf("metrics obtained for %d/%d of ready pods (metrics obtained for %d pods total%s)", count, readyPods.Len(), len(metrics.Items), hint)
	}

	return sum / count, metrics.Items[0].Timestamp.Time, nil
}

// GetCustomMetric returns the average value of the given custom metric from the
// pods picked using the namespace and selector passed as arguments.
func (h *HeapsterMetricsClient) GetCustomMetric(customMetricName string, namespace string, selector labels.Selector) (*UtilizationInfo, error) {
	metricSpec := getHeapsterCustomMetricDefinition(customMetricName)

	podList, err := h.client.Core().Pods(namespace).List(api.ListOptions{LabelSelector: selector})

	if err != nil {
		return nil, fmt.Errorf("failed to get pod list: %v", err)
	}
	readyPods := []string{}
	unreadyPodsCount := 0
	for _, pod := range podList.Items {
		if pod.Status.Phase == api.PodPending {
			// Skip pending pods.
			continue
		}
		if !api.IsPodReady(&pod) {
			// skip unready pods
			unreadyPodsCount++
			continue
		}
		readyPods = append(readyPods, pod.Name)
	}
	if len(readyPods) == 0 && len(podList.Items) > 0 {
		return nil, fmt.Errorf("no running and ready pods")
	}

	value, timestamp, err := h.getCustomMetricForPods(metricSpec, namespace, readyPods)
	if err != nil {
		return nil, err
	}

	info := &UtilizationInfo{
		ReadyPodsCount:   len(readyPods),
		UnreadyPodsCount: unreadyPodsCount,
		OldestTimestamp:  timestamp,
		Utilization:      value.floatValue,
	}

	return info, nil
}

func (h *HeapsterMetricsClient) getCustomMetricForPods(metricSpec metricDefinition, namespace string, podNames []string) (*intAndFloat, time.Time, error) {

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
		missing := make([]string, 0)
		for i, expected := range podNames {
			if len(metrics.Items) > i && len(metrics.Items[i].Metrics) == 0 {
				missing = append(missing, expected)
			}
		}
		hint := ""
		if len(missing) > 0 {
			hint = fmt.Sprintf(" (sample missing pod: %s/%s)", namespace, missing[0])
		}
		return nil, time.Time{}, fmt.Errorf("metrics obtained for %d/%d of pods%s", count, len(podNames), hint)
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
