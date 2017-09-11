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
	heapster "k8s.io/heapster/metrics/api/v1/types"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1alpha1"

	autoscaling "k8s.io/api/autoscaling/v2beta1"
	"k8s.io/api/core/v1"
	clientgov1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
)

const (
	DefaultHeapsterNamespace = "kube-system"
	DefaultHeapsterScheme    = "http"
	DefaultHeapsterService   = "heapster"
	DefaultHeapsterPort      = "" // use the first exposed port on the service
)

var heapsterQueryStart = -5 * time.Minute

type HeapsterMetricsClient struct {
	services        v1core.ServiceInterface
	podsGetter      v1core.PodsGetter
	heapsterScheme  string
	heapsterService string
	heapsterPort    string
}

func NewHeapsterMetricsClient(client clientset.Interface, namespace, scheme, service, port string) MetricsClient {
	return &HeapsterMetricsClient{
		services:        client.Core().Services(namespace),
		podsGetter:      client.Core(),
		heapsterScheme:  scheme,
		heapsterService: service,
		heapsterPort:    port,
	}
}

func (h *HeapsterMetricsClient) GetResourceMetric(resource v1.ResourceName, namespace string, selector labels.Selector) (PodMetricsInfo, time.Time, error) {
	metricPath := fmt.Sprintf("/apis/metrics/v1alpha1/namespaces/%s/pods", namespace)
	params := map[string]string{"labelSelector": selector.String()}

	resultRaw, err := h.services.
		ProxyGet(h.heapsterScheme, h.heapsterService, h.heapsterPort, metricPath, params).
		DoRaw()
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to get pod resource metrics: %v", err)
	}

	glog.V(4).Infof("Heapster metrics result: %s", string(resultRaw))

	metrics := metricsapi.PodMetricsList{}
	err = json.Unmarshal(resultRaw, &metrics)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to unmarshal heapster response: %v", err)
	}

	if len(metrics.Items) == 0 {
		return nil, time.Time{}, fmt.Errorf("no metrics returned from heapster")
	}

	res := make(PodMetricsInfo, len(metrics.Items))

	for _, m := range metrics.Items {
		podSum := int64(0)
		missing := len(m.Containers) == 0
		for _, c := range m.Containers {
			resValue, found := c.Usage[clientgov1.ResourceName(resource)]
			if !found {
				missing = true
				glog.V(2).Infof("missing resource metric %v for container %s in pod %s/%s", resource, c.Name, namespace, m.Name)
				continue
			}
			podSum += resValue.MilliValue()
		}

		if !missing {
			res[m.Name] = int64(podSum)
		}
	}

	timestamp := metrics.Items[0].Timestamp.Time

	return res, timestamp, nil
}

func (h *HeapsterMetricsClient) GetRawMetric(metricName string, namespace string, selector labels.Selector) (PodMetricsInfo, time.Time, error) {
	podList, err := h.podsGetter.Pods(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to get pod list while fetching metrics: %v", err)
	}

	if len(podList.Items) == 0 {
		return nil, time.Time{}, fmt.Errorf("no pods matched the provided selector")
	}

	podNames := make([]string, len(podList.Items))
	for i, pod := range podList.Items {
		podNames[i] = pod.Name
	}

	now := time.Now()

	startTime := now.Add(heapsterQueryStart)
	metricPath := fmt.Sprintf("/api/v1/model/namespaces/%s/pod-list/%s/metrics/%s",
		namespace,
		strings.Join(podNames, ","),
		metricName)

	resultRaw, err := h.services.
		ProxyGet(h.heapsterScheme, h.heapsterService, h.heapsterPort, metricPath, map[string]string{"start": startTime.Format(time.RFC3339)}).
		DoRaw()
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to get pod metrics: %v", err)
	}

	var metrics heapster.MetricResultList
	err = json.Unmarshal(resultRaw, &metrics)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("failed to unmarshal heapster response: %v", err)
	}

	glog.V(4).Infof("Heapster metrics result: %s", string(resultRaw))

	if len(metrics.Items) != len(podNames) {
		// if we get too many metrics or two few metrics, we have no way of knowing which metric goes to which pod
		// (note that Heapster returns *empty* metric items when a pod does not exist or have that metric, so this
		// does not cover the "missing metric entry" case)
		return nil, time.Time{}, fmt.Errorf("requested metrics for %v pods, got metrics for %v", len(podNames), len(metrics.Items))
	}

	var timestamp *time.Time
	res := make(PodMetricsInfo, len(metrics.Items))
	for i, podMetrics := range metrics.Items {
		val, podTimestamp, hadMetrics := collapseTimeSamples(podMetrics, time.Minute)
		if hadMetrics {
			res[podNames[i]] = val
			if timestamp == nil || podTimestamp.Before(*timestamp) {
				timestamp = &podTimestamp
			}
		}
	}

	if timestamp == nil {
		timestamp = &time.Time{}
	}

	return res, *timestamp, nil
}

func (h *HeapsterMetricsClient) GetObjectMetric(metricName string, namespace string, objectRef *autoscaling.CrossVersionObjectReference) (int64, time.Time, error) {
	return 0, time.Time{}, fmt.Errorf("object metrics are not yet supported")
}

func collapseTimeSamples(metrics heapster.MetricResult, duration time.Duration) (int64, time.Time, bool) {
	floatSum := float64(0)
	intSum := int64(0)
	intSumCount := 0
	floatSumCount := 0

	var newest *heapster.MetricPoint // creation time of the newest sample for this pod
	for i, metricPoint := range metrics.Metrics {
		if newest == nil || newest.Timestamp.Before(metricPoint.Timestamp) {
			newest = &metrics.Metrics[i]
		}
	}
	if newest != nil {
		for _, metricPoint := range metrics.Metrics {
			if metricPoint.Timestamp.Add(duration).After(newest.Timestamp) {
				intSum += int64(metricPoint.Value)
				intSumCount++
				if metricPoint.FloatValue != nil {
					floatSum += *metricPoint.FloatValue
					floatSumCount++
				}
			}
		}

		if newest.FloatValue != nil {
			return int64(floatSum / float64(floatSumCount) * 1000), newest.Timestamp, true
		} else {
			return (intSum * 1000) / int64(intSumCount), newest.Timestamp, true
		}
	}

	return 0, time.Time{}, false
}
