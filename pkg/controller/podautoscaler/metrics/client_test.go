/*
Copyright 2017 The Kubernetes Authors.

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
	"context"
	"fmt"
	"testing"
	"time"

	autoscalingapi "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	core "k8s.io/client-go/testing"
	cmapi "k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
	emapi "k8s.io/metrics/pkg/apis/external_metrics/v1beta1"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
	cmfake "k8s.io/metrics/pkg/client/custom_metrics/fake"
	emfake "k8s.io/metrics/pkg/client/external_metrics/fake"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var fixedTimestamp = time.Date(2015, time.November, 10, 12, 30, 0, 0, time.UTC)

// timestamp is used for establishing order on metricPoints
type metricPoint struct {
	level     uint64
	timestamp int
}

type restClientTestCase struct {
	// "timestamps" here are actually the offset in minutes from a base timestamp
	targetTimestamp      int
	window               time.Duration
	reportedMetricPoints []metricPoint
	reportedPodMetrics   []map[string]int64
	singleObject         *autoscalingapi.CrossVersionObjectReference

	namespace           string
	selector            labels.Selector
	metricName          string
	metricLabelSelector labels.Selector
}

const (
	testNamespace = "test-namespace"
	podNamePrefix = "test-pod"
)

func AddListPodMetricsReactor(fakeMetricsClient *metricsfake.Clientset, tc *restClientTestCase) {
	podLabels := map[string]string{"name": podNamePrefix}
	fakeMetricsClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		metrics := &metricsapi.PodMetricsList{}
		for i, containers := range tc.reportedPodMetrics {
			metric := metricsapi.PodMetrics{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("%s-%d", podNamePrefix, i),
					Namespace: testNamespace,
					Labels:    podLabels,
				},
				Timestamp:  metav1.Time{Time: offsetTimestampBy(tc.targetTimestamp)},
				Window:     metav1.Duration{Duration: tc.window},
				Containers: []metricsapi.ContainerMetrics{},
			}
			for containerName, cpu := range containers {
				cm := metricsapi.ContainerMetrics{
					Name: containerName,
					Usage: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(
							cpu,
							resource.DecimalSI),
						v1.ResourceMemory: *resource.NewQuantity(
							int64(1024*1024),
							resource.BinarySI),
					},
				}
				metric.Containers = append(metric.Containers, cm)
			}
			metrics.Items = append(metrics.Items, metric)
		}
		return true, metrics, nil
	})
}

func AddListExternalMetricsReactor(fakeEMClient *emfake.FakeExternalMetricsClient, tc *restClientTestCase) {
	fakeEMClient.AddReactor("list", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		metrics := emapi.ExternalMetricValueList{}
		for _, metricPoint := range tc.reportedMetricPoints {
			timestamp := offsetTimestampBy(metricPoint.timestamp)
			metric := emapi.ExternalMetricValue{
				Value:      *resource.NewMilliQuantity(int64(metricPoint.level), resource.DecimalSI),
				Timestamp:  metav1.Time{Time: timestamp},
				MetricName: tc.metricName,
			}
			metrics.Items = append(metrics.Items, metric)
		}
		return true, &metrics, nil
	})
}

func AddGetCustomMetricsReactor(fakeCMClient *cmfake.FakeCustomMetricsClient, tc *restClientTestCase) {
	fakeCMClient.AddReactor("get", "*", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		getForAction := action.(cmfake.GetForAction)
		if getForAction.GetName() == "*" {
			metrics := cmapi.MetricValueList{}
			for i, metricPoint := range tc.reportedMetricPoints {
				timestamp := offsetTimestampBy(metricPoint.timestamp)
				metric := cmapi.MetricValue{
					DescribedObject: v1.ObjectReference{
						Kind:       "Pod",
						APIVersion: "v1",
						Name:       fmt.Sprintf("%s-%d", podNamePrefix, i),
					},
					Value:     *resource.NewMilliQuantity(int64(metricPoint.level), resource.DecimalSI),
					Timestamp: metav1.Time{Time: timestamp},
					Metric: cmapi.MetricIdentifier{
						Name: tc.metricName,
					},
				}
				metrics.Items = append(metrics.Items, metric)
			}
			return true, &metrics, nil
		}

		metricPoint := tc.reportedMetricPoints[0]
		timestamp := offsetTimestampBy(metricPoint.timestamp)

		metrics := &cmapi.MetricValueList{
			Items: []cmapi.MetricValue{
				{
					DescribedObject: v1.ObjectReference{
						Kind:       tc.singleObject.Kind,
						APIVersion: tc.singleObject.APIVersion,
						Name:       tc.singleObject.Name,
					},
					Timestamp: metav1.Time{Time: timestamp},
					Metric: cmapi.MetricIdentifier{
						Name: tc.metricName,
					},
					Value: *resource.NewMilliQuantity(int64(metricPoint.level), resource.DecimalSI),
				},
			},
		}
		return true, metrics, nil
	})
}

func TestRESTClientResourceMetrics(t *testing.T) {
	tests := []struct {
		name          string
		fixture       restClientTestCase
		container     string
		expectedInfo  PodMetricsInfo
		expectedError string
	}{
		{
			name: "pod CPU",
			fixture: restClientTestCase{
				reportedPodMetrics: []map[string]int64{{"test": 5000}, {"test": 5000}, {"test": 5000}},
				targetTimestamp:    1,
				window:             30 * time.Second,
				namespace:          testNamespace,
				selector:           labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
			},
			expectedInfo: PodMetricsInfo{
				"test-pod-0": {Value: 5000, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
				"test-pod-1": {Value: 5000, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
				"test-pod-2": {Value: 5000, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
			},
		},
		{
			name: "container CPU",
			fixture: restClientTestCase{
				reportedPodMetrics: []map[string]int64{{"test-1": 5000, "test-2": 500}, {"test-1": 5000, "test-2": 500}, {"test-1": 5000, "test-2": 500}},
				targetTimestamp:    1,
				window:             30 * time.Second,
				namespace:          testNamespace,
				selector:           labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
			},
			container: "test-1",
			expectedInfo: PodMetricsInfo{
				"test-pod-0": {Value: 5000, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
				"test-pod-1": {Value: 5000, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
				"test-pod-2": {Value: 5000, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
			},
		},
		{
			name: "empty metrics",
			fixture: restClientTestCase{
				reportedPodMetrics: []map[string]int64{},
				namespace:          testNamespace,
				selector:           labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
			},
			expectedError: "no metrics returned from resource metrics API",
		},
		{
			name: "empty metrics for one pod",
			fixture: restClientTestCase{
				reportedPodMetrics: []map[string]int64{{"test-1": 100}, {"test-1": 300, "test-2": 400}, {}},
				targetTimestamp:    1,
				window:             30 * time.Second,
				namespace:          testNamespace,
				selector:           labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
			},
			expectedInfo: PodMetricsInfo{
				"test-pod-0": {Value: 100, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
				"test-pod-1": {Value: 700, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
			},
		},
		{
			name: "container CPU empty metrics for one pod",
			fixture: restClientTestCase{
				reportedPodMetrics: []map[string]int64{{"test-1": 100}, {"test-1": 300, "test-2": 400}, {}},
				targetTimestamp:    1,
				window:             30 * time.Second,
				namespace:          testNamespace,
				selector:           labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
			},
			container: "test-1",
			expectedInfo: PodMetricsInfo{
				"test-pod-0": {Value: 100, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
				"test-pod-1": {Value: 300, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
			},
		},
		{
			name: "container CPU returns only found container",
			fixture: restClientTestCase{
				reportedPodMetrics: []map[string]int64{{"test-1": 100}, {"test-1": 300, "test-2": 400}, {}},
				targetTimestamp:    1,
				window:             30 * time.Second,
				namespace:          testNamespace,
				selector:           labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
			},
			container: "test-2",
			expectedInfo: PodMetricsInfo{
				"test-pod-1": {Value: 400, Timestamp: offsetTimestampBy(1), Window: 30 * time.Second},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeMetricsClient := &metricsfake.Clientset{}
			AddListPodMetricsReactor(fakeMetricsClient, &tt.fixture)
			metricsClient := NewRESTMetricsClient(fakeMetricsClient.MetricsV1(), &cmfake.FakeCustomMetricsClient{}, &emfake.FakeExternalMetricsClient{})
			info, timestamp, err := metricsClient.GetResourceMetric(context.TODO(), v1.ResourceCPU, tt.fixture.namespace, tt.fixture.selector, tt.container)
			if tt.expectedError != "" {
				require.ErrorContains(t, err, tt.expectedError)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.expectedInfo, info)
			assert.True(t, offsetTimestampBy(tt.fixture.targetTimestamp).Equal(timestamp))
		})
	}
}

func TestRESTClientCustomMetrics(t *testing.T) {
	tests := []struct {
		name                string
		fixture             restClientTestCase
		expectedInfo        PodMetricsInfo
		expectedObjectValue int64
		expectedError       string
	}{
		{
			name: "QPS",
			fixture: restClientTestCase{
				metricName:           "qps",
				reportedMetricPoints: []metricPoint{{10000, 1}, {20000, 1}, {10000, 1}},
				namespace:            testNamespace,
				selector:             labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
			},
			expectedInfo: PodMetricsInfo{
				"test-pod-0": {Value: 10000, Timestamp: offsetTimestampBy(1), Window: metricServerDefaultMetricWindow},
				"test-pod-1": {Value: 20000, Timestamp: offsetTimestampBy(1), Window: metricServerDefaultMetricWindow},
				"test-pod-2": {Value: 10000, Timestamp: offsetTimestampBy(1), Window: metricServerDefaultMetricWindow},
			},
		},
		{
			name: "QPS sum equal zero",
			fixture: restClientTestCase{
				metricName:           "qps",
				reportedMetricPoints: []metricPoint{{0, 0}, {0, 0}, {0, 0}},
				namespace:            testNamespace,
				selector:             labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
			},
			expectedInfo: PodMetricsInfo{
				"test-pod-0": {Value: 0, Timestamp: offsetTimestampBy(0), Window: metricServerDefaultMetricWindow},
				"test-pod-1": {Value: 0, Timestamp: offsetTimestampBy(0), Window: metricServerDefaultMetricWindow},
				"test-pod-2": {Value: 0, Timestamp: offsetTimestampBy(0), Window: metricServerDefaultMetricWindow},
			},
		},
		{
			name: "QPS empty metrics",
			fixture: restClientTestCase{
				metricName:           "qps",
				reportedMetricPoints: []metricPoint{},
				namespace:            testNamespace,
				selector:             labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
			},
			expectedError: "no metrics returned from custom metrics API",
		},
		{
			name: "single object",
			fixture: restClientTestCase{
				metricName:           "queue-length",
				reportedMetricPoints: []metricPoint{{10, 1}},
				namespace:            testNamespace,
				selector:             labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
				singleObject: &autoscalingapi.CrossVersionObjectReference{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "some-dep",
				},
			},
			expectedObjectValue: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeCMClient := &cmfake.FakeCustomMetricsClient{}
			AddGetCustomMetricsReactor(fakeCMClient, &tt.fixture)
			metricsClient := NewRESTMetricsClient((&metricsfake.Clientset{}).MetricsV1(), fakeCMClient, &emfake.FakeExternalMetricsClient{})

			if tt.fixture.singleObject != nil {
				val, timestamp, err := metricsClient.GetObjectMetric(tt.fixture.metricName, tt.fixture.namespace, tt.fixture.singleObject, tt.fixture.metricLabelSelector)
				if tt.expectedError != "" {
					require.ErrorContains(t, err, tt.expectedError)
					return
				}
				require.NoError(t, err)
				assert.Equal(t, tt.expectedObjectValue, val)
				assert.True(t, offsetTimestampBy(tt.fixture.reportedMetricPoints[0].timestamp).Equal(timestamp))
			} else {
				info, timestamp, err := metricsClient.GetRawMetric(tt.fixture.metricName, tt.fixture.namespace, tt.fixture.selector, tt.fixture.metricLabelSelector)
				if tt.expectedError != "" {
					require.ErrorContains(t, err, tt.expectedError)
					return
				}
				require.NoError(t, err)
				require.Equal(t, tt.expectedInfo, info)
				assert.True(t, offsetTimestampBy(tt.fixture.reportedMetricPoints[0].timestamp).Equal(timestamp))
			}
		})
	}
}

func TestRESTClientExternalMetrics(t *testing.T) {
	tests := []struct {
		name          string
		fixture       restClientTestCase
		expectedInfo  PodMetricsInfo
		expectedError string
	}{
		{
			name: "external values",
			fixture: restClientTestCase{
				metricName:           "external",
				reportedMetricPoints: []metricPoint{{10000, 1}, {20000, 1}, {10000, 1}},
				namespace:            testNamespace,
				selector:             labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
				metricLabelSelector:  labels.SelectorFromSet(map[string]string{"label": "value"}),
			},
			expectedInfo: PodMetricsInfo{
				"external-val-0": {Value: 10000}, "external-val-1": {Value: 20000}, "external-val-2": {Value: 10000},
			},
		},
		{
			name: "external sum equal zero",
			fixture: restClientTestCase{
				metricName:           "external",
				reportedMetricPoints: []metricPoint{{0, 0}, {0, 0}, {0, 0}},
				namespace:            testNamespace,
				selector:             labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
				metricLabelSelector:  labels.SelectorFromSet(map[string]string{"label": "value"}),
			},
			expectedInfo: PodMetricsInfo{
				"external-val-0": {Value: 0}, "external-val-1": {Value: 0}, "external-val-2": {Value: 0},
			},
		},
		{
			name: "external empty metrics",
			fixture: restClientTestCase{
				metricName:           "external",
				reportedMetricPoints: []metricPoint{},
				namespace:            testNamespace,
				selector:             labels.SelectorFromSet(map[string]string{"name": podNamePrefix}),
				metricLabelSelector:  labels.SelectorFromSet(map[string]string{"label": "value"}),
			},
			expectedError: "no metrics returned from external metrics API",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeEMClient := &emfake.FakeExternalMetricsClient{}
			AddListExternalMetricsReactor(fakeEMClient, &tt.fixture)
			metricsClient := NewRESTMetricsClient((&metricsfake.Clientset{}).MetricsV1(), &cmfake.FakeCustomMetricsClient{}, fakeEMClient)
			val, timestamp, err := metricsClient.GetExternalMetric(tt.fixture.metricName, tt.fixture.namespace, tt.fixture.metricLabelSelector)
			if tt.expectedError != "" {
				require.ErrorContains(t, err, tt.expectedError)
				return
			}
			require.NoError(t, err)
			info := make(PodMetricsInfo, len(val))
			for i, metricVal := range val {
				info[fmt.Sprintf("%v-val-%v", tt.fixture.metricName, i)] = PodMetric{Value: metricVal}
			}
			require.Equal(t, tt.expectedInfo, info)
			assert.True(t, offsetTimestampBy(tt.fixture.reportedMetricPoints[0].timestamp).Equal(timestamp))
		})
	}
}

func offsetTimestampBy(t int) time.Time {
	return fixedTimestamp.Add(time.Duration(t) * time.Minute)
}
