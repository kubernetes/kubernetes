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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	core "k8s.io/client-go/testing"

	heapster "k8s.io/heapster/metrics/api/v1/types"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1alpha1"

	"github.com/stretchr/testify/assert"
)

var fixedTimestamp = time.Date(2015, time.November, 10, 12, 30, 0, 0, time.UTC)

func (w fakeResponseWrapper) DoRaw(context.Context) ([]byte, error) {
	return w.raw, nil
}

func (w fakeResponseWrapper) Stream(context.Context) (io.ReadCloser, error) {
	return nil, nil
}

func newFakeResponseWrapper(raw []byte) fakeResponseWrapper {
	return fakeResponseWrapper{raw: raw}
}

type fakeResponseWrapper struct {
	raw []byte
}

// timestamp is used for establishing order on metricPoints
type metricPoint struct {
	level     uint64
	timestamp int
}

type testCase struct {
	desiredMetricValues PodMetricsInfo
	desiredError        error

	replicas              int
	targetTimestamp       int
	window                time.Duration
	reportedMetricsPoints [][]metricPoint
	reportedPodMetrics    [][]int64

	namespace      string
	selector       labels.Selector
	metricSelector labels.Selector
	resourceName   v1.ResourceName
	metricName     string
}

func (tc *testCase) prepareTestClient(t *testing.T) *fake.Clientset {
	namespace := "test-namespace"
	tc.namespace = namespace
	podNamePrefix := "test-pod"
	podLabels := map[string]string{"name": podNamePrefix}
	tc.selector = labels.SelectorFromSet(podLabels)

	// it's a resource test if we have a resource name
	isResource := len(tc.resourceName) > 0

	fakeClient := &fake.Clientset{}

	fakeClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.PodList{}
		for i := 0; i < tc.replicas; i++ {
			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
			pod := buildPod(namespace, podName, podLabels, v1.PodRunning, "1024")
			obj.Items = append(obj.Items, pod)
		}
		return true, obj, nil
	})

	if isResource {
		fakeClient.AddProxyReactor("services", func(action core.Action) (handled bool, ret restclient.ResponseWrapper, err error) {
			metrics := metricsapi.PodMetricsList{}
			for i, containers := range tc.reportedPodMetrics {
				metric := metricsapi.PodMetrics{
					ObjectMeta: metav1.ObjectMeta{
						Name:      fmt.Sprintf("%s-%d", podNamePrefix, i),
						Namespace: namespace,
					},
					Timestamp:  metav1.Time{Time: offsetTimestampBy(tc.targetTimestamp)},
					Window:     metav1.Duration{Duration: tc.window},
					Containers: []metricsapi.ContainerMetrics{},
				}
				for j, cpu := range containers {
					cm := metricsapi.ContainerMetrics{
						Name: fmt.Sprintf("%s-%d-container-%d", podNamePrefix, i, j),
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
			heapsterRawMemResponse, _ := json.Marshal(&metrics)
			return true, newFakeResponseWrapper(heapsterRawMemResponse), nil
		})
	} else {
		fakeClient.AddProxyReactor("services", func(action core.Action) (handled bool, ret restclient.ResponseWrapper, err error) {
			metrics := heapster.MetricResultList{}
			var latestTimestamp time.Time
			for _, reportedMetricPoints := range tc.reportedMetricsPoints {
				var heapsterMetricPoints []heapster.MetricPoint
				for _, reportedMetricPoint := range reportedMetricPoints {
					timestamp := offsetTimestampBy(reportedMetricPoint.timestamp)
					if latestTimestamp.Before(timestamp) {
						latestTimestamp = timestamp
					}
					heapsterMetricPoint := heapster.MetricPoint{Timestamp: timestamp, Value: reportedMetricPoint.level, FloatValue: nil}
					heapsterMetricPoints = append(heapsterMetricPoints, heapsterMetricPoint)
				}
				metric := heapster.MetricResult{
					Metrics:         heapsterMetricPoints,
					LatestTimestamp: latestTimestamp,
				}
				metrics.Items = append(metrics.Items, metric)
			}
			heapsterRawMemResponse, _ := json.Marshal(&metrics)
			return true, newFakeResponseWrapper(heapsterRawMemResponse), nil
		})
	}

	return fakeClient
}

func buildPod(namespace, podName string, podLabels map[string]string, phase v1.PodPhase, request string) v1.Pod {
	return v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			Labels:    podLabels,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse(request),
						},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: phase,
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodReady,
					Status: v1.ConditionTrue,
				},
			},
		},
	}
}

func (tc *testCase) verifyResults(t *testing.T, metrics PodMetricsInfo, timestamp time.Time, err error) {
	if tc.desiredError != nil {
		assert.Error(t, err, "there should be an error retrieving the metrics")
		assert.Contains(t, fmt.Sprintf("%v", err), fmt.Sprintf("%v", tc.desiredError), "the error message should be eas expected")
		return
	}
	assert.NoError(t, err, "there should be no error retrieving the metrics")
	assert.NotNil(t, metrics, "there should be metrics returned")
	if len(metrics) != len(tc.desiredMetricValues) {
		t.Errorf("Not equal:\nexpected: %v\nactual: %v", tc.desiredMetricValues, metrics)
	} else {
		for k, m := range metrics {
			if !m.Timestamp.Equal(tc.desiredMetricValues[k].Timestamp) ||
				m.Window != tc.desiredMetricValues[k].Window ||
				m.Value != tc.desiredMetricValues[k].Value {
				t.Errorf("Not equal:\nexpected: %v\nactual: %v", tc.desiredMetricValues, metrics)
				break
			}
		}
	}

	targetTimestamp := offsetTimestampBy(tc.targetTimestamp)
	assert.True(t, targetTimestamp.Equal(timestamp), fmt.Sprintf("the timestamp should be as expected (%s) but was %s", targetTimestamp, timestamp))
}

func (tc *testCase) runTest(t *testing.T) {
	testClient := tc.prepareTestClient(t)
	metricsClient := NewHeapsterMetricsClient(testClient, DefaultHeapsterNamespace, DefaultHeapsterScheme, DefaultHeapsterService, DefaultHeapsterPort)
	isResource := len(tc.resourceName) > 0
	if isResource {
		info, timestamp, err := metricsClient.GetResourceMetric(tc.resourceName, tc.namespace, tc.selector, "")
		tc.verifyResults(t, info, timestamp, err)
	} else {
		info, timestamp, err := metricsClient.GetRawMetric(tc.metricName, tc.namespace, tc.selector, tc.metricSelector)
		tc.verifyResults(t, info, timestamp, err)
	}
}

func TestCPU(t *testing.T) {
	targetTimestamp := 1
	window := 30 * time.Second
	tc := testCase{
		replicas: 3,
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": PodMetric{Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
			"test-pod-1": PodMetric{Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
			"test-pod-2": PodMetric{Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
		},
		resourceName:       v1.ResourceCPU,
		targetTimestamp:    targetTimestamp,
		window:             window,
		reportedPodMetrics: [][]int64{{5000}, {5000}, {5000}},
	}
	tc.runTest(t)
}

func TestQPS(t *testing.T) {
	targetTimestamp := 1
	tc := testCase{
		replicas: 3,
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": PodMetric{Value: 10000, Timestamp: offsetTimestampBy(targetTimestamp), Window: heapsterDefaultMetricWindow},
			"test-pod-1": PodMetric{Value: 20000, Timestamp: offsetTimestampBy(targetTimestamp), Window: heapsterDefaultMetricWindow},
			"test-pod-2": PodMetric{Value: 10000, Timestamp: offsetTimestampBy(targetTimestamp), Window: heapsterDefaultMetricWindow},
		},
		metricName:            "qps",
		targetTimestamp:       targetTimestamp,
		reportedMetricsPoints: [][]metricPoint{{{10, 1}}, {{20, 1}}, {{10, 1}}},
	}
	tc.runTest(t)
}

func TestQpsSumEqualZero(t *testing.T) {
	targetTimestamp := 0
	tc := testCase{
		replicas: 3,
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": PodMetric{Value: 0, Timestamp: offsetTimestampBy(targetTimestamp), Window: heapsterDefaultMetricWindow},
			"test-pod-1": PodMetric{Value: 0, Timestamp: offsetTimestampBy(targetTimestamp), Window: heapsterDefaultMetricWindow},
			"test-pod-2": PodMetric{Value: 0, Timestamp: offsetTimestampBy(targetTimestamp), Window: heapsterDefaultMetricWindow},
		},
		metricName:            "qps",
		targetTimestamp:       targetTimestamp,
		reportedMetricsPoints: [][]metricPoint{{{0, 0}}, {{0, 0}}, {{0, 0}}},
	}
	tc.runTest(t)
}

func TestCPUMoreMetrics(t *testing.T) {
	targetTimestamp := 10
	window := 30 * time.Second
	tc := testCase{
		replicas: 5,
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": PodMetric{Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
			"test-pod-1": PodMetric{Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
			"test-pod-2": PodMetric{Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
			"test-pod-3": PodMetric{Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
			"test-pod-4": PodMetric{Value: 5000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
		},
		resourceName:       v1.ResourceCPU,
		targetTimestamp:    targetTimestamp,
		window:             window,
		reportedPodMetrics: [][]int64{{1000, 2000, 2000}, {5000}, {1000, 1000, 1000, 2000}, {4000, 1000}, {5000}},
	}
	tc.runTest(t)
}

func TestCPUMissingMetrics(t *testing.T) {
	targetTimestamp := 0
	window := 30 * time.Second
	tc := testCase{
		replicas: 3,
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": PodMetric{Value: 4000, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
		},
		resourceName:       v1.ResourceCPU,
		targetTimestamp:    targetTimestamp,
		window:             window,
		reportedPodMetrics: [][]int64{{4000}},
	}
	tc.runTest(t)
}

func TestQpsMissingMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		desiredError:          fmt.Errorf("requested metrics for 3 pods, got metrics for 1"),
		metricName:            "qps",
		targetTimestamp:       1,
		reportedMetricsPoints: [][]metricPoint{{{4000, 4}}},
	}
	tc.runTest(t)
}

func TestQpsSuperfluousMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		desiredError:          fmt.Errorf("requested metrics for 3 pods, got metrics for 6"),
		metricName:            "qps",
		reportedMetricsPoints: [][]metricPoint{{{1000, 1}}, {{2000, 4}}, {{2000, 1}}, {{4000, 5}}, {{2000, 1}}, {{4000, 4}}},
	}
	tc.runTest(t)
}

func TestCPUEmptyMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		resourceName:          v1.ResourceCPU,
		desiredError:          fmt.Errorf("no metrics returned from heapster"),
		reportedMetricsPoints: [][]metricPoint{},
		reportedPodMetrics:    [][]int64{},
	}
	tc.runTest(t)
}

func TestQpsEmptyEntries(t *testing.T) {
	targetTimestamp := 4
	tc := testCase{
		replicas:   3,
		metricName: "qps",
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": PodMetric{Value: 4000000, Timestamp: offsetTimestampBy(targetTimestamp), Window: heapsterDefaultMetricWindow},
			"test-pod-2": PodMetric{Value: 2000000, Timestamp: offsetTimestampBy(targetTimestamp), Window: heapsterDefaultMetricWindow},
		},
		targetTimestamp:       targetTimestamp,
		reportedMetricsPoints: [][]metricPoint{{{4000, 4}}, {}, {{2000, 4}}},
	}
	tc.runTest(t)
}

func TestCPUZeroReplicas(t *testing.T) {
	tc := testCase{
		replicas:           0,
		resourceName:       v1.ResourceCPU,
		desiredError:       fmt.Errorf("no metrics returned from heapster"),
		reportedPodMetrics: [][]int64{},
	}
	tc.runTest(t)
}

func TestCPUEmptyMetricsForOnePod(t *testing.T) {
	targetTimestamp := 0
	window := 30 * time.Second
	tc := testCase{
		replicas:     3,
		resourceName: v1.ResourceCPU,
		desiredMetricValues: PodMetricsInfo{
			"test-pod-0": PodMetric{Value: 100, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
			"test-pod-1": PodMetric{Value: 700, Timestamp: offsetTimestampBy(targetTimestamp), Window: window},
		},
		targetTimestamp:    targetTimestamp,
		window:             window,
		reportedPodMetrics: [][]int64{{100}, {300, 400}, {}},
	}
	tc.runTest(t)
}

func offsetTimestampBy(t int) time.Time {
	return fixedTimestamp.Add(time.Duration(t) * time.Minute)
}
