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
	"io"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	_ "k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"

	heapster "k8s.io/heapster/metrics/api/v1/types"

	"github.com/stretchr/testify/assert"
)

var fixedTimestamp = time.Date(2015, time.November, 10, 12, 30, 0, 0, time.UTC)

func (w fakeResponseWrapper) DoRaw() ([]byte, error) {
	return w.raw, nil
}

func (w fakeResponseWrapper) Stream() (io.ReadCloser, error) {
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
	replicas              int
	desiredValue          float64
	desiredError          error
	targetResource        string
	targetTimestamp       int
	reportedMetricsPoints [][]metricPoint
	namespace             string
	podListOverride       *api.PodList
	selector              labels.Selector
}

func (tc *testCase) prepareTestClient(t *testing.T) *fake.Clientset {
	namespace := "test-namespace"
	tc.namespace = namespace
	podNamePrefix := "test-pod"
	podLabels := map[string]string{"name": podNamePrefix}
	tc.selector = labels.SelectorFromSet(podLabels)

	fakeClient := &fake.Clientset{}

	fakeClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		if tc.podListOverride != nil {
			return true, tc.podListOverride, nil
		}
		obj := &api.PodList{}
		for i := 0; i < tc.replicas; i++ {
			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
			pod := buildPod(namespace, podName, podLabels, api.PodRunning)
			obj.Items = append(obj.Items, pod)
		}
		return true, obj, nil
	})

	fakeClient.AddProxyReactor("services", func(action core.Action) (handled bool, ret restclient.ResponseWrapper, err error) {
		metrics := heapster.MetricResultList{}
		var latestTimestamp time.Time
		for _, reportedMetricPoints := range tc.reportedMetricsPoints {
			var heapsterMetricPoints []heapster.MetricPoint
			for _, reportedMetricPoint := range reportedMetricPoints {
				timestamp := fixedTimestamp.Add(time.Duration(reportedMetricPoint.timestamp) * time.Minute)
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

	return fakeClient
}

func buildPod(namespace, podName string, podLabels map[string]string, phase api.PodPhase) api.Pod {
	return api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			Labels:    podLabels,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Requests: api.ResourceList{
							api.ResourceCPU: resource.MustParse("10"),
						},
					},
				},
			},
		},
		Status: api.PodStatus{
			Phase: phase,
		},
	}
}

func (tc *testCase) verifyResults(t *testing.T, val *float64, timestamp time.Time, err error) {
	assert.Equal(t, tc.desiredError, err)
	if tc.desiredError != nil {
		return
	}
	assert.NotNil(t, val)
	assert.True(t, tc.desiredValue-0.001 < *val)
	assert.True(t, tc.desiredValue+0.001 > *val)

	targetTimestamp := fixedTimestamp.Add(time.Duration(tc.targetTimestamp) * time.Minute)
	assert.Equal(t, targetTimestamp, timestamp)
}

func (tc *testCase) runTest(t *testing.T) {
	testClient := tc.prepareTestClient(t)
	metricsClient := NewHeapsterMetricsClient(testClient, DefaultHeapsterNamespace, DefaultHeapsterScheme, DefaultHeapsterService, DefaultHeapsterPort)
	if tc.targetResource == "cpu-usage" {
		val, _, timestamp, err := metricsClient.GetCpuConsumptionAndRequestInMillis(tc.namespace, tc.selector)
		fval := float64(val)
		tc.verifyResults(t, &fval, timestamp, err)
	} else {
		val, timestamp, err := metricsClient.GetCustomMetric(tc.targetResource, tc.namespace, tc.selector)
		tc.verifyResults(t, val, timestamp, err)
	}
}

func TestCPU(t *testing.T) {
	tc := testCase{
		replicas:              3,
		desiredValue:          5000,
		targetResource:        "cpu-usage",
		targetTimestamp:       1,
		reportedMetricsPoints: [][]metricPoint{{{5000, 1}}, {{5000, 1}}, {{5000, 1}}},
	}
	tc.runTest(t)
}

func TestCPUPending(t *testing.T) {
	tc := testCase{
		replicas:              4,
		desiredValue:          5000,
		targetResource:        "cpu-usage",
		targetTimestamp:       1,
		reportedMetricsPoints: [][]metricPoint{{{5000, 1}}, {{5000, 1}}, {{5000, 1}}},
		podListOverride:       &api.PodList{},
	}

	namespace := "test-namespace"
	podNamePrefix := "test-pod"
	podLabels := map[string]string{"name": podNamePrefix}
	for i := 0; i < tc.replicas; i++ {
		podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
		pod := buildPod(namespace, podName, podLabels, api.PodRunning)
		tc.podListOverride.Items = append(tc.podListOverride.Items, pod)
	}
	tc.podListOverride.Items[0].Status.Phase = api.PodPending

	tc.runTest(t)
}

func TestCPUAllPending(t *testing.T) {
	tc := testCase{
		replicas:              4,
		targetResource:        "cpu-usage",
		targetTimestamp:       1,
		reportedMetricsPoints: [][]metricPoint{},
		podListOverride:       &api.PodList{},
		desiredError:          fmt.Errorf("no running pods"),
	}

	namespace := "test-namespace"
	podNamePrefix := "test-pod"
	podLabels := map[string]string{"name": podNamePrefix}
	for i := 0; i < tc.replicas; i++ {
		podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
		pod := buildPod(namespace, podName, podLabels, api.PodPending)
		tc.podListOverride.Items = append(tc.podListOverride.Items, pod)
	}
	tc.runTest(t)
}

func TestQPS(t *testing.T) {
	tc := testCase{
		replicas:              3,
		desiredValue:          13.33333,
		targetResource:        "qps",
		targetTimestamp:       1,
		reportedMetricsPoints: [][]metricPoint{{{10, 1}}, {{20, 1}}, {{10, 1}}},
	}
	tc.runTest(t)
}

func TestQPSPending(t *testing.T) {
	tc := testCase{
		replicas:              4,
		desiredValue:          13.33333,
		targetResource:        "qps",
		targetTimestamp:       1,
		reportedMetricsPoints: [][]metricPoint{{{10, 1}}, {{20, 1}}, {{10, 1}}},
		podListOverride:       &api.PodList{},
	}

	namespace := "test-namespace"
	podNamePrefix := "test-pod"
	podLabels := map[string]string{"name": podNamePrefix}
	for i := 0; i < tc.replicas; i++ {
		podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
		pod := buildPod(namespace, podName, podLabels, api.PodRunning)
		tc.podListOverride.Items = append(tc.podListOverride.Items, pod)
	}
	tc.podListOverride.Items[0].Status.Phase = api.PodPending
	tc.runTest(t)
}

func TestQPSAllPending(t *testing.T) {
	tc := testCase{
		replicas:              4,
		desiredError:          fmt.Errorf("no running pods"),
		targetResource:        "qps",
		targetTimestamp:       1,
		reportedMetricsPoints: [][]metricPoint{},
		podListOverride:       &api.PodList{},
	}

	namespace := "test-namespace"
	podNamePrefix := "test-pod"
	podLabels := map[string]string{"name": podNamePrefix}
	for i := 0; i < tc.replicas; i++ {
		podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
		pod := buildPod(namespace, podName, podLabels, api.PodPending)
		tc.podListOverride.Items = append(tc.podListOverride.Items, pod)
	}
	tc.podListOverride.Items[0].Status.Phase = api.PodPending
	tc.runTest(t)
}

func TestCPUSumEqualZero(t *testing.T) {
	tc := testCase{
		replicas:              3,
		desiredValue:          0,
		targetResource:        "cpu-usage",
		targetTimestamp:       0,
		reportedMetricsPoints: [][]metricPoint{{{0, 0}}, {{0, 0}}, {{0, 0}}},
	}
	tc.runTest(t)
}

func TestQpsSumEqualZero(t *testing.T) {
	tc := testCase{
		replicas:              3,
		desiredValue:          0,
		targetResource:        "qps",
		targetTimestamp:       0,
		reportedMetricsPoints: [][]metricPoint{{{0, 0}}, {{0, 0}}, {{0, 0}}},
	}
	tc.runTest(t)
}

func TestCPUMoreMetrics(t *testing.T) {
	tc := testCase{
		replicas:        5,
		desiredValue:    5000,
		targetResource:  "cpu-usage",
		targetTimestamp: 10,
		reportedMetricsPoints: [][]metricPoint{
			{{0, 3}, {0, 6}, {5, 4}, {9000, 10}},
			{{5000, 2}, {10, 5}, {66, 1}, {0, 10}},
			{{5000, 3}, {80, 5}, {6000, 10}},
			{{5000, 3}, {40, 3}, {0, 9}, {200, 2}, {8000, 10}},
			{{5000, 2}, {20, 2}, {2000, 10}}},
	}
	tc.runTest(t)
}

func TestCPUResultIsFloat(t *testing.T) {
	tc := testCase{
		replicas:              6,
		desiredValue:          4783,
		targetResource:        "cpu-usage",
		targetTimestamp:       4,
		reportedMetricsPoints: [][]metricPoint{{{4000, 4}}, {{9500, 4}}, {{3000, 4}}, {{7000, 4}}, {{3200, 4}}, {{2000, 4}}},
	}
	tc.runTest(t)
}

func TestCPUSamplesWithRandomTimestamps(t *testing.T) {
	tc := testCase{
		replicas:        3,
		desiredValue:    3000,
		targetResource:  "cpu-usage",
		targetTimestamp: 3,
		reportedMetricsPoints: [][]metricPoint{
			{{1, 1}, {3000, 5}, {2, 2}},
			{{2, 2}, {1, 1}, {3000, 3}},
			{{3000, 4}, {1, 1}, {2, 2}}},
	}
	tc.runTest(t)
}

func TestCPUMissingMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        "cpu-usage",
		desiredError:          fmt.Errorf("metrics obtained for 1/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{{4000, 4}}},
	}
	tc.runTest(t)
}

func TestQpsMissingMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        "qps",
		desiredError:          fmt.Errorf("metrics obtained for 1/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{{4000, 4}}},
	}
	tc.runTest(t)
}

func TestCPUSuperfluousMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        "cpu-usage",
		desiredError:          fmt.Errorf("metrics obtained for 6/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{{1000, 1}}, {{2000, 4}}, {{2000, 1}}, {{4000, 5}}, {{2000, 1}}, {{4000, 4}}},
	}
	tc.runTest(t)
}

func TestQpsSuperfluousMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        "qps",
		desiredError:          fmt.Errorf("metrics obtained for 6/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{{1000, 1}}, {{2000, 4}}, {{2000, 1}}, {{4000, 5}}, {{2000, 1}}, {{4000, 4}}},
	}
	tc.runTest(t)
}

func TestCPUEmptyMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        "cpu-usage",
		desiredError:          fmt.Errorf("metrics obtained for 0/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{},
	}
	tc.runTest(t)
}

func TestCPUZeroReplicas(t *testing.T) {
	tc := testCase{
		replicas:              0,
		targetResource:        "cpu-usage",
		desiredError:          fmt.Errorf("some pods do not have request for cpu"),
		reportedMetricsPoints: [][]metricPoint{},
	}
	tc.runTest(t)
}

func TestCPUEmptyMetricsForOnePod(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        "cpu-usage",
		desiredError:          fmt.Errorf("metrics obtained for 2/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{}, {{100, 1}}, {{400, 2}, {300, 3}}},
	}
	tc.runTest(t)
}

func TestAggregateSum(t *testing.T) {
	//calculateSumFromTimeSample(metrics heapster.MetricResultList, duration time.Duration) (sum intAndFloat, count int, timestamp time.Time) {
	now := time.Now()
	result := heapster.MetricResultList{
		Items: []heapster.MetricResult{
			{
				Metrics: []heapster.MetricPoint{
					{now, 50, nil},
					{now.Add(-15 * time.Second), 100, nil},
					{now.Add(-60 * time.Second), 100000, nil}},
				LatestTimestamp: now,
			},
		},
	}
	sum, cnt, _ := calculateSumFromTimeSample(result, time.Minute)
	assert.Equal(t, int64(75), sum.intValue)
	assert.InEpsilon(t, 75.0, sum.floatValue, 0.1)
	assert.Equal(t, 1, cnt)
}

func TestAggregateSumSingle(t *testing.T) {
	now := time.Now()
	result := heapster.MetricResultList{
		Items: []heapster.MetricResult{
			{
				Metrics: []heapster.MetricPoint{
					{now, 50, nil},
					{now.Add(-65 * time.Second), 100000, nil}},
				LatestTimestamp: now,
			},
		},
	}
	sum, cnt, _ := calculateSumFromTimeSample(result, time.Minute)
	assert.Equal(t, int64(50), sum.intValue)
	assert.InEpsilon(t, 50.0, sum.floatValue, 0.1)
	assert.Equal(t, 1, cnt)
}

// TODO: add proper tests for request
