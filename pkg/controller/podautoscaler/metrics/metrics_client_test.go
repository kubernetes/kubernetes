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
	"io"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	_ "k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"

	heapster "k8s.io/heapster/metrics/api/v1/types"
	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"

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
	reportedPodMetrics    [][]int64
	namespace             string
	podListOverride       *api.PodList
	selector              labels.Selector
	useMetricsApi         bool
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

	if tc.useMetricsApi {
		fakeClient.AddProxyReactor("services", func(action core.Action) (handled bool, ret restclient.ResponseWrapper, err error) {
			metrics := []*metrics_api.PodMetrics{}
			for i, containers := range tc.reportedPodMetrics {
				metric := &metrics_api.PodMetrics{
					ObjectMeta: v1.ObjectMeta{
						Name:      fmt.Sprintf("%s-%d", podNamePrefix, i),
						Namespace: namespace,
					},
					Timestamp:  unversioned.Time{Time: fixedTimestamp.Add(time.Duration(tc.targetTimestamp) * time.Minute)},
					Containers: []metrics_api.ContainerMetrics{},
				}
				for j, cpu := range containers {
					cm := metrics_api.ContainerMetrics{
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
				metrics = append(metrics, metric)
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
	}

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
	if tc.desiredError != nil {
		assert.Error(t, err)
		assert.Contains(t, fmt.Sprintf("%v", err), fmt.Sprintf("%v", tc.desiredError))
		return
	}
	assert.NoError(t, err)
	assert.NotNil(t, val)
	assert.True(t, tc.desiredValue-0.001 < *val)
	assert.True(t, tc.desiredValue+0.001 > *val)

	targetTimestamp := fixedTimestamp.Add(time.Duration(tc.targetTimestamp) * time.Minute)
	assert.True(t, targetTimestamp.Equal(timestamp))
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
		replicas:           3,
		desiredValue:       5000,
		targetResource:     "cpu-usage",
		targetTimestamp:    1,
		reportedPodMetrics: [][]int64{{5000}, {5000}, {5000}},
		useMetricsApi:      true,
	}
	tc.runTest(t)
}

func TestCPUPending(t *testing.T) {
	tc := testCase{
		replicas:           4,
		desiredValue:       5000,
		targetResource:     "cpu-usage",
		targetTimestamp:    1,
		reportedPodMetrics: [][]int64{{5000}, {5000}, {5000}},
		useMetricsApi:      true,
		podListOverride:    &api.PodList{},
	}

	namespace := "test-namespace"
	podNamePrefix := "test-pod"
	podLabels := map[string]string{"name": podNamePrefix}
	for i := 0; i < tc.replicas; i++ {
		podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
		pod := buildPod(namespace, podName, podLabels, api.PodRunning)
		tc.podListOverride.Items = append(tc.podListOverride.Items, pod)
	}
	tc.podListOverride.Items[3].Status.Phase = api.PodPending

	tc.runTest(t)
}

func TestCPUAllPending(t *testing.T) {
	tc := testCase{
		replicas:           4,
		targetResource:     "cpu-usage",
		targetTimestamp:    1,
		reportedPodMetrics: [][]int64{},
		useMetricsApi:      true,
		podListOverride:    &api.PodList{},
		desiredError:       fmt.Errorf("no running pods"),
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
		replicas:           3,
		desiredValue:       0,
		targetResource:     "cpu-usage",
		targetTimestamp:    0,
		reportedPodMetrics: [][]int64{{0}, {0}, {0}},
		useMetricsApi:      true,
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
		replicas:           5,
		desiredValue:       5000,
		targetResource:     "cpu-usage",
		targetTimestamp:    10,
		reportedPodMetrics: [][]int64{{1000, 2000, 2000}, {5000}, {1000, 1000, 1000, 2000}, {4000, 1000}, {5000}},
		useMetricsApi:      true,
	}
	tc.runTest(t)
}

func TestCPUMissingMetrics(t *testing.T) {
	tc := testCase{
		replicas:           3,
		targetResource:     "cpu-usage",
		desiredError:       fmt.Errorf("metrics obtained for 1/3 of pods"),
		reportedPodMetrics: [][]int64{{4000}},
		useMetricsApi:      true,
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
		replicas:           3,
		targetResource:     "cpu-usage",
		desiredError:       fmt.Errorf("metrics obtained for 6/3 of pods"),
		reportedPodMetrics: [][]int64{{1000}, {2000}, {4000}, {4000}, {2000}, {4000}},
		useMetricsApi:      true,
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
		reportedPodMetrics:    [][]int64{},
		useMetricsApi:         true,
	}
	tc.runTest(t)
}

func TestCPUZeroReplicas(t *testing.T) {
	tc := testCase{
		replicas:           0,
		targetResource:     "cpu-usage",
		desiredError:       fmt.Errorf("some pods do not have request for cpu"),
		reportedPodMetrics: [][]int64{},
		useMetricsApi:      true,
	}
	tc.runTest(t)
}

func TestCPUEmptyMetricsForOnePod(t *testing.T) {
	tc := testCase{
		replicas:           3,
		targetResource:     "cpu-usage",
		desiredError:       fmt.Errorf("metrics obtained for 2/3 of pods (sample missing pod: test-namespace/test-pod-2)"),
		reportedPodMetrics: [][]int64{{100}, {300, 400}},
		useMetricsApi:      true,
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
					{Timestamp: now, Value: 50, FloatValue: nil},
					{Timestamp: now.Add(-15 * time.Second), Value: 100, FloatValue: nil},
					{Timestamp: now.Add(-60 * time.Second), Value: 100000, FloatValue: nil}},
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
					{Timestamp: now, Value: 50, FloatValue: nil},
					{Timestamp: now.Add(-65 * time.Second), Value: 100000, FloatValue: nil}},
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
