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
	_ "k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"

	heapster "k8s.io/heapster/api/v1/types"

	"github.com/stretchr/testify/assert"
)

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
	desiredValue          int64
	desiredError          error
	targetResource        api.ResourceName
	reportedMetricsPoints [][]metricPoint
	namespace             string
	selector              map[string]string
}

func (tc *testCase) prepareTestClient(t *testing.T) *testclient.Fake {
	namespace := "test-namespace"
	tc.namespace = namespace
	podNamePrefix := "test-pod"
	selector := map[string]string{"name": podNamePrefix}
	tc.selector = selector

	fakeClient := &testclient.Fake{}

	fakeClient.AddReactor("list", "pods", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
		obj := &api.PodList{}
		for i := 0; i < tc.replicas; i++ {
			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
			pod := api.Pod{
				Status: api.PodStatus{
					Phase: api.PodRunning,
				},
				ObjectMeta: api.ObjectMeta{
					Name:      podName,
					Namespace: namespace,
					Labels:    selector,
				},
			}
			obj.Items = append(obj.Items, pod)
		}
		return true, obj, nil
	})

	fakeClient.AddProxyReactor("services", func(action testclient.Action) (handled bool, ret client.ResponseWrapper, err error) {
		metrics := heapster.MetricResultList{}
		firstTimestamp := time.Now()
		var latestTimestamp time.Time
		for _, reportedMetricPoints := range tc.reportedMetricsPoints {
			var heapsterMetricPoints []heapster.MetricPoint
			for _, reportedMetricPoint := range reportedMetricPoints {
				timestamp := firstTimestamp.Add(time.Duration(reportedMetricPoint.timestamp) * time.Minute)
				if latestTimestamp.Before(timestamp) {
					latestTimestamp = timestamp
				}
				heapsterMetricPoint := heapster.MetricPoint{timestamp, reportedMetricPoint.level}
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

func (tc *testCase) verifyResults(t *testing.T, val *extensions.ResourceConsumption, err error) {
	assert.Equal(t, tc.desiredError, err)
	if tc.desiredError != nil {
		return
	}
	if tc.targetResource == api.ResourceCPU {
		assert.Equal(t, tc.desiredValue, val.Quantity.MilliValue())
	}
	if tc.targetResource == api.ResourceMemory {
		assert.Equal(t, tc.desiredValue, val.Quantity.Value())
	}
}

func (tc *testCase) runTest(t *testing.T) {
	testClient := tc.prepareTestClient(t)
	metricsClient := NewHeapsterMetricsClient(testClient)
	val, err := metricsClient.ResourceConsumption(tc.namespace).Get(tc.targetResource, tc.selector)
	tc.verifyResults(t, val, err)
}

func TestCPU(t *testing.T) {
	tc := testCase{
		replicas:              3,
		desiredValue:          5000,
		targetResource:        api.ResourceCPU,
		reportedMetricsPoints: [][]metricPoint{{{5000, 1}}, {{5000, 1}}, {{5000, 1}}},
	}
	tc.runTest(t)
}

func TestMemory(t *testing.T) {
	tc := testCase{
		replicas:              3,
		desiredValue:          5000,
		targetResource:        api.ResourceMemory,
		reportedMetricsPoints: [][]metricPoint{{{5000, 1}}, {{5000, 2}}, {{5000, 4}}},
	}
	tc.runTest(t)
}

func TestCPUSumEqualZero(t *testing.T) {
	tc := testCase{
		replicas:              3,
		desiredValue:          0,
		targetResource:        api.ResourceCPU,
		reportedMetricsPoints: [][]metricPoint{{{0, 0}}, {{0, 0}}, {{0, 0}}},
	}
	tc.runTest(t)
}

func TestMemorySumEqualZero(t *testing.T) {
	tc := testCase{
		replicas:              3,
		desiredValue:          0,
		targetResource:        api.ResourceMemory,
		reportedMetricsPoints: [][]metricPoint{{{0, 0}}, {{0, 0}}, {{0, 0}}},
	}
	tc.runTest(t)
}

func TestCPUMoreMetrics(t *testing.T) {
	tc := testCase{
		replicas:       5,
		desiredValue:   5000,
		targetResource: api.ResourceCPU,
		reportedMetricsPoints: [][]metricPoint{
			{{0, 3}, {0, 6}, {5, 4}, {9000, 10}},
			{{5000, 2}, {10, 5}, {66, 1}, {0, 10}},
			{{5000, 3}, {80, 5}, {6000, 10}},
			{{5000, 3}, {40, 3}, {0, 9}, {200, 2}, {8000, 10}},
			{{5000, 2}, {20, 2}, {2000, 10}}},
	}
	tc.runTest(t)
}

func TestMemoryMoreMetrics(t *testing.T) {
	tc := testCase{
		replicas:       5,
		desiredValue:   5000,
		targetResource: api.ResourceMemory,
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
		targetResource:        api.ResourceCPU,
		reportedMetricsPoints: [][]metricPoint{{{4000, 4}}, {{9500, 4}}, {{3000, 4}}, {{7000, 4}}, {{3200, 4}}, {{2000, 4}}},
	}
	tc.runTest(t)
}

func TestMemoryResultIsFloat(t *testing.T) {
	tc := testCase{
		replicas:              6,
		desiredValue:          4783,
		targetResource:        api.ResourceMemory,
		reportedMetricsPoints: [][]metricPoint{{{4000, 4}}, {{9500, 4}}, {{3000, 4}}, {{7000, 4}}, {{3200, 4}}, {{2000, 4}}},
	}
	tc.runTest(t)
}

func TestCPUSamplesWithRandomTimestamps(t *testing.T) {
	tc := testCase{
		replicas:       3,
		desiredValue:   3000,
		targetResource: api.ResourceCPU,
		reportedMetricsPoints: [][]metricPoint{
			{{1, 1}, {3000, 3}, {2, 2}},
			{{2, 2}, {1, 1}, {3000, 3}},
			{{3000, 3}, {1, 1}, {2, 2}}},
	}
	tc.runTest(t)
}

func TestMemorySamplesWithRandomTimestamps(t *testing.T) {
	tc := testCase{
		replicas:       3,
		desiredValue:   3000,
		targetResource: api.ResourceMemory,
		reportedMetricsPoints: [][]metricPoint{
			{{1, 1}, {3000, 3}, {2, 2}},
			{{2, 2}, {1, 1}, {3000, 3}},
			{{3000, 3}, {1, 1}, {2, 2}}},
	}
	tc.runTest(t)
}

func TestErrorMetricNotDefined(t *testing.T) {
	tc := testCase{
		replicas:              1,
		desiredError:          fmt.Errorf("heapster metric not defined for "),
		reportedMetricsPoints: [][]metricPoint{{{4000, 4}}},
	}
	tc.runTest(t)
}

func TestCPUMissingMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        api.ResourceCPU,
		desiredError:          fmt.Errorf("metrics obtained for 1/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{{4000, 4}}},
	}
	tc.runTest(t)
}

func TestMemoryMissingMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        api.ResourceMemory,
		desiredError:          fmt.Errorf("metrics obtained for 1/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{{4000, 4}}},
	}
	tc.runTest(t)
}

func TestCPUSuperfluousMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        api.ResourceCPU,
		desiredError:          fmt.Errorf("metrics obtained for 6/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{{1000, 1}}, {{2000, 4}}, {{2000, 1}}, {{4000, 5}}, {{2000, 1}}, {{4000, 4}}},
	}
	tc.runTest(t)
}

func TestMemorySuperfluousMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        api.ResourceMemory,
		desiredError:          fmt.Errorf("metrics obtained for 6/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{{1000, 1}}, {{2000, 4}}, {{2000, 1}}, {{4000, 5}}, {{2000, 1}}, {{4000, 4}}},
	}
	tc.runTest(t)
}

func TestCPUEmptyMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        api.ResourceCPU,
		desiredError:          fmt.Errorf("metrics obtained for 0/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{},
	}
	tc.runTest(t)
}

func TestMemoryEmptyMetrics(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        api.ResourceMemory,
		desiredError:          fmt.Errorf("metrics obtained for 0/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{},
	}
	tc.runTest(t)
}

func TestCPUZeroReplicas(t *testing.T) {
	tc := testCase{
		replicas:              0,
		targetResource:        api.ResourceCPU,
		desiredValue:          0,
		reportedMetricsPoints: [][]metricPoint{},
	}
	tc.runTest(t)
}

func TestMemoryZeroReplicas(t *testing.T) {
	tc := testCase{
		replicas:              0,
		targetResource:        api.ResourceMemory,
		desiredValue:          0,
		reportedMetricsPoints: [][]metricPoint{},
	}
	tc.runTest(t)
}

func TestCPUEmptyMetricsForOnePod(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        api.ResourceCPU,
		desiredError:          fmt.Errorf("metrics obtained for 2/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{}, {{100, 1}}, {{400, 2}, {300, 3}}},
	}
	tc.runTest(t)
}

func TestMemoryEmptyMetricsForOnePod(t *testing.T) {
	tc := testCase{
		replicas:              3,
		targetResource:        api.ResourceMemory,
		desiredError:          fmt.Errorf("metrics obtained for 2/3 of pods"),
		reportedMetricsPoints: [][]metricPoint{{}, {{100, 1}}, {{400, 2}, {300, 3}}},
	}
	tc.runTest(t)
}
