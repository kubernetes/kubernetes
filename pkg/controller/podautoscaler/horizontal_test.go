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

package podautoscaler

import (
	"encoding/json"
	"fmt"
	"io"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/apis/experimental"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
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

type testCase struct {
	minReplicas     int
	maxReplicas     int
	initialReplicas int
	desiredReplicas int
	targetResource  api.ResourceName
	targetLevel     resource.Quantity
	reportedLevels  []uint64
	scaleUpdated    bool
	eventCreated    bool
	verifyEvents    bool
}

func (tc *testCase) prepareTestClient(t *testing.T) *testclient.Fake {
	namespace := "test-namespace"
	hpaName := "test-hpa"
	rcName := "test-rc"
	podNamePrefix := "test-pod"

	tc.scaleUpdated = false
	tc.eventCreated = false

	fakeClient := &testclient.Fake{}
	fakeClient.AddReactor("list", "horizontalpodautoscalers", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
		obj := &experimental.HorizontalPodAutoscalerList{
			Items: []experimental.HorizontalPodAutoscaler{
				{
					ObjectMeta: api.ObjectMeta{
						Name:      hpaName,
						Namespace: namespace,
						SelfLink:  "experimental/v1/namespaces/" + namespace + "/horizontalpodautoscalers/" + hpaName,
					},
					Spec: experimental.HorizontalPodAutoscalerSpec{
						ScaleRef: &experimental.SubresourceReference{
							Kind:        "replicationController",
							Name:        rcName,
							Namespace:   namespace,
							Subresource: "scale",
						},
						MinReplicas: tc.minReplicas,
						MaxReplicas: tc.maxReplicas,
						Target:      experimental.ResourceConsumption{Resource: tc.targetResource, Quantity: tc.targetLevel},
					},
				},
			},
		}
		return true, obj, nil
	})

	fakeClient.AddReactor("get", "replicationController", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
		obj := &experimental.Scale{
			ObjectMeta: api.ObjectMeta{
				Name:      rcName,
				Namespace: namespace,
			},
			Spec: experimental.ScaleSpec{
				Replicas: tc.initialReplicas,
			},
			Status: experimental.ScaleStatus{
				Replicas: tc.initialReplicas,
				Selector: map[string]string{"name": podNamePrefix},
			},
		}
		return true, obj, nil
	})

	fakeClient.AddReactor("list", "pods", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
		obj := &api.PodList{}
		for i := 0; i < tc.initialReplicas; i++ {
			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
			pod := api.Pod{
				Status: api.PodStatus{
					Phase: api.PodRunning,
				},
				ObjectMeta: api.ObjectMeta{
					Name:      podName,
					Namespace: namespace,
					Labels: map[string]string{
						"name": podNamePrefix,
					},
				},
			}
			obj.Items = append(obj.Items, pod)
		}
		return true, obj, nil
	})

	fakeClient.AddProxyReactor("services", func(action testclient.Action) (handled bool, ret client.ResponseWrapper, err error) {
		timestamp := time.Now()
		metrics := heapster.MetricResultList{}
		for _, level := range tc.reportedLevels {
			metric := heapster.MetricResult{
				Metrics:         []heapster.MetricPoint{{timestamp, level}},
				LatestTimestamp: timestamp,
			}
			metrics.Items = append(metrics.Items, metric)
		}
		heapsterRawMemResponse, _ := json.Marshal(&metrics)
		return true, newFakeResponseWrapper(heapsterRawMemResponse), nil
	})

	fakeClient.AddReactor("update", "replicationController", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(testclient.UpdateAction).GetObject().(*experimental.Scale)
		replicas := action.(testclient.UpdateAction).GetObject().(*experimental.Scale).Spec.Replicas
		assert.Equal(t, tc.desiredReplicas, replicas)
		tc.scaleUpdated = true
		return true, obj, nil
	})

	fakeClient.AddReactor("update", "horizontalpodautoscalers", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(testclient.UpdateAction).GetObject().(*experimental.HorizontalPodAutoscaler)
		assert.Equal(t, namespace, obj.Namespace)
		assert.Equal(t, hpaName, obj.Name)
		assert.Equal(t, tc.desiredReplicas, obj.Status.DesiredReplicas)
		return true, obj, nil
	})

	fakeClient.AddReactor("*", "events", func(action testclient.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(testclient.CreateAction).GetObject().(*api.Event)
		if tc.verifyEvents {
			assert.Equal(t, "SuccessfulRescale", obj.Reason)
			assert.Equal(t, fmt.Sprintf("New size: %d", tc.desiredReplicas), obj.Message)
		}
		tc.eventCreated = true
		return true, obj, nil
	})

	return fakeClient
}

func (tc *testCase) verifyResults(t *testing.T) {
	assert.Equal(t, tc.initialReplicas != tc.desiredReplicas, tc.scaleUpdated)
	if tc.verifyEvents {
		assert.Equal(t, tc.initialReplicas != tc.desiredReplicas, tc.eventCreated)
	}
}

func (tc *testCase) runTest(t *testing.T) {
	testClient := tc.prepareTestClient(t)
	hpaController := NewHorizontalController(testClient, metrics.NewHeapsterMetricsClient(testClient))
	err := hpaController.reconcileAutoscalers()
	assert.Equal(t, nil, err)
	if tc.verifyEvents {
		// We need to wait for events to be broadcasted (sleep for longer than record.sleepDuration).
		time.Sleep(12 * time.Second)
	}
	tc.verifyResults(t)
}

func TestCPU(t *testing.T) {
	tc := testCase{
		minReplicas:     1,
		maxReplicas:     5,
		initialReplicas: 1,
		desiredReplicas: 2,
		targetResource:  api.ResourceCPU,
		targetLevel:     resource.MustParse("0.1"),
		reportedLevels:  []uint64{200},
	}
	tc.runTest(t)
}

func TestMemory(t *testing.T) {
	tc := testCase{
		minReplicas:     1,
		maxReplicas:     5,
		initialReplicas: 1,
		desiredReplicas: 2,
		targetResource:  api.ResourceMemory,
		targetLevel:     resource.MustParse("1k"),
		reportedLevels:  []uint64{2000},
	}
	tc.runTest(t)
}

func TestScaleUp(t *testing.T) {
	tc := testCase{
		minReplicas:     2,
		maxReplicas:     6,
		initialReplicas: 3,
		desiredReplicas: 5,
		targetResource:  api.ResourceMemory,
		targetLevel:     resource.MustParse("3k"),
		reportedLevels:  []uint64{3000, 5000, 7000},
	}
	tc.runTest(t)
}

func TestScaleDown(t *testing.T) {
	tc := testCase{
		minReplicas:     2,
		maxReplicas:     6,
		initialReplicas: 5,
		desiredReplicas: 3,
		targetResource:  api.ResourceCPU,
		targetLevel:     resource.MustParse("0.5"),
		reportedLevels:  []uint64{100, 300, 500, 250, 250},
	}
	tc.runTest(t)
}

func TestTolerance(t *testing.T) {
	tc := testCase{
		minReplicas:     1,
		maxReplicas:     5,
		initialReplicas: 3,
		desiredReplicas: 3,
		targetResource:  api.ResourceMemory,
		targetLevel:     resource.MustParse("1k"),
		reportedLevels:  []uint64{1010, 1030, 1020},
	}
	tc.runTest(t)
}

func TestMinReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:     2,
		maxReplicas:     5,
		initialReplicas: 3,
		desiredReplicas: 2,
		targetResource:  api.ResourceMemory,
		targetLevel:     resource.MustParse("1k"),
		reportedLevels:  []uint64{10, 95, 10},
	}
	tc.runTest(t)
}

func TestMaxReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:     2,
		maxReplicas:     5,
		initialReplicas: 3,
		desiredReplicas: 5,
		targetResource:  api.ResourceMemory,
		targetLevel:     resource.MustParse("1k"),
		reportedLevels:  []uint64{8000, 9500, 1000},
	}
	tc.runTest(t)
}

func TestSuperfluousMetrics(t *testing.T) {
	tc := testCase{
		minReplicas:     2,
		maxReplicas:     6,
		initialReplicas: 4,
		desiredReplicas: 4,
		targetResource:  api.ResourceMemory,
		targetLevel:     resource.MustParse("1k"),
		reportedLevels:  []uint64{4000, 9500, 3000, 7000, 3200, 2000},
	}
	tc.runTest(t)
}

func TestMissingMetrics(t *testing.T) {
	tc := testCase{
		minReplicas:     2,
		maxReplicas:     6,
		initialReplicas: 4,
		desiredReplicas: 4,
		targetResource:  api.ResourceMemory,
		targetLevel:     resource.MustParse("1k"),
		reportedLevels:  []uint64{400, 95},
	}
	tc.runTest(t)
}

func TestEmptyMetrics(t *testing.T) {
	tc := testCase{
		minReplicas:     2,
		maxReplicas:     6,
		initialReplicas: 4,
		desiredReplicas: 4,
		targetResource:  api.ResourceMemory,
		targetLevel:     resource.MustParse("1k"),
		reportedLevels:  []uint64{},
	}
	tc.runTest(t)
}

func TestEventCreated(t *testing.T) {
	tc := testCase{
		minReplicas:     1,
		maxReplicas:     5,
		initialReplicas: 1,
		desiredReplicas: 2,
		targetResource:  api.ResourceCPU,
		targetLevel:     resource.MustParse("0.1"),
		reportedLevels:  []uint64{200},
		verifyEvents:    true,
	}
	tc.runTest(t)
}

func TestEventNotCreated(t *testing.T) {
	tc := testCase{
		minReplicas:     1,
		maxReplicas:     5,
		initialReplicas: 2,
		desiredReplicas: 2,
		targetResource:  api.ResourceCPU,
		targetLevel:     resource.MustParse("0.2"),
		reportedLevels:  []uint64{200, 200},
		verifyEvents:    true,
	}
	tc.runTest(t)
}
