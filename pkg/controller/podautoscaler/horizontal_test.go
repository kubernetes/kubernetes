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
	"k8s.io/kubernetes/pkg/api/resource"
	_ "k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
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
	// CPU target utilization as a percentage of the requested resources.
	CPUTarget           int
	CPUCurrent          int
	verifyCPUCurrent    bool
	reportedLevels      []uint64
	reportedCPURequests []resource.Quantity
	cmTarget            *extensions.CustomMetricTargetList
	scaleUpdated        bool
	statusUpdated       bool
	eventCreated        bool
	verifyEvents        bool
}

func (tc *testCase) computeCPUCurrent() {
	if len(tc.reportedLevels) != len(tc.reportedCPURequests) || len(tc.reportedLevels) == 0 {
		return
	}
	reported := 0
	for _, r := range tc.reportedLevels {
		reported += int(r)
	}
	requested := 0
	for _, req := range tc.reportedCPURequests {
		requested += int(req.MilliValue())
	}
	tc.CPUCurrent = 100 * reported / requested
}

func (tc *testCase) prepareTestClient(t *testing.T) *fake.Clientset {
	namespace := "test-namespace"
	hpaName := "test-hpa"
	rcName := "test-rc"
	podNamePrefix := "test-pod"

	tc.scaleUpdated = false
	tc.statusUpdated = false
	tc.eventCreated = false
	tc.computeCPUCurrent()

	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("list", "horizontalpodautoscalers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &extensions.HorizontalPodAutoscalerList{
			Items: []extensions.HorizontalPodAutoscaler{
				{
					ObjectMeta: api.ObjectMeta{
						Name:      hpaName,
						Namespace: namespace,
						SelfLink:  "experimental/v1/namespaces/" + namespace + "/horizontalpodautoscalers/" + hpaName,
					},
					Spec: extensions.HorizontalPodAutoscalerSpec{
						ScaleRef: extensions.SubresourceReference{
							Kind:        "replicationController",
							Name:        rcName,
							Subresource: "scale",
						},
						MinReplicas: &tc.minReplicas,
						MaxReplicas: tc.maxReplicas,
					},
					Status: extensions.HorizontalPodAutoscalerStatus{
						CurrentReplicas: tc.initialReplicas,
						DesiredReplicas: tc.initialReplicas,
					},
				},
			},
		}
		if tc.CPUTarget > 0.0 {
			obj.Items[0].Spec.CPUUtilization = &extensions.CPUTargetUtilization{TargetPercentage: tc.CPUTarget}
		}
		if tc.cmTarget != nil {
			b, err := json.Marshal(tc.cmTarget)
			if err != nil {
				t.Fatalf("Failed to marshal cm: %v", err)
			}
			obj.Items[0].Annotations = make(map[string]string)
			obj.Items[0].Annotations[HpaCustomMetricsTargetAnnotationName] = string(b)
		}
		return true, obj, nil
	})

	fakeClient.AddReactor("get", "replicationController", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &extensions.Scale{
			ObjectMeta: api.ObjectMeta{
				Name:      rcName,
				Namespace: namespace,
			},
			Spec: extensions.ScaleSpec{
				Replicas: tc.initialReplicas,
			},
			Status: extensions.ScaleStatus{
				Replicas: tc.initialReplicas,
				Selector: map[string]string{"name": podNamePrefix},
			},
		}
		return true, obj, nil
	})

	fakeClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &api.PodList{}
		for i := 0; i < len(tc.reportedCPURequests); i++ {
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
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU: tc.reportedCPURequests[i],
								},
							},
						},
					},
				},
			}
			obj.Items = append(obj.Items, pod)
		}
		return true, obj, nil
	})

	fakeClient.AddProxyReactor("services", func(action core.Action) (handled bool, ret client.ResponseWrapper, err error) {
		timestamp := time.Now()
		metrics := heapster.MetricResultList{}
		for _, level := range tc.reportedLevels {
			metric := heapster.MetricResult{
				Metrics:         []heapster.MetricPoint{{timestamp, level, nil}},
				LatestTimestamp: timestamp,
			}
			metrics.Items = append(metrics.Items, metric)
		}
		heapsterRawMemResponse, _ := json.Marshal(&metrics)
		return true, newFakeResponseWrapper(heapsterRawMemResponse), nil
	})

	fakeClient.AddReactor("update", "replicationController", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(testclient.UpdateAction).GetObject().(*extensions.Scale)
		replicas := action.(testclient.UpdateAction).GetObject().(*extensions.Scale).Spec.Replicas
		assert.Equal(t, tc.desiredReplicas, replicas)
		tc.scaleUpdated = true
		return true, obj, nil
	})

	fakeClient.AddReactor("update", "horizontalpodautoscalers", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(testclient.UpdateAction).GetObject().(*extensions.HorizontalPodAutoscaler)
		assert.Equal(t, namespace, obj.Namespace)
		assert.Equal(t, hpaName, obj.Name)
		assert.Equal(t, tc.desiredReplicas, obj.Status.DesiredReplicas)
		tc.statusUpdated = true
		if tc.verifyCPUCurrent {
			assert.NotNil(t, obj.Status.CurrentCPUUtilizationPercentage)
			assert.Equal(t, tc.CPUCurrent, *obj.Status.CurrentCPUUtilizationPercentage)
		}
		return true, obj, nil
	})

	fakeClient.AddReactor("*", "events", func(action core.Action) (handled bool, ret runtime.Object, err error) {
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
	assert.True(t, tc.statusUpdated)
	if tc.verifyEvents {
		assert.Equal(t, tc.initialReplicas != tc.desiredReplicas, tc.eventCreated)
	}
}

func (tc *testCase) runTest(t *testing.T) {
	testClient := tc.prepareTestClient(t)
	metricsClient := metrics.NewHeapsterMetricsClient(testClient, metrics.DefaultHeapsterNamespace, metrics.DefaultHeapsterScheme, metrics.DefaultHeapsterService, metrics.DefaultHeapsterPort)
	hpaController := NewHorizontalController(testClient.Core(), testClient.Extensions(), testClient.Extensions(), metricsClient)
	err := hpaController.reconcileAutoscalers()
	assert.Equal(t, nil, err)
	if tc.verifyEvents {
		// We need to wait for events to be broadcasted (sleep for longer than record.sleepDuration).
		time.Sleep(12 * time.Second)
	}
	tc.verifyResults(t)
}

func TestScaleUp(t *testing.T) {
	tc := testCase{
		minReplicas:         2,
		maxReplicas:         6,
		initialReplicas:     3,
		desiredReplicas:     5,
		CPUTarget:           30,
		verifyCPUCurrent:    true,
		reportedLevels:      []uint64{300, 500, 700},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestScaleUpCM(t *testing.T) {
	tc := testCase{
		minReplicas:     2,
		maxReplicas:     6,
		initialReplicas: 3,
		desiredReplicas: 4,
		CPUTarget:       0,
		cmTarget: &extensions.CustomMetricTargetList{
			Items: []extensions.CustomMetricTarget{{
				Name:        "qps",
				TargetValue: resource.MustParse("15.0"),
			}},
		},
		reportedLevels:      []uint64{20, 10, 30},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestScaleDown(t *testing.T) {
	tc := testCase{
		minReplicas:         2,
		maxReplicas:         6,
		initialReplicas:     5,
		desiredReplicas:     3,
		CPUTarget:           50,
		verifyCPUCurrent:    true,
		reportedLevels:      []uint64{100, 300, 500, 250, 250},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestScaleDownCM(t *testing.T) {
	tc := testCase{
		minReplicas:     2,
		maxReplicas:     6,
		initialReplicas: 5,
		desiredReplicas: 3,
		CPUTarget:       0,
		cmTarget: &extensions.CustomMetricTargetList{
			Items: []extensions.CustomMetricTarget{{
				Name:        "qps",
				TargetValue: resource.MustParse("20"),
			}}},
		reportedLevels:      []uint64{12, 12, 12, 12, 12},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestTolerance(t *testing.T) {
	tc := testCase{
		minReplicas:         1,
		maxReplicas:         5,
		initialReplicas:     3,
		desiredReplicas:     3,
		CPUTarget:           100,
		reportedLevels:      []uint64{1010, 1030, 1020},
		reportedCPURequests: []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
	}
	tc.runTest(t)
}

func TestToleranceCM(t *testing.T) {
	tc := testCase{
		minReplicas:     1,
		maxReplicas:     5,
		initialReplicas: 3,
		desiredReplicas: 3,
		cmTarget: &extensions.CustomMetricTargetList{
			Items: []extensions.CustomMetricTarget{{
				Name:        "qps",
				TargetValue: resource.MustParse("20"),
			}}},
		reportedLevels:      []uint64{20, 21, 21},
		reportedCPURequests: []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
	}
	tc.runTest(t)
}

func TestMinReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:         2,
		maxReplicas:         5,
		initialReplicas:     3,
		desiredReplicas:     2,
		CPUTarget:           90,
		reportedLevels:      []uint64{10, 95, 10},
		reportedCPURequests: []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
	}
	tc.runTest(t)
}

func TestZeroReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:         3,
		maxReplicas:         5,
		initialReplicas:     0,
		desiredReplicas:     3,
		CPUTarget:           90,
		reportedLevels:      []uint64{},
		reportedCPURequests: []resource.Quantity{},
	}
	tc.runTest(t)
}

func TestTooFewReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:         3,
		maxReplicas:         5,
		initialReplicas:     2,
		desiredReplicas:     3,
		CPUTarget:           90,
		reportedLevels:      []uint64{},
		reportedCPURequests: []resource.Quantity{},
	}
	tc.runTest(t)
}

func TestTooManyReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:         3,
		maxReplicas:         5,
		initialReplicas:     10,
		desiredReplicas:     5,
		CPUTarget:           90,
		reportedLevels:      []uint64{},
		reportedCPURequests: []resource.Quantity{},
	}
	tc.runTest(t)
}

func TestMaxReplicas(t *testing.T) {
	tc := testCase{
		minReplicas:         2,
		maxReplicas:         5,
		initialReplicas:     3,
		desiredReplicas:     5,
		CPUTarget:           90,
		reportedLevels:      []uint64{8000, 9500, 1000},
		reportedCPURequests: []resource.Quantity{resource.MustParse("0.9"), resource.MustParse("1.0"), resource.MustParse("1.1")},
	}
	tc.runTest(t)
}

func TestSuperfluousMetrics(t *testing.T) {
	tc := testCase{
		minReplicas:         2,
		maxReplicas:         6,
		initialReplicas:     4,
		desiredReplicas:     4,
		CPUTarget:           100,
		reportedLevels:      []uint64{4000, 9500, 3000, 7000, 3200, 2000},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestMissingMetrics(t *testing.T) {
	tc := testCase{
		minReplicas:         2,
		maxReplicas:         6,
		initialReplicas:     4,
		desiredReplicas:     4,
		CPUTarget:           100,
		reportedLevels:      []uint64{400, 95},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestEmptyMetrics(t *testing.T) {
	tc := testCase{
		minReplicas:         2,
		maxReplicas:         6,
		initialReplicas:     4,
		desiredReplicas:     4,
		CPUTarget:           100,
		reportedLevels:      []uint64{},
		reportedCPURequests: []resource.Quantity{resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0"), resource.MustParse("1.0")},
	}
	tc.runTest(t)
}

func TestEmptyCPURequest(t *testing.T) {
	tc := testCase{
		minReplicas:     1,
		maxReplicas:     5,
		initialReplicas: 1,
		desiredReplicas: 1,
		CPUTarget:       100,
		reportedLevels:  []uint64{200},
	}
	tc.runTest(t)
}

func TestEventCreated(t *testing.T) {
	tc := testCase{
		minReplicas:         1,
		maxReplicas:         5,
		initialReplicas:     1,
		desiredReplicas:     2,
		CPUTarget:           50,
		reportedLevels:      []uint64{200},
		reportedCPURequests: []resource.Quantity{resource.MustParse("0.2")},
		verifyEvents:        true,
	}
	tc.runTest(t)
}

func TestEventNotCreated(t *testing.T) {
	tc := testCase{
		minReplicas:         1,
		maxReplicas:         5,
		initialReplicas:     2,
		desiredReplicas:     2,
		CPUTarget:           50,
		reportedLevels:      []uint64{200, 200},
		reportedCPURequests: []resource.Quantity{resource.MustParse("0.4"), resource.MustParse("0.4")},
		verifyEvents:        true,
	}
	tc.runTest(t)
}

// TODO: add more tests
