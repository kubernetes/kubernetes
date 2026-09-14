/*
Copyright 2025 The Kubernetes Authors.

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

package events

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func TestPodResizeCompletedMsg(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	tests := []struct {
		name               string
		containers         []testContainer
		observedGeneration int64
		expected           string
	}{{
		name: "simple running container",
		containers: []testContainer{{
			resources: testResources{100, 100, 100, 100},
		}},
		observedGeneration: 1,
		expected:           `Pod resize completed: {"containers":[{"name":"c0","resources":{"limits":{"cpu":"100m","memory":"100"},"requests":{"cpu":"100m","memory":"100"}}}],"generation":1}`,
	}, {
		name: "several containers",
		containers: []testContainer{{
			resources: testResources{cpuReq: 100, cpuLim: 200},
			sidecar:   true,
		}, {
			resources: testResources{memReq: 100, memLim: 200},
		}, {
			resources: testResources{cpuReq: 200, memReq: 100},
		}},
		observedGeneration: 2,
		expected:           `Pod resize completed: {"initContainers":[{"name":"c0","resources":{"limits":{"cpu":"200m"},"requests":{"cpu":"100m"}}}],"containers":[{"name":"c1","resources":{"limits":{"memory":"200"},"requests":{"memory":"100"}}},{"name":"c2","resources":{"requests":{"cpu":"200m","memory":"100"}}}],"generation":2}`,
	}, {
		name: "best-effort pod",
		containers: []testContainer{{
			resources: testResources{},
		}},
		observedGeneration: 3,
		expected:           `Pod resize completed: {"containers":[{"name":"c0","resources":{}}],"generation":3}`,
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
					UID:  "12345",
				},
			}

			for i, c := range test.containers {
				// Add the container to the pod
				container := mkContainer(i, c.resources, c.sidecar)
				if c.nonSidecarInit || c.sidecar {
					pod.Spec.InitContainers = append(pod.Spec.InitContainers, container)
				} else {
					pod.Spec.Containers = append(pod.Spec.Containers, container)
				}
			}

			msg := PodResizeCompletedMsg(logger, pod, test.observedGeneration)
			assert.Equal(t, test.expected, msg)
		})
	}
}

func TestPodResizeStartedMsg(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	tests := []struct {
		name               string
		allocated          []testContainer
		actual             []testContainer
		observedGeneration int64
		expected           string
	}{
		{
			name:               "simple running container",
			allocated:          []testContainer{{resources: testResources{100, 100, 100, 100}}},
			actual:             []testContainer{{resources: testResources{50, 50, 50, 50}}},
			observedGeneration: 1,
			expected:           `Pod resize started: {"containers":[{"name":"c0","resources":{"limits":{"cpu":"100m","memory":"100"},"requests":{"cpu":"100m","memory":"100"}}}],"generation":1}`,
		}, {
			name: "several containers",
			allocated: []testContainer{
				{resources: testResources{cpuReq: 100, cpuLim: 200}, sidecar: true},
				{resources: testResources{memReq: 100, memLim: 200}},
				{resources: testResources{cpuReq: 200, memReq: 100}},
			},
			actual: []testContainer{
				{resources: testResources{cpuReq: 50, cpuLim: 100}, sidecar: true},
				{resources: testResources{cpuReq: 50, cpuLim: 100}},
				{resources: testResources{cpuReq: 50, cpuLim: 100}},
			},
			observedGeneration: 2,
			expected:           `Pod resize started: {"initContainers":[{"name":"c0","resources":{"limits":{"cpu":"200m"},"requests":{"cpu":"100m"}}}],"containers":[{"name":"c1","resources":{"limits":{"memory":"200"},"requests":{"memory":"100"}}},{"name":"c2","resources":{"requests":{"cpu":"200m","memory":"100"}}}],"generation":2}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			allocatedPod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
					UID:  "12345",
				},
			}

			for i, c := range test.actual {
				actual := mkContainer(i, c.resources, c.sidecar)
				if c.nonSidecarInit || c.sidecar {
					allocatedPod.Status.InitContainerStatuses = append(allocatedPod.Status.InitContainerStatuses, v1.ContainerStatus{
						Name:      actual.Name,
						Resources: &actual.Resources,
					})
				} else {
					allocatedPod.Status.ContainerStatuses = append(allocatedPod.Status.ContainerStatuses, v1.ContainerStatus{
						Name:      actual.Name,
						Resources: &actual.Resources,
					})
				}
			}

			for i, c := range test.allocated {
				allocated := mkContainer(i, c.resources, c.sidecar)
				if c.nonSidecarInit || c.sidecar {
					allocatedPod.Spec.InitContainers = append(allocatedPod.Spec.InitContainers, allocated)
				} else {
					allocatedPod.Spec.Containers = append(allocatedPod.Spec.Containers, allocated)
				}
			}

			msg := PodResizeStartedMsg(logger, allocatedPod, test.observedGeneration)
			assert.Equal(t, test.expected, msg)
		})
	}
}

func TestPodResizeErrorMsg(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	tests := []struct {
		name               string
		allocated          []testContainer
		actual             []testContainer
		observedGeneration int64
		errMsg             string
		expected           string
	}{
		{
			name:               "simple running container",
			allocated:          []testContainer{{resources: testResources{100, 100, 100, 100}}},
			actual:             []testContainer{{resources: testResources{50, 50, 50, 50}}},
			observedGeneration: 1,
			errMsg:             "some error occurred",
			expected:           `Pod resize error: {"containers":[{"name":"c0","resources":{"limits":{"cpu":"100m","memory":"100"},"requests":{"cpu":"100m","memory":"100"}}}],"generation":1,"error":"some error occurred"}`,
		}, {
			name: "several containers",
			allocated: []testContainer{
				{resources: testResources{cpuReq: 100, cpuLim: 200}, sidecar: true},
				{resources: testResources{memReq: 100, memLim: 200}},
				{resources: testResources{cpuReq: 200, memReq: 100}},
			},
			actual: []testContainer{
				{resources: testResources{cpuReq: 50, cpuLim: 100}, sidecar: true},
				{resources: testResources{cpuReq: 50, cpuLim: 100}},
				{resources: testResources{cpuReq: 50, cpuLim: 100}},
			},
			observedGeneration: 2,
			errMsg:             "some error occurred",
			expected:           `Pod resize error: {"initContainers":[{"name":"c0","resources":{"limits":{"cpu":"200m"},"requests":{"cpu":"100m"}}}],"containers":[{"name":"c1","resources":{"limits":{"memory":"200"},"requests":{"memory":"100"}}},{"name":"c2","resources":{"requests":{"cpu":"200m","memory":"100"}}}],"generation":2,"error":"some error occurred"}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			allocatedPod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
					UID:  "12345",
				},
			}

			for i, c := range test.actual {
				actual := mkContainer(i, c.resources, c.sidecar)
				if c.nonSidecarInit || c.sidecar {
					allocatedPod.Status.InitContainerStatuses = append(allocatedPod.Status.InitContainerStatuses, v1.ContainerStatus{
						Name:      actual.Name,
						Resources: &actual.Resources,
					})
				} else {
					allocatedPod.Status.ContainerStatuses = append(allocatedPod.Status.ContainerStatuses, v1.ContainerStatus{
						Name:      actual.Name,
						Resources: &actual.Resources,
					})
				}
			}

			for i, c := range test.allocated {
				allocated := mkContainer(i, c.resources, c.sidecar)
				if c.nonSidecarInit || c.sidecar {
					allocatedPod.Spec.InitContainers = append(allocatedPod.Spec.InitContainers, allocated)
				} else {
					allocatedPod.Spec.Containers = append(allocatedPod.Spec.Containers, allocated)
				}
			}

			msg := PodResizeErrorMsg(logger, allocatedPod, test.observedGeneration, test.errMsg)
			assert.Equal(t, test.expected, msg)
		})
	}
}

func TestPodResizePendingMsg(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	tests := []struct {
		name               string
		desired            []testContainer
		allocated          []testContainer
		reason             string
		message            string
		observedGeneration int64
		expected           string
	}{
		{
			name:               "simple running container",
			desired:            []testContainer{{resources: testResources{100, 100, 100, 100}}},
			allocated:          []testContainer{{resources: testResources{50, 50, 50, 50}}},
			observedGeneration: 1,
			reason:             "Deferred",
			message:            "Node didn't have enough resource: memory",
			expected:           `Pod resize Deferred: {"containers":[{"name":"c0","resources":{"limits":{"cpu":"100m","memory":"100"},"requests":{"cpu":"100m","memory":"100"}}}],"generation":1,"error":"Node didn't have enough resource: memory"}`,
		}, {
			name: "several containers",
			desired: []testContainer{
				{resources: testResources{cpuReq: 100, cpuLim: 200}, sidecar: true},
				{resources: testResources{memReq: 100, memLim: 200}},
				{resources: testResources{cpuReq: 200, memReq: 100}},
			},
			allocated: []testContainer{
				{resources: testResources{cpuReq: 50, cpuLim: 100}, sidecar: true},
				{resources: testResources{cpuReq: 50, cpuLim: 100}},
				{resources: testResources{cpuReq: 50, cpuLim: 100}},
			},
			observedGeneration: 2,
			reason:             "Infeasible",
			message:            "In-place resize of containers with swap is not supported",
			expected:           `Pod resize Infeasible: {"initContainers":[{"name":"c0","resources":{"limits":{"cpu":"200m"},"requests":{"cpu":"100m"}}}],"containers":[{"name":"c1","resources":{"limits":{"memory":"200"},"requests":{"memory":"100"}}},{"name":"c2","resources":{"requests":{"cpu":"200m","memory":"100"}}}],"generation":2,"error":"In-place resize of containers with swap is not supported"}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			allocatedPod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
					UID:  "12345",
				},
			}
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
					UID:  "12345",
				},
			}

			for i, c := range test.allocated {
				allocated := mkContainer(i, c.resources, c.sidecar)
				if c.nonSidecarInit || c.sidecar {
					allocatedPod.Spec.InitContainers = append(allocatedPod.Spec.InitContainers, allocated)
				} else {
					allocatedPod.Spec.Containers = append(allocatedPod.Spec.Containers, allocated)
				}
			}

			for i, c := range test.desired {
				desired := mkContainer(i, c.resources, c.sidecar)
				if c.nonSidecarInit || c.sidecar {
					pod.Spec.InitContainers = append(pod.Spec.InitContainers, desired)
				} else {
					pod.Spec.Containers = append(pod.Spec.Containers, desired)
				}
			}

			msg := PodResizePendingMsg(logger, pod, test.reason, test.message, test.observedGeneration)
			assert.Equal(t, test.expected, msg)
		})
	}
}

type testResources struct {
	cpuReq, cpuLim, memReq, memLim int64
}
type testContainer struct {
	resources               testResources
	nonSidecarInit, sidecar bool
}

func mkRequirements(r testResources) v1.ResourceRequirements {
	res := v1.ResourceRequirements{
		Requests: v1.ResourceList{},
		Limits:   v1.ResourceList{},
	}
	if r.cpuReq != 0 {
		res.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(r.cpuReq, resource.DecimalSI)
	}
	if r.cpuLim != 0 {
		res.Limits[v1.ResourceCPU] = *resource.NewMilliQuantity(r.cpuLim, resource.DecimalSI)
	}
	if r.memReq != 0 {
		res.Requests[v1.ResourceMemory] = *resource.NewQuantity(r.memReq, resource.DecimalSI)
	}
	if r.memLim != 0 {
		res.Limits[v1.ResourceMemory] = *resource.NewQuantity(r.memLim, resource.DecimalSI)
	}
	return res
}

func mkContainer(index int, resources testResources, sidecar bool) v1.Container {
	container := v1.Container{
		Name:      fmt.Sprintf("c%d", index),
		Resources: mkRequirements(resources),
	}
	if sidecar {
		container.RestartPolicy = ptr.To(v1.ContainerRestartPolicyAlways)
	}
	return container
}
