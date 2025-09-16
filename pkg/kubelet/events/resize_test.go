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
	"k8s.io/utils/ptr"
)

func TestPodResizeCompletedMsg(t *testing.T) {
	type testResources struct {
		cpuReq, cpuLim, memReq, memLim int64
	}
	type testContainer struct {
		allocated               testResources
		nonSidecarInit, sidecar bool
		isRunning               bool
	}

	tests := []struct {
		name                        string
		containers                  []testContainer
		expectPodResizeCompletedMsg string
	}{{
		name: "simple running container",
		containers: []testContainer{{
			allocated: testResources{100, 100, 100, 100},
			isRunning: true,
		}},
		expectPodResizeCompletedMsg: `Pod resize completed: {"containers":[{"name":"c0","resources":{"limits":{"cpu":"100m","memory":"100"},"requests":{"cpu":"100m","memory":"100"}}}]}`,
	}, {
		name: "several containers",
		containers: []testContainer{{
			allocated: testResources{cpuReq: 100, cpuLim: 200},
			sidecar:   true,
			isRunning: true,
		}, {
			allocated: testResources{memReq: 100, memLim: 200},
			isRunning: true,
		}, {
			allocated: testResources{cpuReq: 200, memReq: 100},
			isRunning: true,
		}},
		expectPodResizeCompletedMsg: `Pod resize completed: {"initContainers":[{"name":"c0","resources":{"limits":{"cpu":"200m"},"requests":{"cpu":"100m"}}}],"containers":[{"name":"c1","resources":{"limits":{"memory":"200"},"requests":{"memory":"100"}}},{"name":"c2","resources":{"requests":{"cpu":"200m","memory":"100"}}}]}`,
	}, {
		name: "best-effort pod",
		containers: []testContainer{{
			allocated: testResources{},
			isRunning: true,
		}},
		expectPodResizeCompletedMsg: `Pod resize completed: {"containers":[{"name":"c0","resources":{}}]}`,
	}}

	mkRequirements := func(r testResources) v1.ResourceRequirements {
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
	mkContainer := func(index int, c testContainer) v1.Container {
		container := v1.Container{
			Name:      fmt.Sprintf("c%d", index),
			Resources: mkRequirements(c.allocated),
		}
		if c.sidecar {
			container.RestartPolicy = ptr.To(v1.ContainerRestartPolicyAlways)
		}
		return container
	}

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
				container := mkContainer(i, c)
				if c.nonSidecarInit || c.sidecar {
					pod.Spec.InitContainers = append(pod.Spec.InitContainers, container)
				} else {
					pod.Spec.Containers = append(pod.Spec.Containers, container)
				}
			}

			// Verify pod resize completed event is emitted
			msg := PodResizeCompletedMsg(pod)
			assert.Equal(t, test.expectPodResizeCompletedMsg, msg)
		})
	}
}
