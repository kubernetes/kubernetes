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

package allocation

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/status"
	statustest "k8s.io/kubernetes/pkg/kubelet/status/testing"
	kubeletutil "k8s.io/kubernetes/pkg/kubelet/util"
	_ "k8s.io/kubernetes/pkg/volume/hostpath"
	"k8s.io/utils/ptr"
)

func TestUpdatePodFromAllocation(t *testing.T) {
	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345",
			Name:      "test",
			Namespace: "default",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(400, resource.DecimalSI),
						},
					},
				},
				{
					Name: "c2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(600, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(700, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(800, resource.DecimalSI),
						},
					},
				},
			},
			InitContainers: []v1.Container{
				{
					Name: "c1-restartable-init",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(300, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
						},
					},
					RestartPolicy: &containerRestartPolicyAlways,
				},
				{
					Name: "c1-init",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(600, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(700, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(800, resource.DecimalSI),
						},
					},
				},
			},
		},
	}

	resizedPod := pod.DeepCopy()
	resizedPod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(200, resource.DecimalSI)
	resizedPod.Spec.InitContainers[0].Resources.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(300, resource.DecimalSI)

	tests := []struct {
		name         string
		pod          *v1.Pod
		allocs       state.PodResourceInfoMap
		expectPod    *v1.Pod
		expectUpdate bool
	}{{
		name: "steady state",
		pod:  pod,
		allocs: state.PodResourceInfoMap{
			pod.UID: state.PodResourceInfo{
				ContainerResources: map[string]v1.ResourceRequirements{
					"c1":                  *pod.Spec.Containers[0].Resources.DeepCopy(),
					"c2":                  *pod.Spec.Containers[1].Resources.DeepCopy(),
					"c1-restartable-init": *pod.Spec.InitContainers[0].Resources.DeepCopy(),
					"c1-init":             *pod.Spec.InitContainers[1].Resources.DeepCopy(),
				},
			},
		},
		expectUpdate: false,
	}, {
		name:         "no allocations",
		pod:          pod,
		allocs:       state.PodResourceInfoMap{},
		expectUpdate: false,
	}, {
		name: "missing container allocation",
		pod:  pod,
		allocs: state.PodResourceInfoMap{
			pod.UID: state.PodResourceInfo{
				ContainerResources: map[string]v1.ResourceRequirements{
					"c2": *pod.Spec.Containers[1].Resources.DeepCopy(),
				},
			},
		},
		expectUpdate: false,
	}, {
		name: "resized container",
		pod:  pod,
		allocs: state.PodResourceInfoMap{
			pod.UID: state.PodResourceInfo{
				ContainerResources: map[string]v1.ResourceRequirements{
					"c1":                  *resizedPod.Spec.Containers[0].Resources.DeepCopy(),
					"c2":                  *resizedPod.Spec.Containers[1].Resources.DeepCopy(),
					"c1-restartable-init": *resizedPod.Spec.InitContainers[0].Resources.DeepCopy(),
					"c1-init":             *resizedPod.Spec.InitContainers[1].Resources.DeepCopy(),
				},
			},
		},
		expectUpdate: true,
		expectPod:    resizedPod,
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pod := test.pod.DeepCopy()
			allocatedPod, updated := updatePodFromAllocation(pod, test.allocs)

			if test.expectUpdate {
				assert.True(t, updated, "updated")
				assert.Equal(t, test.expectPod, allocatedPod)
				assert.NotEqual(t, pod, allocatedPod)
			} else {
				assert.False(t, updated, "updated")
				assert.Same(t, pod, allocatedPod)
			}
		})
	}
}

func TestIsPodResizeInProgress(t *testing.T) {
	type testResources struct {
		cpuReq, cpuLim, memReq, memLim int64
	}
	type testContainer struct {
		allocated               testResources
		actuated                *testResources
		nonSidecarInit, sidecar bool
		isRunning               bool
		unstarted               bool // Whether the container is missing from the pod status
	}

	tests := []struct {
		name            string
		containers      []testContainer
		expectHasResize bool
	}{{
		name: "simple running container",
		containers: []testContainer{{
			allocated: testResources{100, 100, 100, 100},
			actuated:  &testResources{100, 100, 100, 100},
			isRunning: true,
		}},
		expectHasResize: false,
	}, {
		name: "simple unstarted container",
		containers: []testContainer{{
			allocated: testResources{100, 100, 100, 100},
			unstarted: true,
		}},
		expectHasResize: false,
	}, {
		name: "simple resized container/cpu req",
		containers: []testContainer{{
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{150, 200, 100, 200},
			isRunning: true,
		}},
		expectHasResize: true,
	}, {
		name: "simple resized container/cpu limit",
		containers: []testContainer{{
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{100, 300, 100, 200},
			isRunning: true,
		}},
		expectHasResize: true,
	}, {
		name: "simple resized container/mem req",
		containers: []testContainer{{
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{100, 200, 150, 200},
			isRunning: true,
		}},
		expectHasResize: true,
	}, {
		name: "simple resized container/cpu+mem req",
		containers: []testContainer{{
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{150, 200, 150, 200},
			isRunning: true,
		}},
		expectHasResize: true,
	}, {
		name: "simple resized container/mem limit",
		containers: []testContainer{{
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{100, 200, 100, 300},
			isRunning: true,
		}},
		expectHasResize: true,
	}, {
		name: "terminated resized container",
		containers: []testContainer{{
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{200, 200, 100, 200},
			isRunning: false,
		}},
		expectHasResize: false,
	}, {
		name: "non-sidecar init container",
		containers: []testContainer{{
			allocated:      testResources{100, 200, 100, 200},
			nonSidecarInit: true,
			isRunning:      true,
		}, {
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{100, 200, 100, 200},
			isRunning: true,
		}},
		expectHasResize: false,
	}, {
		name: "non-resized sidecar",
		containers: []testContainer{{
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{100, 200, 100, 200},
			sidecar:   true,
			isRunning: true,
		}, {
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{100, 200, 100, 200},
			isRunning: true,
		}},
		expectHasResize: false,
	}, {
		name: "resized sidecar",
		containers: []testContainer{{
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{200, 200, 100, 200},
			sidecar:   true,
			isRunning: true,
		}, {
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{100, 200, 100, 200},
			isRunning: true,
		}},
		expectHasResize: true,
	}, {
		name: "several containers and a resize",
		containers: []testContainer{{
			allocated:      testResources{100, 200, 100, 200},
			nonSidecarInit: true,
			isRunning:      true,
		}, {
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{100, 200, 100, 200},
			isRunning: true,
		}, {
			allocated: testResources{100, 200, 100, 200},
			unstarted: true,
		}, {
			allocated: testResources{100, 200, 100, 200},
			actuated:  &testResources{200, 200, 100, 200}, // Resized
			isRunning: true,
		}},
		expectHasResize: true,
	}, {
		name: "best-effort pod",
		containers: []testContainer{{
			allocated: testResources{},
			actuated:  &testResources{},
			isRunning: true,
		}},
		expectHasResize: false,
	}, {
		name: "burstable pod/not resizing",
		containers: []testContainer{{
			allocated: testResources{cpuReq: 100},
			actuated:  &testResources{cpuReq: 100},
			isRunning: true,
		}},
		expectHasResize: false,
	}, {
		name: "burstable pod/resized",
		containers: []testContainer{{
			allocated: testResources{cpuReq: 100},
			actuated:  &testResources{cpuReq: 500},
			isRunning: true,
		}},
		expectHasResize: true,
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

	statusManager := status.NewManager(&fake.Clientset{}, kubepod.NewBasicPodManager(), &statustest.FakePodDeletionSafetyProvider{}, kubeletutil.NewPodStartupLatencyTracker())
	containerManager := cm.NewFakeContainerManager()
	am := NewInMemoryManager(containerManager, statusManager)

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
					UID:  "12345",
				},
			}
			t.Cleanup(func() { am.RemovePod(pod.UID) })
			podStatus := &kubecontainer.PodStatus{
				ID:   pod.UID,
				Name: pod.Name,
			}
			for i, c := range test.containers {
				// Add the container to the pod
				container := mkContainer(i, c)
				if c.nonSidecarInit || c.sidecar {
					pod.Spec.InitContainers = append(pod.Spec.InitContainers, container)
				} else {
					pod.Spec.Containers = append(pod.Spec.Containers, container)
				}

				// Add the container to the pod status, if it's started.
				if !test.containers[i].unstarted {
					cs := kubecontainer.Status{
						Name: container.Name,
					}
					if test.containers[i].isRunning {
						cs.State = kubecontainer.ContainerStateRunning
					} else {
						cs.State = kubecontainer.ContainerStateExited
					}
					podStatus.ContainerStatuses = append(podStatus.ContainerStatuses, &cs)
				}

				// Register the actuated container (if needed)
				if c.actuated != nil {
					actuatedContainer := container.DeepCopy()
					actuatedContainer.Resources = mkRequirements(*c.actuated)
					require.NoError(t, am.SetActuatedResources(pod, actuatedContainer))

					fetched, found := am.GetActuatedResources(pod.UID, container.Name)
					require.True(t, found)
					assert.Equal(t, actuatedContainer.Resources, fetched)
				} else {
					_, found := am.GetActuatedResources(pod.UID, container.Name)
					require.False(t, found)
				}
			}
			require.NoError(t, am.SetAllocatedResources(pod))

			hasResizedResources := am.(*manager).isPodResizeInProgress(pod, podStatus)
			require.Equal(t, test.expectHasResize, hasResizedResources, "hasResizedResources")
		})
	}
}
