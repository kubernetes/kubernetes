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
	goruntime "runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/status"
	statustest "k8s.io/kubernetes/pkg/kubelet/status/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
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

func TestHandlePodResourcesResize(t *testing.T) {
	if goruntime.GOOS == "windows" {
		t.Skip("InPlacePodVerticalScaling is not currently supported for Windows")
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways

	cpu2m := resource.MustParse("2m")
	cpu500m := resource.MustParse("500m")
	cpu1000m := resource.MustParse("1")
	cpu1500m := resource.MustParse("1500m")
	cpu2500m := resource.MustParse("2500m")
	cpu5000m := resource.MustParse("5000m")
	mem500M := resource.MustParse("500Mi")
	mem1000M := resource.MustParse("1Gi")
	mem1500M := resource.MustParse("1500Mi")
	mem2500M := resource.MustParse("2500Mi")
	mem4500M := resource.MustParse("4500Mi")

	testPod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "1111",
			Name:      "pod1",
			Namespace: "ns1",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "c1",
					Image: "i1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name:               "c1",
					AllocatedResources: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
					Resources:          &v1.ResourceRequirements{},
				},
			},
		},
	}
	testPod2 := testPod1.DeepCopy()
	testPod2.UID = "2222"
	testPod2.Name = "pod2"
	testPod2.Namespace = "ns2"
	testPod2.Spec = v1.PodSpec{
		InitContainers: []v1.Container{
			{
				Name:  "c1-init",
				Image: "i1",
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
				},
				RestartPolicy: &containerRestartPolicyAlways,
			},
		},
	}
	testPod2.Status = v1.PodStatus{
		Phase: v1.PodRunning,
		InitContainerStatuses: []v1.ContainerStatus{
			{
				Name:               "c1-init",
				AllocatedResources: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
				Resources:          &v1.ResourceRequirements{},
			},
		},
	}
	testPod3 := testPod1.DeepCopy()
	testPod3.UID = "3333"
	testPod3.Name = "pod3"
	testPod3.Namespace = "ns2"

	tests := []struct {
		name                  string
		originalRequests      v1.ResourceList
		newRequests           v1.ResourceList
		originalLimits        v1.ResourceList
		newLimits             v1.ResourceList
		newResourcesAllocated bool // Whether the new requests have already been allocated (but not actuated)
		expectedAllocatedReqs v1.ResourceList
		expectedAllocatedLims v1.ResourceList
		expectedResize        []*v1.PodCondition
		annotations           map[string]string
	}{
		{
			name:                  "Request CPU and memory decrease - expect InProgress",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},

			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
		{
			name:                  "Request CPU increase, memory decrease - expect InProgress",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem500M},

			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
		{
			name:                  "Request CPU decrease, memory increase - expect InProgress",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem1500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem1500M},

			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
		{
			name:                  "Request CPU and memory increase beyond current capacity - expect Deferred",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu2500m, v1.ResourceMemory: mem2500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},

			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Deferred",
					Message: "",
				},
			},
		},
		{
			name:                  "Request CPU decrease and memory increase beyond current capacity - expect Deferred",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem2500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},

			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Deferred",
					Message: "Node didn't have enough resource: memory",
				},
			},
		},
		{
			name:                  "Request memory increase beyond node capacity - expect Infeasible",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem4500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},

			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Infeasible",
					Message: "Node didn't have enough capacity: memory, requested: 4718592000, capacity: 4294967296",
				},
			},
		},
		{
			name:                  "Request CPU increase beyond node capacity - expect Infeasible",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu5000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},

			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Infeasible",
					Message: "Node didn't have enough capacity: cpu, requested: 5000, capacity: 4000",
				},
			},
		},
		{
			name:                  "CPU increase in progress - expect InProgress",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1000M},
			newResourcesAllocated: true,
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1000M},

			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
		{
			name:                  "No resize",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedResize:        nil,
		},
		{
			name:                  "static pod, expect Infeasible",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			annotations:           map[string]string{kubetypes.ConfigSourceAnnotationKey: kubetypes.FileSource},

			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Infeasible",
					Message: "In-place resize of static-pods is not supported",
				},
			},
		},
		{
			name:                  "Increase CPU from min shares",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu2m},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu1000m},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1000m},

			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
		{
			name:                  "Decrease CPU to min shares",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu2m},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu2m},

			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
		{
			name:                  "Increase CPU from min limit",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: resource.MustParse("10m")},
			originalLimits:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("10m")},
			newRequests:           v1.ResourceList{v1.ResourceCPU: resource.MustParse("10m")}, // Unchanged
			newLimits:             v1.ResourceList{v1.ResourceCPU: resource.MustParse("20m")},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: resource.MustParse("10m")},
			expectedAllocatedLims: v1.ResourceList{v1.ResourceCPU: resource.MustParse("20m")},

			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
		{
			name:                  "Decrease CPU to min limit",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: resource.MustParse("10m")},
			originalLimits:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("20m")},
			newRequests:           v1.ResourceList{v1.ResourceCPU: resource.MustParse("10m")}, // Unchanged
			newLimits:             v1.ResourceList{v1.ResourceCPU: resource.MustParse("10m")},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: resource.MustParse("10m")},
			expectedAllocatedLims: v1.ResourceList{v1.ResourceCPU: resource.MustParse("10m")},

			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
	}

	for _, tt := range tests {
		for _, isSidecarContainer := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/sidecar=%t", tt.name, isSidecarContainer), func(t *testing.T) {
				var originalPod *v1.Pod
				var originalCtr *v1.Container
				if isSidecarContainer {
					originalPod = testPod2.DeepCopy()
					originalCtr = &originalPod.Spec.InitContainers[0]
				} else {
					originalPod = testPod1.DeepCopy()
					originalCtr = &originalPod.Spec.Containers[0]
				}
				originalPod.Annotations = tt.annotations
				originalCtr.Resources.Requests = tt.originalRequests
				originalCtr.Resources.Limits = tt.originalLimits

				newPod := originalPod.DeepCopy()
				if isSidecarContainer {
					newPod.Spec.InitContainers[0].Resources.Requests = tt.newRequests
					newPod.Spec.InitContainers[0].Resources.Limits = tt.newLimits
				} else {
					newPod.Spec.Containers[0].Resources.Requests = tt.newRequests
					newPod.Spec.Containers[0].Resources.Limits = tt.newLimits
				}

				podStatus := &kubecontainer.PodStatus{
					ID:        originalPod.UID,
					Name:      originalPod.Name,
					Namespace: originalPod.Namespace,
				}

				setContainerStatus := func(podStatus *kubecontainer.PodStatus, c *v1.Container, idx int) {
					podStatus.ContainerStatuses[idx] = &kubecontainer.Status{
						Name:  c.Name,
						State: kubecontainer.ContainerStateRunning,
						Resources: &kubecontainer.ContainerResources{
							CPURequest:  c.Resources.Requests.Cpu(),
							CPULimit:    c.Resources.Limits.Cpu(),
							MemoryLimit: c.Resources.Limits.Memory(),
						},
					}
				}

				podStatus.ContainerStatuses = make([]*kubecontainer.Status, len(originalPod.Spec.Containers)+len(originalPod.Spec.InitContainers))
				for i, c := range originalPod.Spec.InitContainers {
					setContainerStatus(podStatus, &c, i)
				}
				for i, c := range originalPod.Spec.Containers {
					setContainerStatus(podStatus, &c, i+len(originalPod.Spec.InitContainers))
				}
				allocationManager := makeAllocationManager(t, &containertest.FakeRuntime{PodStatus: *podStatus}, []*v1.Pod{testPod1, testPod2, testPod3})

				if !tt.newResourcesAllocated {
					require.NoError(t, allocationManager.SetAllocatedResources(originalPod))
				} else {
					require.NoError(t, allocationManager.SetAllocatedResources(newPod))
				}
				require.NoError(t, allocationManager.SetActuatedResources(originalPod, nil))
				t.Cleanup(func() { allocationManager.RemovePod(originalPod.UID) })

				allocationManager.(*manager).getPodByUID = func(uid types.UID) (*v1.Pod, bool) {
					return newPod, true
				}
				allocationManager.PushPendingResize(originalPod.UID)
				allocationManager.RetryPendingResizes()

				var updatedPod *v1.Pod
				if allocationManager.(*manager).statusManager.IsPodResizeInfeasible(newPod.UID) || allocationManager.(*manager).statusManager.IsPodResizeDeferred(newPod.UID) {
					updatedPod = originalPod
				} else {
					updatedPod = newPod
				}

				var updatedPodCtr v1.Container
				if isSidecarContainer {
					updatedPodCtr = updatedPod.Spec.InitContainers[0]
				} else {
					updatedPodCtr = updatedPod.Spec.Containers[0]
				}
				assert.Equal(t, tt.expectedAllocatedReqs, updatedPodCtr.Resources.Requests, "updated pod spec requests")
				assert.Equal(t, tt.expectedAllocatedLims, updatedPodCtr.Resources.Limits, "updated pod spec limits")

				alloc, found := allocationManager.GetContainerResourceAllocation(newPod.UID, updatedPodCtr.Name)
				require.True(t, found, "container allocation")
				assert.Equal(t, tt.expectedAllocatedReqs, alloc.Requests, "stored container request allocation")
				assert.Equal(t, tt.expectedAllocatedLims, alloc.Limits, "stored container limit allocation")

				resizeStatus := allocationManager.(*manager).statusManager.GetPodResizeConditions(newPod.UID)
				for i := range resizeStatus {
					// Ignore probe time and last transition time during comparison.
					resizeStatus[i].LastProbeTime = metav1.Time{}
					resizeStatus[i].LastTransitionTime = metav1.Time{}

					// Message is a substring assertion, since it can change slightly.
					assert.Contains(t, resizeStatus[i].Message, tt.expectedResize[i].Message)
					resizeStatus[i].Message = tt.expectedResize[i].Message
				}
				assert.Equal(t, tt.expectedResize, resizeStatus)
			})
		}
	}
}

func TestHandlePodResourcesResizeWithSwap(t *testing.T) {
	if goruntime.GOOS == "windows" {
		t.Skip("InPlacePodVerticalScaling is not currently supported for Windows")
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeSwap, true)
	noSwapContainerName, swapContainerName := "test-container-noswap", "test-container-limitedswap"

	cpu500m := resource.MustParse("500m")
	cpu1000m := resource.MustParse("1")
	mem500M := resource.MustParse("500Mi")
	mem1000M := resource.MustParse("1Gi")
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "1111",
			Name:      "pod1",
			Namespace: "ns1",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "c1",
					Image: "i1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name:               "c1",
					AllocatedResources: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
					Resources:          &v1.ResourceRequirements{},
				},
			},
		},
	}

	tests := []struct {
		name                  string
		newRequests           v1.ResourceList
		expectedAllocatedReqs v1.ResourceList
		resizePolicy          v1.ContainerResizePolicy
		swapBehavior          kubetypes.SwapBehavior
		expectedResize        []*v1.PodCondition
	}{
		{
			name:                  "NoSwap Request Memory decrease ResizePolicy RestartContainer - expect InProgress",
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			swapBehavior:          kubetypes.NoSwap,
			resizePolicy:          v1.ContainerResizePolicy{ResourceName: v1.ResourceMemory, RestartPolicy: v1.RestartContainer},
			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
		{
			name:                  "LimitedSwap Request Memory increase with ResizePolicy RestartContainer - expect InProgress",
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			swapBehavior:          kubetypes.LimitedSwap,
			resizePolicy:          v1.ContainerResizePolicy{ResourceName: v1.ResourceMemory, RestartPolicy: v1.RestartContainer},
			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
		{
			name:                  "LimitedSwap Request Memory increase with ResizePolicy NotRequired - expect Infeasible",
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			swapBehavior:          kubetypes.LimitedSwap,
			resizePolicy:          v1.ContainerResizePolicy{ResourceName: v1.ResourceMemory, RestartPolicy: v1.NotRequired},
			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Infeasible",
					Message: "In-place resize of containers with swap is not supported",
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalPod := testPod.DeepCopy()
			originalPod.Spec.Containers[0].ResizePolicy = []v1.ContainerResizePolicy{tt.resizePolicy}
			if tt.swapBehavior == kubetypes.NoSwap {
				originalPod.Spec.Containers[0].Name = noSwapContainerName
			} else {
				originalPod.Spec.Containers[0].Name = swapContainerName
			}
			newPod := originalPod.DeepCopy()
			newPod.Spec.Containers[0].Resources.Requests = tt.newRequests

			podStatus := &kubecontainer.PodStatus{
				ID:        originalPod.UID,
				Name:      originalPod.Name,
				Namespace: originalPod.Namespace,
			}
			setContainerStatus := func(podStatus *kubecontainer.PodStatus, c *v1.Container, idx int) {
				podStatus.ContainerStatuses[idx] = &kubecontainer.Status{
					Name:  c.Name,
					State: kubecontainer.ContainerStateRunning,
					Resources: &kubecontainer.ContainerResources{
						CPURequest:  c.Resources.Requests.Cpu(),
						CPULimit:    c.Resources.Limits.Cpu(),
						MemoryLimit: c.Resources.Limits.Memory(),
					},
				}
			}
			podStatus.ContainerStatuses = make([]*kubecontainer.Status, len(originalPod.Spec.Containers))
			for i, c := range originalPod.Spec.Containers {
				setContainerStatus(podStatus, &c, i)
			}
			runtime := &containertest.FakeRuntime{
				SwapBehavior: map[string]kubetypes.SwapBehavior{
					noSwapContainerName: kubetypes.NoSwap,
					swapContainerName:   kubetypes.LimitedSwap,
				},
				PodStatus: *podStatus,
			}
			allocationManager := makeAllocationManager(t, runtime, []*v1.Pod{testPod})

			require.NoError(t, allocationManager.SetAllocatedResources(originalPod))
			require.NoError(t, allocationManager.SetActuatedResources(originalPod, nil))
			t.Cleanup(func() { allocationManager.RemovePod(originalPod.UID) })

			allocationManager.(*manager).getPodByUID = func(uid types.UID) (*v1.Pod, bool) {
				return newPod, true
			}
			allocationManager.PushPendingResize(testPod.UID)
			allocationManager.RetryPendingResizes()

			var updatedPod *v1.Pod
			if allocationManager.(*manager).statusManager.IsPodResizeInfeasible(newPod.UID) {
				updatedPod = originalPod
			} else {
				updatedPod = newPod
			}

			updatedPodCtr := updatedPod.Spec.Containers[0]
			assert.Equal(t, tt.expectedAllocatedReqs, updatedPodCtr.Resources.Requests, "updated pod spec requests")

			alloc, found := allocationManager.GetContainerResourceAllocation(newPod.UID, updatedPodCtr.Name)
			require.True(t, found, "container allocation")
			assert.Equal(t, tt.expectedAllocatedReqs, alloc.Requests, "stored container request allocation")

			resizeStatus := allocationManager.(*manager).statusManager.GetPodResizeConditions(newPod.UID)
			for i := range resizeStatus {
				// Ignore probe time and last transition time during comparison.
				resizeStatus[i].LastProbeTime = metav1.Time{}
				resizeStatus[i].LastTransitionTime = metav1.Time{}
				assert.Contains(t, resizeStatus[i].Message, tt.expectedResize[i].Message)
				resizeStatus[i].Message = tt.expectedResize[i].Message
			}
			assert.Equal(t, tt.expectedResize, resizeStatus)
		})
	}
}

func TestIsResizeIncreasingAnyRequests(t *testing.T) {
	cpu500m := resource.MustParse("500m")
	cpu1000m := resource.MustParse("1")
	cpu1500m := resource.MustParse("1500m")
	mem500M := resource.MustParse("500Mi")
	mem1000M := resource.MustParse("1Gi")
	mem1500M := resource.MustParse("1500Mi")

	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "1111",
			Name:      "pod1",
			Namespace: "ns1",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "c1",
					Image: "i1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
					},
				},
				{
					Name:  "c2",
					Image: "i2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
					},
				},
				{
					Name:  "c3",
					Image: "i3",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{},
					},
				},
				{
					Name:  "c4",
					Image: "i4",
					Resources: v1.ResourceRequirements{
						Requests: nil,
					},
				},
			},
		},
	}
	allocationManager := makeAllocationManager(t, &containertest.FakeRuntime{}, []*v1.Pod{testPod})
	require.NoError(t, allocationManager.SetAllocatedResources(testPod))

	tests := []struct {
		name        string
		newRequests map[int]v1.ResourceList
		expected    bool
	}{
		{
			name:        "increase requests, one container",
			newRequests: map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M}},
			expected:    true,
		},
		{
			name:        "decrease requests, one container",
			newRequests: map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M}},
			expected:    false,
		},
		{
			name:        "increase cpu, decrease memory, one container",
			newRequests: map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem500M}},
			expected:    true,
		},
		{
			name:        "increase memory, decrease cpu, one container",
			newRequests: map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem1500M}},
			expected:    true,
		},
		{
			name: "increase one container, decrease another container",
			newRequests: map[int]v1.ResourceList{
				0: {v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M},
				1: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			},
			expected: true,
		},
		{
			name: "decrease requests, two containers",
			newRequests: map[int]v1.ResourceList{
				0: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
				1: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			},
			expected: false,
		},
		{
			name:        "remove requests, set as empty struct",
			newRequests: map[int]v1.ResourceList{0: {}},
			expected:    false,
		},
		{
			name:        "remove requests, set as nil",
			newRequests: map[int]v1.ResourceList{0: nil},
			expected:    false,
		},
		{
			name:        "add requests, set as empty struct",
			newRequests: map[int]v1.ResourceList{2: {v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M}},
			expected:    true,
		},
		{
			name:        "add requests, set as nil",
			newRequests: map[int]v1.ResourceList{3: {v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M}},
			expected:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for k, v := range tc.newRequests {
				testPod.Spec.Containers[k].Resources.Requests = v
			}
			require.Equal(t, tc.expected, allocationManager.(*manager).isResizeIncreasingAnyRequests(testPod))
		})
	}
}

func TestSortPendingPodsByPriority(t *testing.T) {
	cpu500m := resource.MustParse("500m")
	cpu1000m := resource.MustParse("1")
	cpu1500m := resource.MustParse("1500m")
	mem500M := resource.MustParse("500Mi")
	mem1000M := resource.MustParse("1Gi")
	mem1500M := resource.MustParse("1500Mi")

	createTestPod := func(podNumber int) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:       types.UID(fmt.Sprintf("%d", podNumber)),
				Name:      fmt.Sprintf("pod%d", podNumber),
				Namespace: fmt.Sprintf("ns%d", podNumber),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{
					Name:  fmt.Sprintf("c%d", podNumber),
					Image: fmt.Sprintf("i%d", podNumber),
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
					},
				}},
			},
		}
	}

	testPods := []*v1.Pod{createTestPod(0), createTestPod(1), createTestPod(2), createTestPod(3), createTestPod(4), createTestPod(5)}
	allocationManager := makeAllocationManager(t, &containertest.FakeRuntime{}, testPods)
	for _, testPod := range testPods {
		require.NoError(t, allocationManager.SetAllocatedResources(testPod))
	}

	// testPods[0] has the highest priority, as it doesn't increase resource requests.
	// testPods[1] has the highest PriorityClass.
	// testPods[2] is the only pod with QoS class "guaranteed" (all others are burstable).
	// testPods[3] has been in a "deferred" state for longer than testPods[4].
	// testPods[5] has no resize conditions yet, indicating it is a newer request than the other deferred resizes.

	testPods[0].Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M}
	for i := 1; i < len(testPods); i++ {
		testPods[i].Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M}
	}

	testPods[1].Spec.Priority = ptr.To(int32(100))
	testPods[2].Status.QOSClass = v1.PodQOSGuaranteed
	allocationManager.(*manager).statusManager.SetPodResizePendingCondition(testPods[3].UID, v1.PodReasonDeferred, "some-message")
	time.Sleep(5 * time.Millisecond)
	allocationManager.(*manager).statusManager.SetPodResizePendingCondition(testPods[4].UID, v1.PodReasonDeferred, "some-message")

	allocationManager.(*manager).getPodByUID = func(uid types.UID) (*v1.Pod, bool) {
		pods := map[types.UID]*v1.Pod{
			testPods[0].UID: testPods[0],
			testPods[1].UID: testPods[1],
			testPods[2].UID: testPods[2],
			testPods[3].UID: testPods[3],
			testPods[4].UID: testPods[4],
			testPods[5].UID: testPods[5],
		}
		pod, found := pods[uid]
		return pod, found
	}

	expected := []types.UID{testPods[0].UID, testPods[1].UID, testPods[2].UID, testPods[3].UID, testPods[4].UID, testPods[5].UID}

	// Push all the pods to the queue.
	for i := range testPods {
		allocationManager.PushPendingResize(testPods[i].UID)
	}
	require.Equal(t, expected, allocationManager.(*manager).podsWithPendingResizes)

	// Clear the queue and push the pods in reverse order to spice things up.
	allocationManager.(*manager).podsWithPendingResizes = nil
	for i := 5; i >= 0; i-- {
		allocationManager.PushPendingResize(testPods[i].UID)
	}
	require.Equal(t, expected, allocationManager.(*manager).podsWithPendingResizes)
}

func makeAllocationManager(t *testing.T, runtime *containertest.FakeRuntime, allocatedPods []*v1.Pod) Manager {
	t.Helper()
	statusManager := status.NewManager(&fake.Clientset{}, kubepod.NewBasicPodManager(), &statustest.FakePodDeletionSafetyProvider{}, kubeletutil.NewPodStartupLatencyTracker())
	containerManager := cm.NewFakeContainerManager()
	allocationManager := NewInMemoryManager(
		containerManager,
		statusManager,
		func(pod *v1.Pod) { /* no-op for testing */ },
		func() []*v1.Pod { return allocatedPods },
		func(uid types.UID) (*v1.Pod, bool) {
			for _, p := range allocatedPods {
				if p.UID == uid {
					return p, true
				}
			}
			return nil, false
		},
		containertest.NewFakeCache(runtime),
		config.NewSourcesReady(func(_ sets.Set[string]) bool { return true }),
	)
	allocationManager.SetContainerRuntime(runtime)

	getNode := func() (*v1.Node, error) {
		return &v1.Node{
			Status: v1.NodeStatus{
				Capacity: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("8"),
					v1.ResourceMemory: resource.MustParse("8Gi"),
				},
				Allocatable: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("4"),
					v1.ResourceMemory: resource.MustParse("4Gi"),
					v1.ResourcePods:   *resource.NewQuantity(40, resource.DecimalSI),
				},
			},
		}, nil
	}
	handler := lifecycle.NewPredicateAdmitHandler(getNode, lifecycle.NewAdmissionFailureHandlerStub(), allocationManager.(*manager).containerManager.UpdatePluginResources)
	allocationManager.AddPodAdmitHandlers(lifecycle.PodAdmitHandlers{handler})

	return allocationManager
}
