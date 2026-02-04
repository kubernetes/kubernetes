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
	"context"
	"fmt"
	goruntime "runtime"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/status"
	statustest "k8s.io/kubernetes/pkg/kubelet/status/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubeletutil "k8s.io/kubernetes/pkg/kubelet/util"
	_ "k8s.io/kubernetes/pkg/volume/hostpath"
	"k8s.io/kubernetes/test/utils/ktesting"
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

	podWithPodLevelResources := pod.DeepCopy()
	podWithPodLevelResources.Spec.Resources = &v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(1200, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(1700, resource.DecimalSI),
		},
		Limits: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(2000, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(2500, resource.DecimalSI),
		},
	}

	podWithoutContainerResources := pod.DeepCopy()
	podWithoutContainerResources.Spec.Containers[0].Resources = v1.ResourceRequirements{}
	podWithoutContainerResources.Spec.InitContainers[0].Resources = v1.ResourceRequirements{}

	resizedPod := pod.DeepCopy()
	resizedPod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(200, resource.DecimalSI)
	resizedPod.Spec.InitContainers[0].Resources.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(300, resource.DecimalSI)

	resizedPodWithPodLevelResources := resizedPod.DeepCopy()
	resizedPodWithPodLevelResources.Spec.Resources = &v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(1500, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(1700, resource.DecimalSI),
		},
		Limits: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(2200, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(2500, resource.DecimalSI),
		},
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, true)
	tests := []struct {
		name                         string
		pod                          *v1.Pod
		allocated                    state.PodResourceInfo
		expectPod                    *v1.Pod
		expectUpdate                 bool
		inPlacePodLevelResizeEnabled bool
	}{{
		name: "steady state",
		pod:  pod,
		allocated: state.PodResourceInfo{
			ContainerResources: map[string]v1.ResourceRequirements{
				"c1":                  *pod.Spec.Containers[0].Resources.DeepCopy(),
				"c2":                  *pod.Spec.Containers[1].Resources.DeepCopy(),
				"c1-restartable-init": *pod.Spec.InitContainers[0].Resources.DeepCopy(),
				"c1-init":             *pod.Spec.InitContainers[1].Resources.DeepCopy(),
			},
		},
		expectUpdate: false,
	}, {
		name:         "no allocations",
		pod:          pod,
		allocated:    state.PodResourceInfo{},
		expectUpdate: false,
	}, {
		name: "missing container allocation",
		pod:  pod,
		allocated: state.PodResourceInfo{
			ContainerResources: map[string]v1.ResourceRequirements{
				"c2": *pod.Spec.Containers[1].Resources.DeepCopy(),
			},
		},
		expectUpdate: false,
	}, {
		name: "resized container",
		pod:  pod,
		allocated: state.PodResourceInfo{
			ContainerResources: map[string]v1.ResourceRequirements{
				"c1":                  *resizedPod.Spec.Containers[0].Resources.DeepCopy(),
				"c2":                  *resizedPod.Spec.Containers[1].Resources.DeepCopy(),
				"c1-restartable-init": *resizedPod.Spec.InitContainers[0].Resources.DeepCopy(),
				"c1-init":             *resizedPod.Spec.InitContainers[1].Resources.DeepCopy(),
			},
		},
		expectUpdate: true,
		expectPod:    resizedPod,
	}, {
		name: "resized pod-level allocation",
		pod:  pod,
		allocated: state.PodResourceInfo{
			ContainerResources: map[string]v1.ResourceRequirements{
				"c1":                  *resizedPod.Spec.Containers[0].Resources.DeepCopy(),
				"c2":                  *resizedPod.Spec.Containers[1].Resources.DeepCopy(),
				"c1-restartable-init": *resizedPod.Spec.InitContainers[0].Resources.DeepCopy(),
				"c1-init":             *resizedPod.Spec.InitContainers[1].Resources.DeepCopy(),
			},
			PodLevelResources: resizedPodWithPodLevelResources.Spec.Resources.DeepCopy(),
		},
		expectUpdate:                 true,
		expectPod:                    resizedPodWithPodLevelResources,
		inPlacePodLevelResizeEnabled: true,
	}, {
		name: "resized pod-level resources and allocation",
		pod:  podWithPodLevelResources,
		allocated: state.PodResourceInfo{
			ContainerResources: map[string]v1.ResourceRequirements{
				"c1":                  *resizedPod.Spec.Containers[0].Resources.DeepCopy(),
				"c2":                  *resizedPod.Spec.Containers[1].Resources.DeepCopy(),
				"c1-restartable-init": *resizedPod.Spec.InitContainers[0].Resources.DeepCopy(),
				"c1-init":             *resizedPod.Spec.InitContainers[1].Resources.DeepCopy(),
			},
			PodLevelResources: resizedPodWithPodLevelResources.Spec.Resources.DeepCopy(),
		},
		expectUpdate:                 true,
		expectPod:                    resizedPodWithPodLevelResources,
		inPlacePodLevelResizeEnabled: true,
	}, {
		name: "resized pod-level resources and no container resources",
		pod:  podWithoutContainerResources,
		allocated: state.PodResourceInfo{
			ContainerResources: map[string]v1.ResourceRequirements{
				"c1":                  *resizedPod.Spec.Containers[0].Resources.DeepCopy(),
				"c2":                  *resizedPod.Spec.Containers[1].Resources.DeepCopy(),
				"c1-restartable-init": *resizedPod.Spec.InitContainers[0].Resources.DeepCopy(),
				"c1-init":             *resizedPod.Spec.InitContainers[1].Resources.DeepCopy(),
			},
			PodLevelResources: resizedPodWithPodLevelResources.Spec.Resources.DeepCopy(),
		},
		expectUpdate:                 true,
		expectPod:                    resizedPodWithPodLevelResources,
		inPlacePodLevelResizeEnabled: true,
	}, {
		name: "resized pod-level resources with feature gate disabled",
		pod:  podWithPodLevelResources,
		allocated: state.PodResourceInfo{
			ContainerResources: map[string]v1.ResourceRequirements{
				"c1":                  *pod.Spec.Containers[0].Resources.DeepCopy(),
				"c2":                  *pod.Spec.Containers[1].Resources.DeepCopy(),
				"c1-restartable-init": *pod.Spec.InitContainers[0].Resources.DeepCopy(),
				"c1-init":             *pod.Spec.InitContainers[1].Resources.DeepCopy(),
			},
			PodLevelResources: resizedPod.Spec.Resources.DeepCopy(),
		},
		expectUpdate:                 false,
		expectPod:                    podWithPodLevelResources,
		inPlacePodLevelResizeEnabled: false,
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.inPlacePodLevelResizeEnabled {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)
			}
			pod := test.pod.DeepCopy()
			allocatedPod, updated := updatePodFromAllocation(pod, test.allocated)

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

func TestRetryPendingResizes(t *testing.T) {
	if goruntime.GOOS == "windows" {
		t.Skip("InPlacePodVerticalScaling is not currently supported for Windows")
	}
	metrics.Register()
	metrics.PodInfeasibleResizes.Reset()

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways

	cpu2m := resource.MustParse("2m")
	cpu500m := resource.MustParse("500m")
	cpu1000m := resource.MustParse("1")
	cpu1200m := resource.MustParse("1200m")
	cpu1500m := resource.MustParse("1500m")
	cpu2000m := resource.MustParse("2")
	cpu2500m := resource.MustParse("2500m")
	cpu5000m := resource.MustParse("5000m")
	mem500M := resource.MustParse("500Mi")
	mem1000M := resource.MustParse("1Gi")
	mem1200M := resource.MustParse("1200Mi")
	mem1500M := resource.MustParse("1500Mi")
	mem2000M := resource.MustParse("2Gi")
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

	const (
		originalInProgressMsg = "original in-progress"
		originalPendingMsg    = "original pending"
	)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, true)
	tests := []struct {
		name                         string
		originalRequests             v1.ResourceList
		newRequests                  v1.ResourceList
		originalLimits               v1.ResourceList
		newLimits                    v1.ResourceList
		originalPodRequests          v1.ResourceList
		newPodRequests               v1.ResourceList
		originalInProgress           bool // Whether to prepopulate an in-progress condition.
		originalPending              bool // Whether to prepopulate a pending condition.
		newResourcesAllocated        bool // Whether the new requests have already been allocated (but not actuated)
		expectedAllocatedReqs        v1.ResourceList
		expectedAllocatedLims        v1.ResourceList
		expectedAllocatedPodReqs     v1.ResourceList
		expectedResize               []*v1.PodCondition
		expectPodSyncTriggered       string
		annotations                  map[string]string
		inPlacePodLevelResizeEnabled bool
	}{
		{
			name:                  "Request CPU and memory decrease - expect InProgress",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},

			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizeInProgress,
					Status:  "True",
					Message: "", // Expect the original InProgress condition to be cleared.
				},
			},
			expectPodSyncTriggered: "true",
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
			expectPodSyncTriggered: "true",
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
			expectPodSyncTriggered: "true",
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
					Message: "Node didn't have enough resource: cpu",
				},
			},
			expectPodSyncTriggered: "true",
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
			expectPodSyncTriggered: "true",
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
			expectPodSyncTriggered: "true",
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
			expectPodSyncTriggered: "true",
		},
		{
			name:                  "CPU increase in progress, resources unchanged - expect InProgress",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1000M},
			newResourcesAllocated: true,
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1000M},
			originalInProgress:    true,
			originalPending:       true,

			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizeInProgress,
					Status:  "True",
					Message: originalInProgressMsg, // Since the allocation didn't change, the condition isn't reset.
				},
			},
			// Should trigger a sync since the conditions changed (to update status), even though
			// the allocated resources are unchanged.
			expectPodSyncTriggered: "true",
		},
		{
			name:                  "CPU increase in progress, new allocation - expect InProgress reset",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1000M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1000M},
			originalInProgress:    true,
			originalPending:       true,

			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizeInProgress,
					Status:  "True",
					Message: "",
				},
			},
			expectPodSyncTriggered: "true",
		},
		{
			name:                  "Deferred resize doesn't clear in-progress status",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem2500M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			originalInProgress:    true,
			originalPending:       true,

			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Deferred",
					Message: "Node didn't have enough resource: memory", // Deferred condition should be updated.
				},
				{
					Type:    v1.PodResizeInProgress,
					Status:  "True",
					Message: originalInProgressMsg, // InProgress condition should be unchanged.
				},
			},
			expectPodSyncTriggered: "true",
		},
		{
			name:                  "No resize",
			originalRequests:      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:           v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedReqs: v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedResize:        nil,
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
			expectPodSyncTriggered: "true",
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
			expectPodSyncTriggered: "true",
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
			expectPodSyncTriggered: "true",
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
			expectPodSyncTriggered: "true",
		},
		{
			name:                         "pod-level: Request CPU and memory increase - expect InProgress",
			inPlacePodLevelResizeEnabled: true,
			originalPodRequests:          v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M},
			newPodRequests:               v1.ResourceList{v1.ResourceCPU: cpu2000m, v1.ResourceMemory: mem2000M},
			originalRequests:             v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:                  v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedReqs:        v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedPodReqs:     v1.ResourceList{v1.ResourceCPU: cpu2000m, v1.ResourceMemory: mem2000M},
			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
			expectPodSyncTriggered: "true",
		},
		{
			name:                         "pod-level: Request CPU and memory decrease - expect InProgress",
			inPlacePodLevelResizeEnabled: true,
			originalPodRequests:          v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M},
			newPodRequests:               v1.ResourceList{v1.ResourceCPU: cpu1200m, v1.ResourceMemory: mem1200M}, // still > container reqs
			originalRequests:             v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:                  v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedReqs:        v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedPodReqs:     v1.ResourceList{v1.ResourceCPU: cpu1200m, v1.ResourceMemory: mem1200M},
			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizeInProgress,
					Status:  "True",
					Message: "",
				},
			},
			expectPodSyncTriggered: "true",
		},
		{
			name:                         "pod-level: Request CPU increase beyond current capacity - expect Deferred",
			inPlacePodLevelResizeEnabled: true,
			originalPodRequests:          v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M},
			newPodRequests:               v1.ResourceList{v1.ResourceCPU: cpu2500m, v1.ResourceMemory: mem1500M},
			originalRequests:             v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:                  v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedReqs:        v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedPodReqs:     v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M},
			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Deferred",
					Message: "Node didn't have enough resource: cpu",
				},
			},
			expectPodSyncTriggered: "true",
		},
		{
			name:                         "pod-level: Request memory increase beyond node capacity - expect Infeasible",
			inPlacePodLevelResizeEnabled: true,
			originalPodRequests:          v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M},
			newPodRequests:               v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem4500M},
			originalRequests:             v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			newRequests:                  v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedReqs:        v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedPodReqs:     v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M},
			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Infeasible",
					Message: "Node didn't have enough capacity: memory, requested: 4718592000, capacity: 4294967296",
				},
			},
			expectPodSyncTriggered: "true",
		},
	}

	for _, tt := range tests {
		for _, isSidecarContainer := range []bool{false, true} {
			if tt.inPlacePodLevelResizeEnabled && isSidecarContainer {
				continue // pod level resources makes the distinction between container types irrelevant
			}
			t.Run(fmt.Sprintf("%s/sidecar=%t", tt.name, isSidecarContainer), func(t *testing.T) {
				if tt.inPlacePodLevelResizeEnabled {
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)
				}
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
				if tt.originalPodRequests != nil {
					originalPod.Spec.Resources = &v1.ResourceRequirements{Requests: tt.originalPodRequests.DeepCopy()}
				}

				newPod := originalPod.DeepCopy()
				if isSidecarContainer {
					newPod.Spec.InitContainers[0].Resources.Requests = tt.newRequests
					newPod.Spec.InitContainers[0].Resources.Limits = tt.newLimits
				} else {
					newPod.Spec.Containers[0].Resources.Requests = tt.newRequests
					newPod.Spec.Containers[0].Resources.Limits = tt.newLimits
				}
				if tt.newPodRequests != nil {
					if newPod.Spec.Resources == nil {
						newPod.Spec.Resources = &v1.ResourceRequirements{}
					}
					newPod.Spec.Resources.Requests = tt.newPodRequests.DeepCopy()
				}

				podStatus := &kubecontainer.PodStatus{
					ID:        originalPod.UID,
					Name:      originalPod.Name,
					Namespace: originalPod.Namespace,
				}

				podStatus.ContainerStatuses = make([]*kubecontainer.Status, len(originalPod.Spec.Containers)+len(originalPod.Spec.InitContainers))
				for i, c := range originalPod.Spec.InitContainers {
					setContainerStatus(podStatus, &c, i)
				}
				for i, c := range originalPod.Spec.Containers {
					setContainerStatus(podStatus, &c, i+len(originalPod.Spec.InitContainers))
				}
				allocationManager := makeAllocationManager(t, &containertest.FakeRuntime{PodStatus: *podStatus}, []*v1.Pod{testPod1, testPod2, testPod3}, nil)

				if !tt.newResourcesAllocated {
					require.NoError(t, allocationManager.SetAllocatedResources(originalPod))
				} else {
					require.NoError(t, allocationManager.SetAllocatedResources(newPod))
				}
				t.Cleanup(func() { allocationManager.RemovePod(originalPod.UID) })

				if tt.originalInProgress {
					allocationManager.(*manager).statusManager.SetPodResizeInProgressCondition(originalPod.UID, "", originalInProgressMsg, 0)
				}
				if tt.originalPending {
					allocationManager.(*manager).statusManager.SetPodResizePendingCondition(originalPod.UID, v1.PodReasonDeferred, originalPendingMsg, 0)
				}

				allocationManager.(*manager).getPodByUID = func(uid types.UID) (*v1.Pod, bool) {
					return newPod, true
				}
				allocationManager.PushPendingResize(originalPod.UID)
				allocationManager.RetryPendingResizes(TriggerReasonPodUpdated)

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

				if tt.inPlacePodLevelResizeEnabled {
					if tt.expectedAllocatedPodReqs != nil {
						require.NotNil(t, updatedPod.Spec.Resources)
						assert.Equal(t, tt.expectedAllocatedPodReqs, updatedPod.Spec.Resources.Requests, "updated pod spec pod requests")
					} else if updatedPod.Spec.Resources != nil {
						assert.Empty(t, updatedPod.Spec.Resources.Requests, "updated pod spec pod requests should be empty")
					}

					alloc, found := allocationManager.(*manager).allocated.GetPodResourceInfo(newPod.UID)
					if tt.expectedAllocatedPodReqs != nil {
						require.True(t, found, "pod allocation")
						if alloc.PodLevelResources == nil {
							assert.Equal(t, tt.expectedAllocatedPodReqs, alloc.PodLevelResources.Requests, "stored pod request allocation")
						}
					} else {
						require.False(t, found, "pod allocation should not be found")
					}
				}

				alloc, found := allocationManager.GetContainerResourceAllocation(newPod.UID, updatedPodCtr.Name)
				require.True(t, found, "container allocation")
				assert.Equal(t, tt.expectedAllocatedReqs, alloc.Requests, "stored container request allocation")
				assert.Equal(t, tt.expectedAllocatedLims, alloc.Limits, "stored container limit allocation")

				resizeStatus := allocationManager.(*manager).statusManager.GetPodResizeConditions(newPod.UID)
				if assert.Len(t, resizeStatus, len(tt.expectedResize), "different number of resize conditions") {
					for i := range resizeStatus {
						// Ignore probe time and last transition time during comparison.
						resizeStatus[i].LastProbeTime = metav1.Time{}
						resizeStatus[i].LastTransitionTime = metav1.Time{}

						// Message is a substring assertion, since it can change slightly.
						if tt.expectedResize[i].Message != "" {
							assert.Contains(t, resizeStatus[i].Message, tt.expectedResize[i].Message)
						} else {
							assert.Empty(t, resizeStatus[i].Message)
						}
						resizeStatus[i].Message = tt.expectedResize[i].Message
					}
					assert.Equal(t, tt.expectedResize, resizeStatus)
				}
				assert.Equal(t, tt.expectPodSyncTriggered, newPod.Annotations["pod-sync-triggered"], "pod sync annotation should be set")
			})
		}
	}

	expectedMetrics := `
		# HELP kubelet_pod_infeasible_resizes_total [ALPHA] Number of infeasible resizes for pods.
	    # TYPE kubelet_pod_infeasible_resizes_total counter
	    kubelet_pod_infeasible_resizes_total{reason_detail="insufficient_node_allocatable"} 5
	`
	assert.NoError(t, testutil.GatherAndCompare(
		legacyregistry.DefaultGatherer, strings.NewReader(expectedMetrics), "kubelet_pod_infeasible_resizes_total",
	))
}

func TestRetryPendingResizesGuanteedQOSPods(t *testing.T) {
	if goruntime.GOOS == "windows" {
		t.Skip("InPlacePodVerticalScaling is not currently supported for Windows")
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)

	nodeConfig := cm.NodeConfig{}
	nodeConfig.CPUManagerPolicy = string(cpumanager.PolicyStatic)
	nodeConfig.MemoryManagerPolicy = string(memorymanager.PolicyTypeStatic)

	createTestPod := func(uid types.UID, name, namespace string, req, lim v1.ResourceList, isSidecar bool) *v1.Pod {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{UID: uid, Name: name, Namespace: namespace},
			Status:     v1.PodStatus{Phase: v1.PodRunning},
		}

		if isSidecar {
			restartPolicy := v1.ContainerRestartPolicyAlways
			pod.Spec.InitContainers = []v1.Container{
				{
					Name:          "c1",
					Image:         "test-image",
					Resources:     v1.ResourceRequirements{Requests: req, Limits: lim},
					RestartPolicy: &restartPolicy,
				},
			}
			pod.Status.InitContainerStatuses = []v1.ContainerStatus{{Name: "c1", AllocatedResources: req}}
		} else {
			pod.Spec.Containers = []v1.Container{
				{
					Name:      "c1",
					Image:     "test-image",
					Resources: v1.ResourceRequirements{Requests: req, Limits: lim},
				},
			}
			pod.Status.ContainerStatuses = []v1.ContainerStatus{{Name: "c1", AllocatedResources: req}}
		}
		return pod
	}

	cpu500m := resource.MustParse("500m")
	cpu1000m := resource.MustParse("1")
	cpu2000m := resource.MustParse("2")
	mem500M := resource.MustParse("500Mi")
	mem1000M := resource.MustParse("1Gi")
	mem2000M := resource.MustParse("2Gi")

	// Set requests and limits to be the same for guaranteed QOS pods
	guaranteedQOSPod := createTestPod("1111", "pod1", "ns1", v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M}, v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M}, false)
	guaranteedQOSPodWithSidecar := createTestPod("2222", "pod2", "ns2", v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M}, v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M}, true)
	// Set requests and limits to be different for best effort pods
	bestEffortPod := createTestPod("333", "pod3", "ns3", v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M}, v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M}, false)
	bestEffortPodWithSidecar := createTestPod("444", "pod4", "ns4", v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M}, v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M}, true)

	tests := []struct {
		name                           string
		originalRequests               v1.ResourceList
		newRequests                    v1.ResourceList
		originalLimits                 v1.ResourceList
		podsToTest                     []*v1.Pod
		newLimits                      v1.ResourceList
		expectedAllocatedReqs          v1.ResourceList
		expectedAllocatedLims          v1.ResourceList
		expectedResize                 []*v1.PodCondition
		ipprExclusiveCPUsFeatureGate   bool
		ipprExclusiveMemoryFeatureGate bool
	}{
		{
			name:                           "Guaranteed QOS pod - CPU resize with `InPlacePodVerticalScalingExclusiveCPUs`feature gate off and only CPU resize. Infeasible due to static cpu policy",
			originalRequests:               v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			originalLimits:                 v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			podsToTest:                     []*v1.Pod{guaranteedQOSPod, guaranteedQOSPodWithSidecar},
			ipprExclusiveCPUsFeatureGate:   false,
			ipprExclusiveMemoryFeatureGate: true,
			newRequests:                    v1.ResourceList{v1.ResourceCPU: cpu2000m, v1.ResourceMemory: mem1000M},
			newLimits:                      v1.ResourceList{v1.ResourceCPU: cpu2000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedReqs:          v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedLims:          v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Infeasible",
					Message: "Resize is infeasible for Guaranteed Pods alongside CPU Manager policy \"static\"",
				},
			},
		},
		{
			name:                           "Guaranteed QOS pod - Memory resize with `InPlacePodVerticalScalingExclusiveMemory` feature gates off and only memory resize. Infeasible due to static memory policy",
			originalRequests:               v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			originalLimits:                 v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			podsToTest:                     []*v1.Pod{guaranteedQOSPod, guaranteedQOSPodWithSidecar},
			ipprExclusiveCPUsFeatureGate:   true,
			ipprExclusiveMemoryFeatureGate: false,
			newRequests:                    v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem2000M},
			newLimits:                      v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem2000M},
			expectedAllocatedReqs:          v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedLims:          v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Infeasible",
					Message: "Resize is infeasible for Guaranteed Pods alongside Memory Manager policy \"Static\"",
				},
			},
		},
		{
			name:                           "Guaranteed QOS pod - CPU and Memory resize with both feature gates off. Infeasible due to static cpu policy",
			originalRequests:               v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			originalLimits:                 v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			podsToTest:                     []*v1.Pod{guaranteedQOSPod, guaranteedQOSPodWithSidecar},
			ipprExclusiveCPUsFeatureGate:   false,
			ipprExclusiveMemoryFeatureGate: false,
			newRequests:                    v1.ResourceList{v1.ResourceCPU: cpu2000m, v1.ResourceMemory: mem2000M},
			newLimits:                      v1.ResourceList{v1.ResourceCPU: cpu2000m, v1.ResourceMemory: mem2000M},
			expectedAllocatedReqs:          v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedLims:          v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			expectedResize: []*v1.PodCondition{
				{
					Type:    v1.PodResizePending,
					Status:  "True",
					Reason:  "Infeasible",
					Message: "Resize is infeasible for Guaranteed Pods alongside CPU Manager policy \"static\"",
				},
			},
		},
		{
			name:                           "BestEffortPod - CPU and Memory limit resize. Resize in progess. Static CPU and Memory policy feature gate settings should not affect best effort pods",
			originalRequests:               v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			originalLimits:                 v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M},
			podsToTest:                     []*v1.Pod{bestEffortPod, bestEffortPodWithSidecar},
			ipprExclusiveCPUsFeatureGate:   false,
			ipprExclusiveMemoryFeatureGate: false,
			newRequests:                    v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			newLimits:                      v1.ResourceList{v1.ResourceCPU: cpu2000m, v1.ResourceMemory: mem1000M},
			expectedAllocatedReqs:          v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			expectedAllocatedLims:          v1.ResourceList{v1.ResourceCPU: cpu2000m, v1.ResourceMemory: mem1000M},
			expectedResize: []*v1.PodCondition{
				{
					Type:   v1.PodResizeInProgress,
					Status: "True",
				},
			},
		},
	}

	for _, tt := range tests {
		for _, originalPod := range tt.podsToTest {
			isSidecarContainer := len(originalPod.Spec.InitContainers) > 0
			t.Run(fmt.Sprintf("%s/sidecar=%t", tt.name, isSidecarContainer), func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.InPlacePodVerticalScalingExclusiveCPUs:   tt.ipprExclusiveCPUsFeatureGate,
					features.InPlacePodVerticalScalingExclusiveMemory: tt.ipprExclusiveMemoryFeatureGate,
				})
				var originalCtr *v1.Container

				if isSidecarContainer {
					originalCtr = &originalPod.Spec.InitContainers[0]
				} else {
					originalCtr = &originalPod.Spec.Containers[0]
				}

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
				allocationManager := makeAllocationManager(t, &containertest.FakeRuntime{PodStatus: *podStatus}, []*v1.Pod{guaranteedQOSPod, guaranteedQOSPodWithSidecar, bestEffortPod}, &nodeConfig)

				require.NoError(t, allocationManager.SetAllocatedResources(originalPod))
				t.Cleanup(func() { allocationManager.RemovePod(originalPod.UID) })

				allocationManager.(*manager).getPodByUID = func(uid types.UID) (*v1.Pod, bool) {
					return newPod, true
				}
				allocationManager.PushPendingResize(originalPod.UID)
				allocationManager.RetryPendingResizes(TriggerReasonPodUpdated)

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

func TestRetryPendingResizesWithSwap(t *testing.T) {
	if goruntime.GOOS == "windows" {
		t.Skip("InPlacePodVerticalScaling is not currently supported for Windows")
	}
	metrics.Register()
	metrics.PodInfeasibleResizes.Reset()

	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.InPlacePodVerticalScaling: true,
		features.NodeSwap:                  true,
	})
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
		expectedMetrics       string
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
			expectedMetrics: `
			    # HELP kubelet_pod_infeasible_resizes_total [ALPHA] Number of infeasible resizes for pods.
				# TYPE kubelet_pod_infeasible_resizes_total counter
				kubelet_pod_infeasible_resizes_total{reason_detail="swap_limitation"} 1
			`,
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
			allocationManager := makeAllocationManager(t, runtime, []*v1.Pod{testPod}, nil)

			require.NoError(t, allocationManager.SetAllocatedResources(originalPod))
			t.Cleanup(func() { allocationManager.RemovePod(originalPod.UID) })

			allocationManager.(*manager).getPodByUID = func(uid types.UID) (*v1.Pod, bool) {
				return newPod, true
			}
			allocationManager.PushPendingResize(testPod.UID)
			allocationManager.RetryPendingResizes(TriggerReasonPodUpdated)

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
			assert.Equal(t, "true", newPod.Annotations["pod-sync-triggered"], "pod sync annotation should be set")

			assert.NoError(t, testutil.GatherAndCompare(
				legacyregistry.DefaultGatherer, strings.NewReader(tt.expectedMetrics), "kubelet_pod_infeasible_resizes_total",
			))
		})
	}
}

func TestRetryPendingResizesMultipleConditions(t *testing.T) {
	if goruntime.GOOS == "windows" {
		t.Skip("InPlacePodVerticalScaling is not currently supported for Windows")
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)

	cpu500m := resource.MustParse("500m")
	cpu1000m := resource.MustParse("1")
	cpu5000m := resource.MustParse("5000m")
	mem500M := resource.MustParse("500Mi")
	mem1000M := resource.MustParse("1Gi")
	mem4500M := resource.MustParse("4500Mi")

	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:        "1111",
			Name:       "pod",
			Namespace:  "ns",
			Generation: 1,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "c1",
					Image: "i1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name:               "c1",
					AllocatedResources: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
					Resources:          &v1.ResourceRequirements{},
				},
			},
		},
	}

	podStatus := &kubecontainer.PodStatus{
		ID:        testPod.UID,
		Name:      testPod.Name,
		Namespace: testPod.Namespace,
	}

	podStatus.ContainerStatuses = make([]*kubecontainer.Status, len(testPod.Spec.Containers)+len(testPod.Spec.InitContainers))
	for i, c := range testPod.Spec.InitContainers {
		setContainerStatus(podStatus, &c, i)
	}
	for i, c := range testPod.Spec.Containers {
		setContainerStatus(podStatus, &c, i+len(testPod.Spec.InitContainers))
	}

	allocationManager := makeAllocationManager(t, &containertest.FakeRuntime{PodStatus: *podStatus}, []*v1.Pod{testPod}, nil)
	require.NoError(t, allocationManager.SetAllocatedResources(testPod))
	allocationManager.(*manager).getPodByUID = func(uid types.UID) (*v1.Pod, bool) {
		return testPod, true
	}

	testCases := []struct {
		name               string
		cpu                resource.Quantity
		mem                resource.Quantity
		generation         int64
		expectedConditions []*v1.PodCondition
		expectedEvent      string
	}{
		{
			name:       "allocated != actuated, pod resize should be in progress",
			cpu:        cpu1000m,
			mem:        mem1000M,
			generation: 1,
			expectedConditions: []*v1.PodCondition{{
				Type:               v1.PodResizeInProgress,
				Status:             "True",
				ObservedGeneration: 1,
			}},
			expectedEvent: `Normal ResizeStarted Pod resize started: {"containers":[{"name":"c1","resources":{"requests":{"cpu":"1","memory":"1Gi"}}}],"generation":1}`,
		},
		{
			name:       "desired != allocated != actuated, both conditions should be present in the pod status",
			cpu:        cpu5000m,
			mem:        mem4500M,
			generation: 2,
			expectedConditions: []*v1.PodCondition{
				{
					Type:               v1.PodResizePending,
					Status:             "True",
					Reason:             v1.PodReasonInfeasible,
					Message:            "Node didn't have enough capacity: memory, requested: 4718592000, capacity: 4294967296",
					ObservedGeneration: 2,
				},
				{
					Type:               v1.PodResizeInProgress,
					Status:             "True",
					ObservedGeneration: 1,
				},
			},
			expectedEvent: `Warning ResizeInfeasible Pod resize Infeasible: {"containers":[{"name":"c1","resources":{"requests":{"cpu":"5","memory":"4500Mi"}}}],"generation":2,"error":"Node didn't have enough capacity: memory, requested: 4718592000, capacity: 4294967296"}`,
		},
		{
			name:       "same as previous case to ensure no new event is generated",
			cpu:        cpu5000m,
			mem:        mem4500M,
			generation: 2,
			expectedConditions: []*v1.PodCondition{
				{
					Type:               v1.PodResizePending,
					Status:             "True",
					Reason:             v1.PodReasonInfeasible,
					Message:            "Node didn't have enough capacity: memory, requested: 4718592000, capacity: 4294967296",
					ObservedGeneration: 2,
				},
				{
					Type:               v1.PodResizeInProgress,
					Status:             "True",
					ObservedGeneration: 1,
				},
			},
		},
		{
			name:       "revert back to the original resize request",
			cpu:        cpu1000m,
			mem:        mem1000M,
			generation: 3,
			expectedConditions: []*v1.PodCondition{{
				Type:               v1.PodResizeInProgress,
				Status:             "True",
				ObservedGeneration: 1,
			}},
		},

		{
			name:       "no changes except generation",
			cpu:        cpu1000m,
			mem:        mem1000M,
			generation: 4,
			expectedConditions: []*v1.PodCondition{{
				Type:               v1.PodResizeInProgress,
				Status:             "True",
				ObservedGeneration: 1,
			}},
		},
		{
			name:       "allocate a new resize",
			cpu:        cpu500m,
			mem:        mem500M,
			generation: 5,
			expectedConditions: []*v1.PodCondition{{
				Type:               v1.PodResizeInProgress,
				Status:             "True",
				ObservedGeneration: 5,
			}},
			expectedEvent: `Normal ResizeStarted Pod resize started: {"containers":[{"name":"c1","resources":{"requests":{"cpu":"500m","memory":"500Mi"}}}],"generation":5}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testPod.Generation = tc.generation
			testPod.Spec = v1.PodSpec{
				Containers: []v1.Container{{
					Name:  "c1",
					Image: "i1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: tc.cpu, v1.ResourceMemory: tc.mem},
					}},
				},
			}

			allocationManager.PushPendingResize(testPod.UID)
			allocationManager.RetryPendingResizes(TriggerReasonPodUpdated)

			conditions := allocationManager.(*manager).statusManager.GetPodResizeConditions(testPod.UID)
			require.Len(t, conditions, len(tc.expectedConditions))
			for _, c := range conditions {
				c.LastProbeTime = metav1.Time{}
				c.LastTransitionTime = metav1.Time{}
			}
			require.Equal(t, tc.expectedConditions, conditions)

			fakeRecorder := allocationManager.(*manager).recorder.(*record.FakeRecorder)
			if tc.expectedEvent != "" {
				require.Len(t, fakeRecorder.Events, 1)
				event := <-fakeRecorder.Events
				require.Equal(t, tc.expectedEvent, event)
			} else {
				require.Empty(t, fakeRecorder.Events)
			}
		})
	}
}

// testPodAdmitHandler is a lifecycle.PodAdmitHandler for testing.
type testPodAdmitHandler struct {
	// admitFunc contains the custom logic for admitting or rejecting a pod.
	admitFunc func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult
}

// Admit rejects all pods in the podsToReject list with a matching UID.
func (a *testPodAdmitHandler) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	if a.admitFunc == nil {
		return lifecycle.PodAdmitResult{Admit: true}
	}
	return a.admitFunc(attrs)
}

func TestAllocationManagerAddPodWithPLR(t *testing.T) {
	if goruntime.GOOS == "windows" {
		t.Skip("InPlacePodVerticalScaling is not currently supported for Windows")
	}

	const containerName = "c1"

	cpu1Mem1G := v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("1Gi")}
	cpu2Mem2G := v1.ResourceList{v1.ResourceCPU: resource.MustParse("2"), v1.ResourceMemory: resource.MustParse("2Gi")}

	createTestPod := func(uid types.UID, name, namespace string, resources v1.ResourceList) *v1.Pod {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{UID: uid, Name: name, Namespace: namespace},
		}
		container := v1.Container{
			Name:  containerName,
			Image: "i1",
			Resources: v1.ResourceRequirements{
				Requests: resources,
			},
		}
		pod.Spec.Containers = append(pod.Spec.Containers, container)
		return pod
	}

	pod1UID := types.UID("1111")
	pod2UID := types.UID("2222")
	pod1Small := createTestPod(pod1UID, "pod1", "ns1", cpu1Mem1G)
	pod1Large := createTestPod(pod1UID, "pod1", "ns1", cpu2Mem2G)
	pod2Small := createTestPod(pod2UID, "pod2", "ns2", cpu1Mem1G)
	pod2Large := createTestPod(pod2UID, "pod2", "ns2", cpu2Mem2G)

	pod1SmallWithPLR := pod1Small.DeepCopy()
	pod1SmallWithPLR.Spec.Resources = &v1.ResourceRequirements{Requests: cpu1Mem1G}
	pod1LargeWithPLR := pod1Large.DeepCopy()
	pod1LargeWithPLR.Spec.Resources = &v1.ResourceRequirements{Requests: cpu2Mem2G}
	pod2SmallWithPLR := pod2Small.DeepCopy()
	pod2SmallWithPLR.Spec.Resources = &v1.ResourceRequirements{Requests: cpu1Mem1G}
	pod2LargeWithPLR := pod2Large.DeepCopy()
	pod2LargeWithPLR.Spec.Resources = &v1.ResourceRequirements{Requests: cpu2Mem2G}

	type resourceState struct {
		podResources       v1.ResourceList
		containerResources v1.ResourceList
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, true)
	testCases := []struct {
		name                            string
		initialAllocatedResourcesState  map[types.UID]resourceState
		currentActivePods               []*v1.Pod
		podToAdd                        *v1.Pod
		admitFunc                       func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult
		expectAdmit                     bool
		admissionFailureReason          string
		admissionFailureMessage         string
		expectedAllocatedResourcesState map[types.UID]resourceState
		ipprPLRFeatureGate              bool
	}{
		{
			name:                           "PLR IPPR Enabled - New pod admitted and allocated resources updated",
			initialAllocatedResourcesState: map[types.UID]resourceState{},
			currentActivePods:              []*v1.Pod{},
			podToAdd:                       pod1SmallWithPLR,
			admitFunc:                      nil,
			expectAdmit:                    true,
			// allocated resources updated with pod1's resources
			expectedAllocatedResourcesState: map[types.UID]resourceState{pod1UID: {podResources: cpu1Mem1G, containerResources: cpu1Mem1G}},
			ipprPLRFeatureGate:              true,
		},
		{
			name:                            "PLR IPPR Disabled - New pod admitted with PLR but allocated pod-level resources not updated",
			initialAllocatedResourcesState:  map[types.UID]resourceState{},
			currentActivePods:               []*v1.Pod{},
			podToAdd:                        pod1SmallWithPLR,
			admitFunc:                       nil,
			expectAdmit:                     true,
			expectedAllocatedResourcesState: map[types.UID]resourceState{pod1UID: {containerResources: cpu1Mem1G}},
		},
		{
			name:                           "PLR IPPR Enabled - New pod not admitted due to insufficient resources",
			initialAllocatedResourcesState: map[types.UID]resourceState{pod1UID: {containerResources: cpu1Mem1G}},
			currentActivePods:              []*v1.Pod{pod1Small},
			podToAdd:                       pod2LargeWithPLR,
			admitFunc: func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
				cpuRequest := attrs.Pod.Spec.Resources.Requests.Cpu().Value()
				if cpuRequest > 1 {
					return lifecycle.PodAdmitResult{
						Admit:   false,
						Reason:  "OutOfcpu",
						Message: fmt.Sprintf("not enough CPUs available for pod %s/%s, requested: %d, available:1", attrs.Pod.Namespace, attrs.Pod.Name, cpuRequest),
					}
				}
				return lifecycle.PodAdmitResult{Admit: true}
			},
			expectAdmit:             false,
			admissionFailureReason:  "OutOfcpu",
			admissionFailureMessage: "not enough CPUs available for pod ns2/pod2, requested: 2, available:1",
			// allocated resources not modified
			expectedAllocatedResourcesState: map[types.UID]resourceState{pod1UID: {containerResources: cpu1Mem1G}},
			ipprPLRFeatureGate:              true,
		},
		{
			name:                           "PLR IPPR Disabled - New pod not admitted due to insufficient resources",
			initialAllocatedResourcesState: map[types.UID]resourceState{pod1UID: {containerResources: cpu1Mem1G}},
			currentActivePods:              []*v1.Pod{pod1Small},
			podToAdd:                       pod2LargeWithPLR,
			admitFunc: func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
				cpuRequest := attrs.Pod.Spec.Resources.Requests.Cpu().Value()
				if cpuRequest > 1 {
					return lifecycle.PodAdmitResult{
						Admit:   false,
						Reason:  "OutOfcpu",
						Message: fmt.Sprintf("not enough CPUs available for pod %s/%s, requested: %d, available:1", attrs.Pod.Namespace, attrs.Pod.Name, cpuRequest),
					}
				}
				return lifecycle.PodAdmitResult{Admit: true}
			},
			expectAdmit:             false,
			admissionFailureReason:  "OutOfcpu",
			admissionFailureMessage: "not enough CPUs available for pod ns2/pod2, requested: 2, available:1",
			// allocated resources not modified
			expectedAllocatedResourcesState: map[types.UID]resourceState{pod1UID: {containerResources: cpu1Mem1G}},
		},
		{
			name: "PLR IPPR Enabled - no pod resize request. Resource request same as existing allocation",
			initialAllocatedResourcesState: map[types.UID]resourceState{
				pod1UID: {containerResources: cpu1Mem1G, podResources: cpu1Mem1G},
				pod2UID: {containerResources: cpu1Mem1G, podResources: cpu1Mem1G},
			},
			currentActivePods: []*v1.Pod{pod1SmallWithPLR},
			podToAdd:          pod1SmallWithPLR,
			admitFunc:         nil,
			expectAdmit:       true,
			expectedAllocatedResourcesState: map[types.UID]resourceState{
				pod1UID: {containerResources: cpu1Mem1G, podResources: cpu1Mem1G},
				pod2UID: {containerResources: cpu1Mem1G, podResources: cpu1Mem1G},
			},
			ipprPLRFeatureGate: true,
		},
		{
			name:                           "PLR IPPR Enabled - current allocation not found for added pod. Allocated resources updated.",
			initialAllocatedResourcesState: map[types.UID]resourceState{},
			currentActivePods:              []*v1.Pod{pod1Small},
			podToAdd:                       pod1LargeWithPLR,
			admitFunc:                      nil,
			expectAdmit:                    true,
			// pod2's resources added to allocated resources
			expectedAllocatedResourcesState: map[types.UID]resourceState{pod1UID: {containerResources: cpu2Mem2G, podResources: cpu2Mem2G}},
			ipprPLRFeatureGate:              true,
		},
		{
			name: "PLR IPPR Enabled - request different from current allocation. Pod still admitted based on existing allocation, but allocated resources remains unchanges.",
			initialAllocatedResourcesState: map[types.UID]resourceState{
				pod1UID: {containerResources: cpu1Mem1G, podResources: cpu1Mem1G},
				pod2UID: {containerResources: cpu1Mem1G, podResources: cpu1Mem1G},
			},
			currentActivePods: []*v1.Pod{pod1SmallWithPLR, pod2SmallWithPLR},
			podToAdd:          pod1LargeWithPLR,
			admitFunc: func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
				cpuRequest := attrs.Pod.Spec.Resources.Requests.Cpu().Value()
				if cpuRequest > 1 {
					return lifecycle.PodAdmitResult{
						Admit:   false,
						Reason:  "OutOfcpu",
						Message: fmt.Sprintf("not enough CPUs available for pod %s/%s, requested: %d, available:1", attrs.Pod.Namespace, attrs.Pod.Name, cpuRequest),
					}
				}
				return lifecycle.PodAdmitResult{Admit: true}
			},
			// pod is still admitted as allocated resources are considered during admission.
			expectAdmit: true,
			//  allocated Resources state must not be updated.
			expectedAllocatedResourcesState: map[types.UID]resourceState{
				pod1UID: {containerResources: cpu1Mem1G, podResources: cpu1Mem1G},
				pod2UID: {containerResources: cpu1Mem1G, podResources: cpu1Mem1G},
			},
			ipprPLRFeatureGate: true,
		},
		{
			name:                           "PLR IPPR Disabled - request different from current allocation. Admission fails. Allocated resources not updated.",
			initialAllocatedResourcesState: map[types.UID]resourceState{pod1UID: {containerResources: cpu1Mem1G}},
			currentActivePods:              []*v1.Pod{pod1SmallWithPLR},
			podToAdd:                       pod1LargeWithPLR,
			admitFunc: func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
				cpuRequest := attrs.Pod.Spec.Resources.Requests.Cpu().Value()
				if cpuRequest > 1 {
					return lifecycle.PodAdmitResult{
						Admit:   false,
						Reason:  "OutOfcpu",
						Message: fmt.Sprintf("not enough CPUs available for pod %s/%s, requested: %d, available:1", attrs.Pod.Namespace, attrs.Pod.Name, cpuRequest),
					}
				}
				return lifecycle.PodAdmitResult{Admit: true}
			},
			// pod is still admitted as allocated resources are considered during admission.
			expectAdmit:             false,
			admissionFailureReason:  "OutOfcpu",
			admissionFailureMessage: "not enough CPUs available for pod ns1/pod1, requested: 2, available:1",
			// allocated Resources state must not be updated.
			expectedAllocatedResourcesState: map[types.UID]resourceState{
				pod1UID: {containerResources: cpu1Mem1G},
			},
			ipprPLRFeatureGate: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, tc.ipprPLRFeatureGate)
			allocationManager := makeAllocationManager(t, &containertest.FakeRuntime{}, []*v1.Pod{}, nil)

			podForAllocation := func(uid types.UID, resources resourceState) *v1.Pod {
				pod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{UID: uid},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name:      containerName,
							Resources: v1.ResourceRequirements{Requests: resources.containerResources},
						}},
					},
				}

				if resources.podResources != nil {
					pod.Spec.Resources = &v1.ResourceRequirements{Requests: resources.podResources}
				}
				return pod
			}

			for podUID, resources := range tc.initialAllocatedResourcesState {
				err := allocationManager.SetAllocatedResources(podForAllocation(podUID, resources))
				require.NoError(t, err)
			}

			if tc.admitFunc != nil {
				handler := &testPodAdmitHandler{admitFunc: tc.admitFunc}
				allocationManager.AddPodAdmitHandlers(lifecycle.PodAdmitHandlers{handler})
			}

			ok, reason, message := allocationManager.AddPod(tc.currentActivePods, tc.podToAdd)
			require.Equal(t, tc.expectAdmit, ok)
			require.Equal(t, tc.admissionFailureReason, reason)
			require.Equal(t, tc.admissionFailureMessage, message)

			for podUID, resources := range tc.expectedAllocatedResourcesState {
				pod := podForAllocation(podUID, resources)
				for _, container := range pod.Spec.Containers {
					allocatedResources, found := allocationManager.GetContainerResourceAllocation(pod.UID, container.Name)
					if pod.UID == tc.podToAdd.UID {
						if tc.expectAdmit && !found {
							t.Fatalf("resource allocation should exist for pod: %s", tc.podToAdd.Name)
						}
						if !tc.expectAdmit && found {
							initialResources := tc.initialAllocatedResourcesState[pod.UID]
							// allocated resources should not be modified when the pod is not admitted
							assert.Equal(t, initialResources.containerResources, allocatedResources.Requests, tc.name)
						}
					}
					assert.Equal(t, container.Resources, allocatedResources, tc.name)
				}
				if pod.Spec.Resources != nil {
					allocatedPodResources, found := allocationManager.GetPodLevelResourceAllocation(pod.UID)
					if tc.expectAdmit && !found {
						t.Fatalf("resource allocation should exist for pod: %s", tc.podToAdd.Name)
					}
					if !tc.expectAdmit && found {
						initialResources := tc.initialAllocatedResourcesState[pod.UID]
						// allocated resources should not be modified when the pod
						// is not admitted
						if initialResources.podResources == nil {
							if allocatedPodResources != nil {
								t.Fatalf("resource allocation should nil for pod: %s", tc.podToAdd.Name)
							}
						}
						if allocatedPodResources == nil {
							t.Fatalf("resource allocation should exist for pod: %s", tc.podToAdd.Name)
						}

						assert.Equal(t, initialResources.podResources, allocatedPodResources.Requests, tc.name)
					}
					assert.Equal(t, pod.Spec.Resources, allocatedPodResources, tc.name)
				}
			}
		})
	}
}

func TestAllocationManagerAddPod(t *testing.T) {
	if goruntime.GOOS == "windows" {
		t.Skip("InPlacePodVerticalScaling is not currently supported for Windows")
	}

	const containerName = "c1"

	cpu1Mem1G := v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("1Gi")}
	cpu2Mem2G := v1.ResourceList{v1.ResourceCPU: resource.MustParse("2"), v1.ResourceMemory: resource.MustParse("2Gi")}

	createTestPod := func(uid types.UID, name, namespace string, resources v1.ResourceList) *v1.Pod {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{UID: uid, Name: name, Namespace: namespace},
		}
		container := v1.Container{
			Name:  containerName,
			Image: "i1",
			Resources: v1.ResourceRequirements{
				Requests: resources,
			},
		}
		pod.Spec.Containers = append(pod.Spec.Containers, container)
		return pod
	}

	pod1UID := types.UID("1111")
	pod2UID := types.UID("2222")
	pod1Small := createTestPod(pod1UID, "pod1", "ns1", cpu1Mem1G)
	pod1Large := createTestPod(pod1UID, "pod1", "ns1", cpu2Mem2G)
	pod2Small := createTestPod(pod2UID, "pod2", "ns2", cpu1Mem1G)
	pod2Large := createTestPod(pod2UID, "pod2", "ns2", cpu2Mem2G)

	testCases := []struct {
		name                            string
		ipprFeatureGate                 bool
		initialAllocatedResourcesState  map[types.UID]v1.ResourceList
		currentActivePods               []*v1.Pod
		podToAdd                        *v1.Pod
		admitFunc                       func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult
		expectAdmit                     bool
		admissionFailureReason          string
		admissionFailureMessage         string
		expectedAllocatedResourcesState map[types.UID]v1.ResourceList
	}{
		{
			name:                           "IPPR Enabled - New pod admitted and allocated resources updated",
			ipprFeatureGate:                true,
			initialAllocatedResourcesState: map[types.UID]v1.ResourceList{},
			currentActivePods:              []*v1.Pod{},
			podToAdd:                       pod1Small,
			admitFunc:                      nil,
			expectAdmit:                    true,
			// allocated resources updated with pod1's resources
			expectedAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G},
		},
		{
			name:                            "IPPR Disabled - New pod admitted but allocated resources not updated",
			ipprFeatureGate:                 false,
			initialAllocatedResourcesState:  map[types.UID]v1.ResourceList{},
			currentActivePods:               []*v1.Pod{},
			podToAdd:                        pod1Small,
			admitFunc:                       nil,
			expectAdmit:                     true,
			expectedAllocatedResourcesState: map[types.UID]v1.ResourceList{},
		},
		{
			name:                           "IPPR Enabled - New pod not admititted due to insufficient resources",
			ipprFeatureGate:                true,
			initialAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G},
			currentActivePods:              []*v1.Pod{pod1Small},
			podToAdd:                       pod2Large,
			admitFunc: func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
				cpuRequest := attrs.Pod.Spec.Containers[0].Resources.Requests.Cpu().Value()
				if cpuRequest > 1 {
					return lifecycle.PodAdmitResult{
						Admit:   false,
						Reason:  "OutOfcpu",
						Message: fmt.Sprintf("not enough CPUs available for pod %s/%s, requested: %d, available:1", attrs.Pod.Namespace, attrs.Pod.Name, cpuRequest),
					}
				}
				return lifecycle.PodAdmitResult{Admit: true}
			},
			expectAdmit:             false,
			admissionFailureReason:  "OutOfcpu",
			admissionFailureMessage: "not enough CPUs available for pod ns2/pod2, requested: 2, available:1",
			// allocated resources not modified
			expectedAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G},
		},
		{
			name:                           "IPPR Disabled - New pod not admitted due to insufficient resources",
			ipprFeatureGate:                false,
			initialAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G},
			currentActivePods:              []*v1.Pod{pod1Small},
			podToAdd:                       pod2Large,
			admitFunc: func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
				cpuRequest := attrs.Pod.Spec.Containers[0].Resources.Requests.Cpu().Value()
				if cpuRequest > 1 {
					return lifecycle.PodAdmitResult{
						Admit:   false,
						Reason:  "OutOfcpu",
						Message: fmt.Sprintf("not enough CPUs available for pod %s/%s, requested: %d, available:1", attrs.Pod.Namespace, attrs.Pod.Name, cpuRequest),
					}
				}
				return lifecycle.PodAdmitResult{Admit: true}
			},
			expectAdmit:             false,
			admissionFailureReason:  "OutOfcpu",
			admissionFailureMessage: "not enough CPUs available for pod ns2/pod2, requested: 2, available:1",
			// allocated resources not modified
			expectedAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G},
		},
		{
			name:                            "IPPR Enabled - no pod resize request. Resource request same as existing allocation",
			ipprFeatureGate:                 true,
			initialAllocatedResourcesState:  map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G, pod2UID: cpu1Mem1G},
			currentActivePods:               []*v1.Pod{pod1Small},
			podToAdd:                        pod1Small,
			admitFunc:                       nil,
			expectAdmit:                     true,
			expectedAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G, pod2UID: cpu1Mem1G},
		},
		{
			name:                           "IPPR Enabled - current allocation not found for added pod. Allocated resources updated.",
			ipprFeatureGate:                true,
			initialAllocatedResourcesState: map[types.UID]v1.ResourceList{},
			currentActivePods:              []*v1.Pod{pod1Small},
			podToAdd:                       pod1Large,
			admitFunc:                      nil,
			expectAdmit:                    true,
			// pod2's resources added to allocated resources
			expectedAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu2Mem2G},
		},
		{
			name:                           "IPPR Enabled - request different from current allocation. Pod still admitted based on existing allocation, but allocated resources remains unchanges.",
			ipprFeatureGate:                true,
			initialAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G, pod2UID: cpu1Mem1G},
			currentActivePods:              []*v1.Pod{pod1Small, pod2Small},
			podToAdd:                       pod1Large,
			admitFunc: func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
				cpuRequest := attrs.Pod.Spec.Containers[0].Resources.Requests.Cpu().Value()
				if cpuRequest > 1 {
					return lifecycle.PodAdmitResult{
						Admit:   false,
						Reason:  "OutOfcpu",
						Message: fmt.Sprintf("not enough CPUs available for pod %s/%s, requested: %d, available:1", attrs.Pod.Namespace, attrs.Pod.Name, cpuRequest),
					}
				}
				return lifecycle.PodAdmitResult{Admit: true}
			},
			// pod is still admitted as allocated resources are considered during admission.
			expectAdmit: true,
			//  allocated Resources state must not be updated.
			expectedAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G, pod2UID: cpu1Mem1G},
		},
		{
			name:                           "IPPR Disabled - request different from current allocation. Admission fails. Allocated resources not updated.",
			ipprFeatureGate:                false,
			initialAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G},
			currentActivePods:              []*v1.Pod{pod1Small},
			podToAdd:                       pod1Large,
			admitFunc: func(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
				cpuRequest := attrs.Pod.Spec.Containers[0].Resources.Requests.Cpu().Value()
				if cpuRequest > 1 {
					return lifecycle.PodAdmitResult{
						Admit:   false,
						Reason:  "OutOfcpu",
						Message: fmt.Sprintf("not enough CPUs available for pod %s/%s, requested: %d, available:1", attrs.Pod.Namespace, attrs.Pod.Name, cpuRequest),
					}
				}
				return lifecycle.PodAdmitResult{Admit: true}
			},
			// pod is still admitted as allocated resources are considered during admission.
			expectAdmit:             false,
			admissionFailureReason:  "OutOfcpu",
			admissionFailureMessage: "not enough CPUs available for pod ns1/pod1, requested: 2, available:1",
			// allocated Resources state must not be updated.
			expectedAllocatedResourcesState: map[types.UID]v1.ResourceList{pod1UID: cpu1Mem1G},
		},
	}

	for _, tc := range testCases {
		featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.34"))
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, tc.ipprFeatureGate)
			allocationManager := makeAllocationManager(t, &containertest.FakeRuntime{}, []*v1.Pod{}, nil)

			podForAllocation := func(uid types.UID, resources v1.ResourceList) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{UID: uid},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name:      containerName,
							Resources: v1.ResourceRequirements{Requests: resources},
						}},
					},
				}
			}

			for podUID, resources := range tc.initialAllocatedResourcesState {
				err := allocationManager.SetAllocatedResources(podForAllocation(podUID, resources))
				require.NoError(t, err)
			}

			if tc.admitFunc != nil {
				handler := &testPodAdmitHandler{admitFunc: tc.admitFunc}
				allocationManager.AddPodAdmitHandlers(lifecycle.PodAdmitHandlers{handler})
			}

			ok, reason, message := allocationManager.AddPod(tc.currentActivePods, tc.podToAdd)
			require.Equal(t, tc.expectAdmit, ok)
			require.Equal(t, tc.admissionFailureReason, reason)
			require.Equal(t, tc.admissionFailureMessage, message)

			for podUID, resources := range tc.expectedAllocatedResourcesState {
				pod := podForAllocation(podUID, resources)
				for _, container := range pod.Spec.Containers {
					allocatedResources, found := allocationManager.GetContainerResourceAllocation(pod.UID, container.Name)
					if pod.UID == tc.podToAdd.UID {
						if tc.expectAdmit && !found {
							t.Fatalf("resource allocation should exist for pod: %s", tc.podToAdd.Name)
						}
						if !tc.expectAdmit && found {
							initialResources := tc.initialAllocatedResourcesState[pod.UID]
							// allocated resources should not be modified when the pod is not admitted
							assert.Equal(t, initialResources, allocatedResources.Requests, tc.name)
						}
					}
					assert.Equal(t, container.Resources, allocatedResources, tc.name)
				}
			}
		})
	}
}

func TestIsResizeIncreasingRequests(t *testing.T) {
	cpu500m := resource.MustParse("500m")
	cpu1000m := resource.MustParse("1")
	cpu1500m := resource.MustParse("1500m")
	cpu2500m := resource.MustParse("2500m")
	mem500M := resource.MustParse("500Mi")
	mem1000M := resource.MustParse("1Gi")
	mem1500M := resource.MustParse("1500Mi")
	mem2500M := resource.MustParse("2500Mi")

	tests := []struct {
		name                         string
		newPodRequests               v1.ResourceList
		newContRequests              map[int]v1.ResourceList
		expected                     bool
		inPlacePodLevelResizeEnabled bool
	}{
		{
			name:            "increase requests, one container",
			newContRequests: map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M}},
			expected:        true,
		},
		{
			name:            "decrease requests, one container",
			newContRequests: map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M}},
			expected:        false,
		},
		{
			name:            "increase cpu, decrease memory, one container",
			newContRequests: map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem500M}},
			expected:        true,
		},
		{
			name:            "increase memory, decrease cpu, one container",
			newContRequests: map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem1500M}},
			expected:        true,
		},
		{
			name: "increase one container, decrease another container, net neutral",
			newContRequests: map[int]v1.ResourceList{
				0: {v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M},
				1: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			},
			expected: false,
		},
		{
			name: "decrease requests, two containers",
			newContRequests: map[int]v1.ResourceList{
				0: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
				1: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			},
			expected: false,
		},
		{
			name:            "remove requests, set as empty struct",
			newContRequests: map[int]v1.ResourceList{0: {}},
			expected:        false,
		},
		{
			name:            "remove requests, set as nil",
			newContRequests: map[int]v1.ResourceList{0: nil},
			expected:        false,
		},
		{
			name:            "add requests, set as empty struct",
			newContRequests: map[int]v1.ResourceList{2: {v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M}},
			expected:        true,
		},
		{
			name:            "add requests, set as nil",
			newContRequests: map[int]v1.ResourceList{3: {v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem1000M}},
			expected:        true,
		},
		{
			name:                         "pod-level: increase cpu",
			newPodRequests:               v1.ResourceList{v1.ResourceCPU: resource.MustParse("3000m"), v1.ResourceMemory: resource.MustParse("2500Mi")},
			expected:                     true,
			inPlacePodLevelResizeEnabled: true,
		},
		{
			name:                         "pod-level: decrease memory",
			newPodRequests:               v1.ResourceList{v1.ResourceCPU: resource.MustParse("2500m"), v1.ResourceMemory: resource.MustParse("2000Mi")},
			expected:                     false,
			inPlacePodLevelResizeEnabled: true,
		},
		{
			name:                         "pod-level: increase cpu, decrease memory",
			newPodRequests:               v1.ResourceList{v1.ResourceCPU: resource.MustParse("3000m"), v1.ResourceMemory: resource.MustParse("2000Mi")},
			expected:                     true,
			inPlacePodLevelResizeEnabled: true,
		},
		{
			name:                         "container-level: increase cpu",
			newContRequests:              map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1000M}},
			expected:                     false,
			inPlacePodLevelResizeEnabled: true,
		},
		{
			name:                         "container-level: decrease cpu",
			newContRequests:              map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem1000M}},
			expected:                     false,
			inPlacePodLevelResizeEnabled: true,
		},
		{
			name:                         "pod & container: increase container cpu, increase pod cpu, net increase",
			newPodRequests:               v1.ResourceList{v1.ResourceCPU: resource.MustParse("2600m"), v1.ResourceMemory: resource.MustParse("2500Mi")}, // +100m
			newContRequests:              map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1000M}},                           // +500m
			expected:                     true,
			inPlacePodLevelResizeEnabled: true,
		},
		{
			name:                         "pod & container: decrease container cpu, decrease pod cpu, net decrease",
			newPodRequests:               v1.ResourceList{v1.ResourceCPU: resource.MustParse("2000m"), v1.ResourceMemory: resource.MustParse("2500Mi")}, // -500m
			newContRequests:              map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem1000M}},                            // -500m
			expected:                     false,
			inPlacePodLevelResizeEnabled: true,
		},
		{
			name:                         "pod & container: increase container cpu, same pod cpu, net neutral",
			newPodRequests:               v1.ResourceList{v1.ResourceCPU: resource.MustParse("2500m"), v1.ResourceMemory: resource.MustParse("2500Mi")}, // 0
			newContRequests:              map[int]v1.ResourceList{0: {v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1000M}},                           // +500m
			expected:                     false,
			inPlacePodLevelResizeEnabled: true,
		},
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, true)
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.inPlacePodLevelResizeEnabled == true {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)
			}

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
			if tc.inPlacePodLevelResizeEnabled == true {
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: cpu2500m, v1.ResourceMemory: mem2500M},
				}
			}

			allocationManager := makeAllocationManager(t, &containertest.FakeRuntime{}, []*v1.Pod{testPod}, nil)
			require.NoError(t, allocationManager.SetAllocatedResources(testPod))

			if tc.newPodRequests != nil {
				testPod.Spec.Resources.Requests = tc.newPodRequests
			}
			for k, v := range tc.newContRequests {
				testPod.Spec.Containers[k].Resources.Requests = v
			}
			require.Equal(t, tc.expected, allocationManager.(*manager).isResizeIncreasingRequests(testPod))
		})
	}
}

func TestSortPendingResizes(t *testing.T) {
	cpu500m := resource.MustParse("500m")
	cpu1000m := resource.MustParse("1")
	cpu1500m := resource.MustParse("1500m")
	mem500M := resource.MustParse("500Mi")
	mem1000M := resource.MustParse("1Gi")
	mem1500M := resource.MustParse("1500Mi")
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, true)
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

	testPods := []*v1.Pod{createTestPod(0), createTestPod(1), createTestPod(2), createTestPod(3), createTestPod(4), createTestPod(5), createTestPod(6)}
	allocationManager := makeAllocationManager(t, &containertest.FakeRuntime{}, testPods, nil)
	for _, testPod := range testPods {
		require.NoError(t, allocationManager.SetAllocatedResources(testPod))
	}

	// testPods[0] has the highest priority, as it doesn't increase resource requests (pod-level).
	// testPods[1] has the next highest priority, as it doesn't increase resource requests (container-level).
	// testPods[2] has the highest PriorityClass.
	// testPods[3] is the only pod with QoS class "guaranteed" (all others are burstable).
	// testPods[4] has been in a "deferred" state for longer than testPods[5].
	// testPods[6] has no resize conditions yet, indicating it is a newer request than the other deferred resizes.

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)

	testPods[0].Spec.Resources = &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M}}
	testPods[1].Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M}
	for i := 2; i < len(testPods); i++ {
		testPods[i].Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: cpu1500m, v1.ResourceMemory: mem1500M}
	}

	testPods[2].Spec.Priority = ptr.To(int32(100))
	testPods[3].Status.QOSClass = v1.PodQOSGuaranteed
	allocationManager.(*manager).statusManager.SetPodResizePendingCondition(testPods[4].UID, v1.PodReasonDeferred, "some-message", 1)
	time.Sleep(5 * time.Millisecond)
	allocationManager.(*manager).statusManager.SetPodResizePendingCondition(testPods[5].UID, v1.PodReasonDeferred, "some-message", 1)

	allocationManager.(*manager).getPodByUID = func(uid types.UID) (*v1.Pod, bool) {
		pods := make(map[types.UID]*v1.Pod)
		for _, pod := range testPods {
			pods[pod.UID] = pod
		}
		pod, found := pods[uid]
		return pod, found
	}

	expected := []types.UID{testPods[0].UID, testPods[1].UID, testPods[2].UID, testPods[3].UID, testPods[4].UID, testPods[5].UID, testPods[6].UID}

	// Push all the pods to the queue.
	for i := range testPods {
		allocationManager.PushPendingResize(testPods[i].UID)
	}
	require.Equal(t, expected, allocationManager.(*manager).podsWithPendingResizes)

	// Clear the queue and push the pods in reverse order to spice things up.
	allocationManager.(*manager).podsWithPendingResizes = nil
	for i := len(testPods) - 1; i >= 0; i-- {
		allocationManager.PushPendingResize(testPods[i].UID)
	}
	require.Equal(t, expected, allocationManager.(*manager).podsWithPendingResizes)
}

func TestRecordPodDeferredAcceptedResizes(t *testing.T) {
	if goruntime.GOOS == "windows" {
		t.Skip("InPlacePodVerticalScaling is not currently supported for Windows")
	}

	metrics.Register()
	metrics.PodDeferredAcceptedResizes.Reset()

	cpu500m := resource.MustParse("500m")
	cpu1000m := resource.MustParse("1")
	mem500M := resource.MustParse("500Mi")
	mem1000M := resource.MustParse("1Gi")

	for _, tc := range []struct {
		name                string
		trigger             string
		hasPendingCondition bool
		expectedMetrics     string
		expectedEvent       string
	}{
		{
			name:          "trigger reason: pod updated, no pending condition",
			trigger:       TriggerReasonPodUpdated,
			expectedEvent: `Normal ResizeStarted Pod resize started: {"containers":[{"name":"c1","resources":{"requests":{"cpu":"500m","memory":"500Mi"}}}],"generation":0}`,
		},
		{
			name:                "trigger reason: pod resized, pending condition",
			trigger:             TriggerReasonPodResized,
			hasPendingCondition: true,
			expectedMetrics: `
					# HELP kubelet_pod_deferred_accepted_resizes_total [ALPHA] Cumulative number of resizes that were accepted after being deferred.
					# TYPE kubelet_pod_deferred_accepted_resizes_total counter
					kubelet_pod_deferred_accepted_resizes_total{retry_trigger="pod_resized"} 1
			`,
			expectedEvent: `Normal ResizeStarted Pod resize started: {"containers":[{"name":"c1","resources":{"requests":{"cpu":"500m","memory":"500Mi"}}}],"generation":0}`,
		},
		{
			name:                "trigger reason: periodic retry, pending condition",
			trigger:             triggerReasonPeriodic,
			hasPendingCondition: true,
			expectedMetrics: `
					# HELP kubelet_pod_deferred_accepted_resizes_total [ALPHA] Cumulative number of resizes that were accepted after being deferred.
					# TYPE kubelet_pod_deferred_accepted_resizes_total counter
					kubelet_pod_deferred_accepted_resizes_total{retry_trigger="periodic_retry"} 1
					kubelet_pod_deferred_accepted_resizes_total{retry_trigger="pod_resized"} 1
			`,
			expectedEvent: `Normal ResizeStarted Pod resize started: {"containers":[{"name":"c1","resources":{"requests":{"cpu":"500m","memory":"500Mi"}}}],"generation":0}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			original := &v1.Pod{
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

			resizedPod := original.DeepCopy()
			resizedPod.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M}

			am := makeAllocationManager(t, &containertest.FakeRuntime{}, []*v1.Pod{original}, nil)
			require.NoError(t, am.SetAllocatedResources(original))
			if tc.hasPendingCondition {
				am.(*manager).statusManager.SetPodResizePendingCondition(original.UID, v1.PodReasonDeferred, "message", 1)
			}

			am.(*manager).getPodByUID = func(uid types.UID) (*v1.Pod, bool) {
				return resizedPod, true
			}
			am.PushPendingResize(original.UID)
			resizedPods := am.(*manager).retryPendingResizes(logger, tc.trigger)

			require.Len(t, resizedPods, 1)
			require.Equal(t, original.UID, resizedPods[0].UID)

			require.NoError(t, testutil.GatherAndCompare(
				legacyregistry.DefaultGatherer, strings.NewReader(tc.expectedMetrics), "kubelet_pod_deferred_accepted_resizes_total",
			))

			fakeRecorder := am.(*manager).recorder.(*record.FakeRecorder)
			require.Len(t, fakeRecorder.Events, 1)
			event := <-fakeRecorder.Events
			require.Equal(t, tc.expectedEvent, event)
		})
	}
}

func makeAllocationManager(t *testing.T, runtime *containertest.FakeRuntime, allocatedPods []*v1.Pod, nodeConfig *cm.NodeConfig) Manager {
	t.Helper()
	statusManager := status.NewManager(&fake.Clientset{}, kubepod.NewBasicPodManager(), &statustest.FakePodDeletionSafetyProvider{}, kubeletutil.NewPodStartupLatencyTracker())
	var containerManager *cm.FakeContainerManager
	if nodeConfig == nil {
		containerManager = cm.NewFakeContainerManager()
	} else {
		containerManager = cm.NewFakeContainerManagerWithNodeConfig(*nodeConfig)
	}
	allocationManager := NewInMemoryManager(
		containerManager.GetNodeConfig(),
		containerManager.GetNodeAllocatableAbsolute(),
		statusManager,
		func(pod *v1.Pod) {
			/* For testing, just mark the pod as having a pod sync triggered in an annotation. */
			if pod.Annotations == nil {
				pod.Annotations = make(map[string]string)
			}
			pod.Annotations["pod-sync-triggered"] = "true"
		},
		func() []*v1.Pod { return allocatedPods },
		func(uid types.UID) (*v1.Pod, bool) {
			for _, p := range allocatedPods {
				if p.UID == uid {
					return p, true
				}
			}
			return nil, false
		},
		config.NewSourcesReady(func(_ sets.Set[string]) bool { return true }),
		record.NewFakeRecorder(20),
	)
	allocationManager.SetContainerRuntime(runtime)

	getNode := func(context.Context, bool) (*v1.Node, error) {
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
	handler := lifecycle.NewPredicateAdmitHandler(getNode, lifecycle.NewAdmissionFailureHandlerStub(), containerManager.UpdatePluginResources)
	allocationManager.AddPodAdmitHandlers(lifecycle.PodAdmitHandlers{handler})

	return allocationManager
}

func setContainerStatus(podStatus *kubecontainer.PodStatus, c *v1.Container, idx int) {
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
