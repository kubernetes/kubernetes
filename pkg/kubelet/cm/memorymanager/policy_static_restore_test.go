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

package memorymanager

import (
	"runtime"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// For the scope of the test, any pod that has pod-level resources and the
// PodLevelResourceManagers feature is enabled, will be processed by AllocatePod
func TestMemoryManagerRestoreState(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Memory Manager static policy is not available on Windows")
	}

	tCtx := ktesting.Init(t)

	testCases := []struct {
		description                     string
		podLevelResourcesEnabled        bool
		podLevelResourceManagersEnabled bool
		podMemoryRequest                string
		containers                      []containerSpec
		expectPodBlocks                 bool
	}{
		{
			description:                     "PodLevelResources and PodLevelResourceManagers enabled",
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			podMemoryRequest:                "128Mi",
			containers: []containerSpec{
				{name: "container1", memRequest: "100Mi", memLimit: "100Mi"},
			},
			expectPodBlocks: true,
		},
		{
			description:                     "PodLevelResources enabled, PodLevelResourceManagers disabled",
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: false,
			podMemoryRequest:                "128Mi",
			containers: []containerSpec{
				{name: "container1", memRequest: "100Mi", memLimit: "100Mi"},
			},
			expectPodBlocks: false,
		},
		{
			description:                     "Container-level pod, features enabled",
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			podMemoryRequest:                "",
			containers: []containerSpec{
				{name: "container1", memRequest: "100Mi", memLimit: "100Mi"},
				{name: "container2", memRequest: "100Mi", memLimit: "100Mi"},
			},
			expectPodBlocks: false,
		},
		{
			description:                     "Container-level pod, features disabled",
			podLevelResourcesEnabled:        false,
			podLevelResourceManagersEnabled: false,
			podMemoryRequest:                "",
			containers: []containerSpec{
				{name: "container1", memRequest: "100Mi", memLimit: "100Mi"},
				{name: "container2", memRequest: "100Mi", memLimit: "100Mi"},
			},
			expectPodBlocks: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, tc.podLevelResourcesEnabled)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, tc.podLevelResourceManagersEnabled)

			logger, ctx := ktesting.NewTestContext(t)
			machineInfo := returnMachineInfo()
			nodeAllocatableReservation := v1.ResourceList{
				v1.ResourceMemory: *resource.NewQuantity(2*gb, resource.BinarySI),
			}
			systemReservedMemory := []kubeletconfig.MemoryReservation{
				{
					NumaNode: 0,
					Limits: v1.ResourceList{
						v1.ResourceMemory: *resource.NewQuantity(gb, resource.BinarySI),
					},
				},
				{
					NumaNode: 1,
					Limits: v1.ResourceList{
						v1.ResourceMemory: *resource.NewQuantity(gb, resource.BinarySI),
					},
				},
			}
			affinity := topologymanager.NewFakeManager(logger)

			// Create new manager
			sDir := t.TempDir()
			mgr, err := NewManager(logger, string(PolicyTypeStatic), &machineInfo, nodeAllocatableReservation, systemReservedMemory, sDir, affinity)
			if err != nil {
				t.Fatalf("could not create manager: %v", err)
			}

			// Create a pod with pod-level resources
			pod := getPodWithContainersAndPodLevelResources("pod1", tc.podMemoryRequest, tc.podMemoryRequest, nil, tc.containers)

			// Start manager to initialize state
			err = mgr.Start(tCtx, func() []*v1.Pod { return []*v1.Pod{pod} }, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
			if err != nil {
				t.Fatalf("could not start manager: %v", err)
			}

			// Allocate resources
			if tc.podLevelResourceManagersEnabled && resourcehelper.IsPodLevelResourcesSet(pod) {
				err = mgr.AllocatePod(logger, pod, lifecycle.AddOperation)
				if err != nil {
					t.Fatalf("could not allocate pod: %v", err)
				}
			} else {
				// Add containers (allocates exclusive resources from the pod pool)
				for i := range pod.Spec.Containers {
					container := &pod.Spec.Containers[i]
					err = mgr.Allocate(ctx, pod, container, lifecycle.AddOperation)
					if err != nil {
						t.Fatalf("could not allocate container %s: %v", container.Name, err)
					}
					mgr.AddContainer(logger, pod, container, container.Name)
				}
			}

			// Verify state before restart
			podBlocks := mgr.State().GetPodMemoryBlocks(string(pod.UID))
			if tc.expectPodBlocks && len(podBlocks) == 0 {
				t.Errorf("expected pod memory blocks to be present")
			} else if !tc.expectPodBlocks && len(podBlocks) > 0 {
				t.Errorf("expected no pod memory blocks, but got some")
			}

			// Re-create manager to simulate restart
			mgr2, err := NewManager(logger, string(PolicyTypeStatic), &machineInfo, nodeAllocatableReservation, systemReservedMemory, sDir, affinity)
			if err != nil {
				t.Fatalf("could not create manager 2: %v", err)
			}

			err = mgr2.Start(tCtx, func() []*v1.Pod { return []*v1.Pod{pod} }, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
			if err != nil {
				t.Fatalf("could not start manager 2: %v", err)
			}

			// Verify state restored
			podBlocksRestored := mgr2.State().GetPodMemoryBlocks(string(pod.UID))
			if tc.expectPodBlocks {
				if len(podBlocksRestored) == 0 {
					t.Errorf("expected pod memory blocks to be present after restore")
				}
				if len(podBlocksRestored) != len(podBlocks) {
					t.Errorf("expected pod memory blocks count to match, got %d want %d", len(podBlocksRestored), len(podBlocks))
				}
			} else if len(podBlocksRestored) > 0 {
				t.Errorf("expected no pod memory blocks after restore, but got some")
			}

			// Verify containers restored
			for _, container := range pod.Spec.Containers {
				containerBlocksRestored := mgr2.State().GetMemoryBlocks(string(pod.UID), container.Name)
				// If pod-level resources are enabled but managers are disabled, allocation is skipped, so no blocks.
				if tc.podLevelResourcesEnabled && !tc.podLevelResourceManagersEnabled {
					if len(containerBlocksRestored) > 0 {
						t.Errorf("expected no container memory blocks after restore (allocation skipped) for %s, but got some", container.Name)
					}
				} else {
					if len(containerBlocksRestored) == 0 {
						t.Errorf("expected container memory blocks to be present after restore for %s", container.Name)
					}
				}
			}
		})
	}
}

// TestMemoryManagerRestoreStateWithMemoryDrift simulates a reboot after which
// the total memory reported for each NUMA node changes by a few MiB, because
// the kernel frees a varying amount of boot-time memory (see
// https://issue.k8s.io/131253). The manager must still start and restore the
// recorded assignment instead of failing with "the expected machine state is
// different from the real one".
func TestMemoryManagerRestoreStateWithMemoryDrift(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Memory Manager static policy is not available on Windows")
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, false)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, false)

	tCtx := ktesting.Init(t)
	logger, ctx := ktesting.NewTestContext(t)

	nodeAllocatableReservation := v1.ResourceList{
		v1.ResourceMemory: *resource.NewQuantity(2*gb, resource.BinarySI),
	}
	systemReservedMemory := []kubeletconfig.MemoryReservation{
		{NumaNode: 0, Limits: v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(gb, resource.BinarySI)}},
		{NumaNode: 1, Limits: v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(gb, resource.BinarySI)}},
	}
	affinity := topologymanager.NewFakeManager(logger)
	sDir := t.TempDir()
	pod := getPodWithContainersAndPodLevelResources("pod1", "", "", nil, []containerSpec{
		{name: "container1", memRequest: "100Mi", memLimit: "100Mi"},
	})
	activePods := func() []*v1.Pod { return []*v1.Pod{pod} }

	// First boot: initialize the state and allocate the pod so the state file
	// holds a real assignment.
	machineInfo := returnMachineInfo()
	mgr, err := NewManager(logger, string(PolicyTypeStatic), &machineInfo, nodeAllocatableReservation, systemReservedMemory, sDir, affinity)
	if err != nil {
		t.Fatalf("could not create manager: %v", err)
	}
	if err := mgr.Start(tCtx, activePods, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap()); err != nil {
		t.Fatalf("could not start manager: %v", err)
	}
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		if err := mgr.Allocate(ctx, pod, container, lifecycle.AddOperation); err != nil {
			t.Fatalf("could not allocate container %s: %v", container.Name, err)
		}
		mgr.AddContainer(logger, pod, container, container.Name)
	}
	blocks := mgr.State().GetMemoryBlocks(string(pod.UID), "container1")
	if len(blocks) == 0 {
		t.Fatalf("expected memory blocks after allocation")
	}

	// Reboot: cAdvisor reports a slightly smaller total for each NUMA node.
	rebootedMachineInfo := returnMachineInfo()
	rebootedMachineInfo.Topology[0].Memory -= 4 * mb
	rebootedMachineInfo.Topology[1].Memory -= 4 * mb

	mgr2, err := NewManager(logger, string(PolicyTypeStatic), &rebootedMachineInfo, nodeAllocatableReservation, systemReservedMemory, sDir, affinity)
	if err != nil {
		t.Fatalf("could not create manager after reboot: %v", err)
	}
	// Before the fix this returns "the expected machine state is different from
	// the real one" and the node stays NotReady.
	if err := mgr2.Start(tCtx, activePods, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap()); err != nil {
		t.Fatalf("manager should start after a small NUMA node memory drift: %v", err)
	}
	if restored := mgr2.State().GetMemoryBlocks(string(pod.UID), "container1"); len(restored) != len(blocks) {
		t.Fatalf("expected %d restored memory blocks after reboot, got %d", len(blocks), len(restored))
	}
}
