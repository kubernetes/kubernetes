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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type crashAfterPodCheckpointState struct {
	state.State
}

const simulatedProcessInterruption = "simulated process interruption after pod allocation checkpoint"

func (s *crashAfterPodCheckpointState) SetPodMemoryBlocks(podUID string, blocks []state.Block) {
	s.State.SetPodMemoryBlocks(podUID, blocks)
	panic(simulatedProcessInterruption)
}

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
		allocationAffinity              []int
		expectedAffinity                []int
	}{
		{
			description:                     "PodLevelResources and PodLevelResourceManagers enabled",
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			podMemoryRequest:                "128Mi",
			containers: []containerSpec{
				{name: "container1", memRequest: "100Mi", memLimit: "100Mi"},
			},
			expectPodBlocks:  true,
			expectedAffinity: []int{0},
		},
		{
			description:                     "Pod topology hint is restored from a non-zero NUMA node",
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			podMemoryRequest:                "128Mi",
			containers: []containerSpec{
				{name: "container1", memRequest: "100Mi", memLimit: "100Mi"},
			},
			expectPodBlocks:    true,
			allocationAffinity: []int{1},
			expectedAffinity:   []int{1},
		},
		{
			description:                     "Pod topology hint is restored from a multi-container pod allocation",
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			podMemoryRequest:                "128Mi",
			containers: []containerSpec{
				{name: "container1", memRequest: "50Mi", memLimit: "50Mi"},
				{name: "container2", memRequest: "50Mi", memLimit: "50Mi"},
			},
			expectPodBlocks:  true,
			expectedAffinity: []int{0},
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
			expectPodBlocks:  false,
			expectedAffinity: []int{0},
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
			expectPodBlocks:  false,
			expectedAffinity: []int{0},
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
			if tc.allocationAffinity != nil {
				affinity = topologymanager.NewFakeManagerWithHint(logger, &topologymanager.TopologyHint{
					NUMANodeAffinity: newNUMAAffinity(tc.allocationAffinity...),
					Preferred:        true,
				})
			}

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
			podMemoryAssignments := mgr.State().GetPodMemoryAssignments()
			memoryAssignments := mgr.State().GetMemoryAssignments()
			machineState := mgr.State().GetMachineState()
			podBlocks := mgr.State().GetPodMemoryBlocks(string(pod.UID))
			if tc.expectPodBlocks && len(podBlocks) == 0 {
				t.Errorf("expected pod memory blocks to be present")
			} else if !tc.expectPodBlocks && len(podBlocks) > 0 {
				t.Errorf("expected no pod memory blocks, but got some")
			}

			// Re-create manager to simulate restart
			restoredAffinity := topologymanager.NewFakeManager(logger)
			mgr2, err := NewManager(logger, string(PolicyTypeStatic), &machineInfo, nodeAllocatableReservation, systemReservedMemory, sDir, restoredAffinity)
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

			hints := mgr2.GetPodTopologyHints(logger, pod, lifecycle.AddOperation)
			memoryHints := hints[string(v1.ResourceMemory)]
			if tc.expectedAffinity == nil {
				if len(memoryHints) != 0 {
					t.Fatalf("expected no restored memory hint, got %v", memoryHints)
				}
			} else {
				if len(memoryHints) != 1 {
					t.Fatalf("expected one restored memory hint, got %v", memoryHints)
				}
				if !memoryHints[0].Preferred {
					t.Error("expected restored memory hint to be preferred")
				}
				expectedAffinity := newNUMAAffinity(tc.expectedAffinity...)
				if !memoryHints[0].NUMANodeAffinity.IsEqual(expectedAffinity) {
					t.Errorf("expected restored memory hint affinity %v, got %v", expectedAffinity, memoryHints[0].NUMANodeAffinity)
				}
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

			if tc.podLevelResourceManagersEnabled && resourcehelper.IsPodLevelResourcesSet(pod) {
				if err := mgr2.AllocatePod(logger, pod, lifecycle.AddOperation); err != nil {
					t.Fatalf("could not allocate restored pod: %v", err)
				}

				if diff := cmp.Diff(podMemoryAssignments, mgr2.State().GetPodMemoryAssignments()); diff != "" {
					t.Errorf("pod memory assignments changed after allocating restored pod (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(memoryAssignments, mgr2.State().GetMemoryAssignments()); diff != "" {
					t.Errorf("container memory assignments changed after allocating restored pod (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(machineState, mgr2.State().GetMachineState()); diff != "" {
					t.Errorf("machine state changed after allocating restored pod (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestMemoryManagerRestorePartialPodCheckpoint(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Memory Manager static policy is not available on Windows")
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)

	tCtx := ktesting.Init(t)
	logger := tCtx.Logger()
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
	pod := getPodWithContainersAndPodLevelResources("pod1", "128Mi", "128Mi", nil, []containerSpec{
		{name: "container1", memRequest: "100Mi", memLimit: "100Mi"},
	})
	pod.UID = "podUID"
	activePods := func() []*v1.Pod { return []*v1.Pod{pod} }

	stateDir := t.TempDir()
	topologyMgr, err := topologymanager.NewManager(logger, machineInfo.Topology, topologymanager.PolicyBestEffort, topologymanager.PodTopologyScope, nil)
	if err != nil {
		t.Fatalf("could not create topology manager: %v", err)
	}
	mgr, err := NewManager(logger, string(PolicyTypeStatic), &machineInfo, nodeAllocatableReservation, systemReservedMemory, stateDir, topologyMgr)
	if err != nil {
		t.Fatalf("could not create manager: %v", err)
	}
	topologyMgr.AddHintProvider(logger, mgr)
	if err := mgr.Start(tCtx, activePods, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap()); err != nil {
		t.Fatalf("could not start manager: %v", err)
	}

	preAllocationMachineState := mgr.State().GetMachineState()
	managerImpl := mgr.(*manager)
	func() {
		defer func() {
			if recovered := recover(); recovered == nil {
				t.Fatal("expected simulated process interruption")
			} else if recovered != simulatedProcessInterruption {
				t.Fatalf("unexpected panic: %v", recovered)
			}
		}()
		managerImpl.state = &crashAfterPodCheckpointState{State: managerImpl.state}
		topologyMgr.Admit(tCtx, &lifecycle.PodAdmitAttributes{Pod: pod, Operation: lifecycle.AddOperation})
	}()

	if blocks := mgr.State().GetPodMemoryBlocks(string(pod.UID)); len(blocks) == 0 {
		t.Fatal("expected pod memory blocks to be persisted before interruption")
	}
	if assignments := mgr.State().GetMemoryAssignments(); len(assignments) != 0 {
		t.Fatalf("expected no container memory assignments before interruption, got %v", assignments)
	}
	if diff := cmp.Diff(preAllocationMachineState, mgr.State().GetMachineState()); diff != "" {
		t.Fatalf("machine state changed before interruption (-want +got):\n%s", diff)
	}

	topologyMgr2, err := topologymanager.NewManager(logger, machineInfo.Topology, topologymanager.PolicyBestEffort, topologymanager.PodTopologyScope, nil)
	if err != nil {
		t.Fatalf("could not create restored topology manager: %v", err)
	}
	mgr2, err := NewManager(logger, string(PolicyTypeStatic), &machineInfo, nodeAllocatableReservation, systemReservedMemory, stateDir, topologyMgr2)
	if err != nil {
		t.Fatalf("could not create restored manager: %v", err)
	}
	topologyMgr2.AddHintProvider(logger, mgr2)
	if err := mgr2.Start(tCtx, activePods, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap()); err != nil {
		// Rejecting an incomplete checkpoint is a valid recovery outcome.
		if !strings.Contains(err.Error(), "has a pod memory assignment but no container memory assignments") {
			t.Fatalf("unexpected restore error: %v", err)
		}
		return
	}

	result := topologyMgr2.Admit(tCtx, &lifecycle.PodAdmitAttributes{Pod: pod, Operation: lifecycle.AddOperation})
	if !result.Admit {
		// Failing allocation instead of accepting incomplete state is also safe.
		return
	}
	affinity := topologyMgr2.GetAffinity(logger, string(pod.UID), pod.Spec.Containers[0].Name)
	if !affinity.Preferred || !affinity.NUMANodeAffinity.IsEqual(newNUMAAffinity(0)) {
		t.Fatalf("expected restored pod hint for NUMA node 0, got %v", affinity)
	}

	for _, container := range pod.Spec.Containers {
		if blocks := mgr2.State().GetMemoryBlocks(string(pod.UID), container.Name); len(blocks) == 0 {
			t.Errorf("successful restored pod allocation left container %q without memory blocks", container.Name)
		}
	}
	if diff := cmp.Diff(preAllocationMachineState, mgr2.State().GetMachineState()); diff == "" {
		t.Error("successful restored pod allocation left machine state without the pod reservation")
	}
}
