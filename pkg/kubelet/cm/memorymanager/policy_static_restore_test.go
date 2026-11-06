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

type interruptedCheckpointState struct {
	state.State
	after string
}

const simulatedProcessInterruption = "simulated process interruption after checkpoint write"

func (s *interruptedCheckpointState) interruptAfter(method string) {
	if s.after == method {
		panic(simulatedProcessInterruption)
	}
}

func (s *interruptedCheckpointState) SetPodMemoryBlocks(podUID string, blocks []state.Block) {
	s.State.SetPodMemoryBlocks(podUID, blocks)
	s.interruptAfter("SetPodMemoryBlocks")
}

func (s *interruptedCheckpointState) SetMemoryBlocks(podUID, containerName string, blocks []state.Block) {
	s.State.SetMemoryBlocks(podUID, containerName, blocks)
	s.interruptAfter("SetMemoryBlocks")
}

func (s *interruptedCheckpointState) Delete(podUID, containerName string) {
	s.State.Delete(podUID, containerName)
	s.interruptAfter("Delete")
}

func (s *interruptedCheckpointState) SetMachineState(machineState state.NUMANodeMap) {
	s.State.SetMachineState(machineState)
	s.interruptAfter("SetMachineState")
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

func TestMemoryManagerRestoreInterruptedPodCheckpoint(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Memory Manager static policy is not available on Windows")
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)

	for _, tc := range []struct {
		name           string
		after          string
		release        bool
		corruptMachine bool
		wantError      string
	}{
		{
			name:  "allocation after pod checkpoint",
			after: "SetPodMemoryBlocks",
		},
		{
			name:    "release after machine checkpoint before pod deletion",
			after:   "SetMachineState",
			release: true,
		},
		{
			name:      "allocation after first container checkpoint",
			after:     "SetMemoryBlocks",
			wantError: "[memorymanager] the expected machine state is different from the real one",
		},
		{
			name:      "release after last container deletion before memory release",
			after:     "Delete",
			release:   true,
			wantError: "[memorymanager] the expected machine state is different from the real one",
		},
		{
			name:           "pod-only entry with unrelated machine inconsistency",
			after:          "SetPodMemoryBlocks",
			corruptMachine: true,
			wantError:      "[memorymanager] the expected machine state is different from the real one",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			logger := tCtx.Logger()
			machineInfo := returnMachineInfo()
			nodeAllocatableReservation := v1.ResourceList{
				v1.ResourceMemory: *resource.NewQuantity(2*gb, resource.BinarySI),
			}
			systemReservedMemory := []kubeletconfig.MemoryReservation{
				{NumaNode: 0, Limits: v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(gb, resource.BinarySI)}},
				{NumaNode: 1, Limits: v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(gb, resource.BinarySI)}},
			}
			pod := getPodWithContainersAndPodLevelResources("interrupted", "128Mi", "128Mi", nil, []containerSpec{
				{name: "exclusive", memRequest: "100Mi", memLimit: "100Mi"},
				{name: "shared"},
			})
			backgroundPod := getPodWithContainersAndPodLevelResources("background", "128Mi", "128Mi", nil, []containerSpec{
				{name: "container", memRequest: "128Mi", memLimit: "128Mi"},
			})
			pods := []*v1.Pod{backgroundPod, pod}
			activePods := func() []*v1.Pod { return pods }
			stateDir := t.TempDir()
			newManagers := func() (Manager, topologymanager.Manager) {
				t.Helper()
				topologyMgr, err := topologymanager.NewManager(logger, machineInfo.Topology, topologymanager.PolicyBestEffort, topologymanager.PodTopologyScope, nil)
				if err != nil {
					t.Fatalf("could not create topology manager: %v", err)
				}
				mgr, err := NewManager(logger, string(PolicyTypeStatic), &machineInfo, nodeAllocatableReservation, systemReservedMemory, stateDir, topologyMgr)
				if err != nil {
					t.Fatalf("could not create memory manager: %v", err)
				}
				topologyMgr.AddHintProvider(logger, mgr)
				return mgr, topologyMgr
			}
			start := func(mgr Manager) error {
				return mgr.Start(tCtx, activePods, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
			}
			checkState := func(s state.Reader, wantPods state.PodMemoryAssignments, wantContainers state.ContainerMemoryAssignments, wantMachine state.NUMANodeMap) {
				t.Helper()
				if diff := cmp.Diff(wantPods, s.GetPodMemoryAssignments()); diff != "" {
					t.Errorf("unexpected pod assignments (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(wantContainers, s.GetMemoryAssignments()); diff != "" {
					t.Errorf("unexpected container assignments (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(wantMachine, s.GetMachineState()); diff != "" {
					t.Errorf("unexpected machine state (-want +got):\n%s", diff)
				}
			}

			mgr, topologyMgr := newManagers()
			if err := start(mgr); err != nil {
				t.Fatalf("could not start manager: %v", err)
			}
			if result := topologyMgr.Admit(tCtx, &lifecycle.PodAdmitAttributes{Pod: backgroundPod, Operation: lifecycle.AddOperation}); !result.Admit {
				t.Fatalf("could not admit background pod: %+v", result)
			}
			// Keep another allocation on the same NUMA node to verify that recovery
			// preserves existing reservations rather than resetting machine state.
			baselinePods := mgr.State().GetPodMemoryAssignments()
			baselineContainers := mgr.State().GetMemoryAssignments()
			baselineMachine := mgr.State().GetMachineState()
			if node := baselineMachine[0]; node.NumberOfAssignments != 1 || node.MemoryMap[v1.ResourceMemory].Reserved != 128*mb {
				t.Fatalf("unexpected background reservation: %+v", node)
			}
			wantPods := baselinePods.Clone()
			wantPods[string(pod.UID)] = state.PodEntry{MemoryBlocks: []state.Block{
				{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 128 * mb},
			}}
			wantContainers := baselineContainers.Clone()
			wantContainers[string(pod.UID)] = map[string][]state.Block{
				"exclusive": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 100 * mb}},
				"shared":    {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 28 * mb}},
			}
			wantMachine := baselineMachine.Clone()
			wantMachine[0].NumberOfAssignments++
			wantMachine[0].MemoryMap[v1.ResourceMemory].Reserved += 128 * mb
			wantMachine[0].MemoryMap[v1.ResourceMemory].Free -= 128 * mb

			if tc.release {
				if result := topologyMgr.Admit(tCtx, &lifecycle.PodAdmitAttributes{Pod: pod, Operation: lifecycle.AddOperation}); !result.Admit {
					t.Fatalf("could not admit pod before release: %+v", result)
				}
				checkState(mgr.State(), wantPods, wantContainers, wantMachine)
				for i := range pod.Spec.Containers {
					container := &pod.Spec.Containers[i]
					mgr.AddContainer(logger, pod, container, container.Name)
				}
				if err := mgr.RemoveContainer(logger, "exclusive"); err != nil {
					t.Fatalf("could not remove first container: %v", err)
				}
			}

			managerImpl := mgr.(*manager)
			managerImpl.state = &interruptedCheckpointState{State: managerImpl.state, after: tc.after}
			func() {
				defer func() {
					if recovered := recover(); recovered != simulatedProcessInterruption {
						t.Fatalf("expected simulated process interruption, got %v", recovered)
					}
				}()
				if tc.release {
					if err := mgr.RemoveContainer(logger, "shared"); err != nil {
						t.Fatalf("could not remove last container: %v", err)
					}
				} else {
					topologyMgr.Admit(tCtx, &lifecycle.PodAdmitAttributes{Pod: pod, Operation: lifecycle.AddOperation})
				}
			}()

			wantInterruptedContainers := baselineContainers.Clone()
			if tc.after == "SetMemoryBlocks" {
				wantInterruptedContainers[string(pod.UID)] = map[string][]state.Block{
					"exclusive": wantContainers[string(pod.UID)]["exclusive"],
				}
			}
			wantInterruptedMachine := baselineMachine.Clone()
			if tc.release && tc.after == "Delete" {
				wantInterruptedMachine = wantMachine.Clone()
			}
			if tc.corruptMachine {
				wantInterruptedMachine[0].MemoryMap[v1.ResourceMemory].Reserved++
				wantInterruptedMachine[0].MemoryMap[v1.ResourceMemory].Free--
				managerImpl.state.(*interruptedCheckpointState).State.SetMachineState(wantInterruptedMachine)
			}
			checkState(mgr.State(), wantPods, wantInterruptedContainers, wantInterruptedMachine)
			if tc.release {
				pods = []*v1.Pod{backgroundPod}
			}

			mgr2, _ := newManagers()
			err := start(mgr2)
			if tc.wantError != "" {
				if err == nil || err.Error() != tc.wantError {
					t.Fatalf("expected restore error %q, got %v", tc.wantError, err)
				}
				// Failed validation must preserve the checkpoint for diagnosis.
				checkState(mgr2.State(), wantPods, wantInterruptedContainers, wantInterruptedMachine)
			} else {
				if err != nil {
					t.Fatalf("expected successful restore, got %v", err)
				}
				checkState(mgr2.State(), baselinePods, baselineContainers, baselineMachine)
			}
			persisted, err := state.NewCheckpointState(logger, stateDir, memoryManagerStateFileName, string(PolicyTypeStatic))
			if err != nil {
				t.Fatalf("could not reload checkpoint: %v", err)
			}
			if tc.wantError != "" {
				checkState(persisted, wantPods, wantInterruptedContainers, wantInterruptedMachine)
				return
			}
			checkState(persisted, baselinePods, baselineContainers, baselineMachine)

			// Recovery is durable and idempotent even if kubelet restarts again
			// before an interrupted allocation is retried.
			mgr3, topologyMgr3 := newManagers()
			if err := start(mgr3); err != nil {
				t.Fatalf("could not restart recovered manager: %v", err)
			}
			checkState(mgr3.State(), baselinePods, baselineContainers, baselineMachine)
			if tc.release {
				return
			}
			if result := topologyMgr3.Admit(tCtx, &lifecycle.PodAdmitAttributes{Pod: pod, Operation: lifecycle.AddOperation}); !result.Admit {
				t.Fatalf("expected successful admission after recovery, got %+v", result)
			}
			for _, container := range pod.Spec.Containers {
				affinity := topologyMgr3.GetAffinity(logger, string(pod.UID), container.Name)
				if !affinity.Preferred || !affinity.NUMANodeAffinity.IsEqual(newNUMAAffinity(0)) {
					t.Errorf("expected preferred affinity for NUMA node 0 for %q, got %v", container.Name, affinity)
				}
			}
			checkState(mgr3.State(), wantPods, wantContainers, wantMachine)
			if result := topologyMgr3.Admit(tCtx, &lifecycle.PodAdmitAttributes{Pod: pod, Operation: lifecycle.AddOperation}); !result.Admit {
				t.Fatalf("expected successful repeated admission, got %+v", result)
			}
			checkState(mgr3.State(), wantPods, wantContainers, wantMachine)
		})
	}
}

func TestMemoryManagerRestorePodOnlyCheckpointWithoutMachineState(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	logger, _ := ktesting.NewTestContext(t)
	machineInfo := returnMachineInfo()
	policy, err := NewPolicyStatic(logger, &machineInfo, systemReservedMemory{
		0: {v1.ResourceMemory: gb},
		1: {v1.ResourceMemory: gb},
	}, topologymanager.NewFakeManager(logger))
	if err != nil {
		t.Fatalf("could not create policy: %v", err)
	}
	stateDir := t.TempDir()
	s, err := state.NewCheckpointState(logger, stateDir, memoryManagerStateFileName, string(PolicyTypeStatic))
	if err != nil {
		t.Fatalf("could not create checkpoint: %v", err)
	}
	wantPods := state.PodMemoryAssignments{
		"pod": {MemoryBlocks: []state.Block{{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 128 * mb}}},
	}
	s.SetPodMemoryAssignments(wantPods)
	wantError := "[memorymanager] machine state can not be empty when it has memory assignments"
	if err := policy.Start(logger, s); err == nil || err.Error() != wantError {
		t.Fatalf("expected restore error %q, got %v", wantError, err)
	}
	persisted, err := state.NewCheckpointState(logger, stateDir, memoryManagerStateFileName, string(PolicyTypeStatic))
	if err != nil {
		t.Fatalf("could not reload checkpoint: %v", err)
	}
	for _, reader := range []state.Reader{s, persisted} {
		if diff := cmp.Diff(wantPods, reader.GetPodMemoryAssignments()); diff != "" {
			t.Errorf("pod assignments changed after failed validation (-want +got):\n%s", diff)
		}
		if machine := reader.GetMachineState(); len(machine) != 0 {
			t.Errorf("machine state changed after failed validation: %v", machine)
		}
	}
}
