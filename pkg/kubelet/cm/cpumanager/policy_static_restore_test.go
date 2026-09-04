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

package cpumanager

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
	"time"

	cadvisorapi "github.com/google/cadvisor/lib/model"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/cpuset"
)

// For the scope of the test, any pod that has pod-level resources and the
// PodLevelResourceManagers feature is enabled, will be processed by AllocatePod
func TestCPUManagerRestoreState(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("CPU Manager static policy is not available on Windows")
	}

	testCases := []struct {
		description                     string
		podLevelResourcesEnabled        bool
		podLevelResourceManagersEnabled bool
		podCPURequest                   string
		containers                      []*containerOptions
		expectPodCPUSet                 bool
	}{
		{
			description:   "Pod-level resources and managers enabled",
			podCPURequest: "2",
			containers: []*containerOptions{
				{name: "container1", request: "1", limit: "1"},
			},
			expectPodCPUSet:                 true,
			podLevelResourceManagersEnabled: true,
			podLevelResourcesEnabled:        true,
		},
		{
			description:   "Pod-level resources enabled, managers disabled",
			podCPURequest: "2",
			containers: []*containerOptions{
				{name: "container1", request: "1", limit: "1"},
			},
			expectPodCPUSet:                 false,
			podLevelResourceManagersEnabled: false,
			podLevelResourcesEnabled:        true,
		},
		{
			description:   "Container-level pod, features enabled",
			podCPURequest: "",
			containers: []*containerOptions{
				{name: "container1", request: "1", limit: "1"},
				{name: "container2", request: "1", limit: "1"},
			},
			expectPodCPUSet:                 false,
			podLevelResourceManagersEnabled: true,
			podLevelResourcesEnabled:        true,
		},
		{
			description:   "Container-level pod, features disabled",
			podCPURequest: "",
			containers: []*containerOptions{
				{name: "container1", request: "1", limit: "1"},
				{name: "container2", request: "1", limit: "1"},
			},
			expectPodCPUSet:                 false,
			podLevelResourceManagersEnabled: false,
			podLevelResourcesEnabled:        false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, tc.podLevelResourcesEnabled)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, tc.podLevelResourceManagersEnabled)

			sDir := t.TempDir()
			logger, tCtx := ktesting.NewTestContext(t)
			mgr, err := newRestoreTestManager(logger, 4, sDir)
			if err != nil {
				t.Fatalf("could not create manager: %v", err)
			}

			// Create a pod
			var pod *v1.Pod
			if tc.podCPURequest != "" {
				pod = makeMultiContainerPodWithOptionsAndPodLevelResources(tc.podCPURequest, nil, tc.containers)
			} else {
				pod = makeMultiContainerPodWithOptions(nil, tc.containers)
			}
			pod.Name = "pod1"
			pod.UID = types.UID("pod1")

			// Start manager to initialize state and activePods
			err = mgr.Start(tCtx, func() []*v1.Pod { return []*v1.Pod{pod} }, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
			if err != nil {
				t.Fatalf("could not start manager: %v", err)
			}

			// Allocate resources (Pod Scope)
			if tc.podLevelResourceManagersEnabled && resourcehelper.IsPodLevelResourcesSet(pod) {
				err = mgr.AllocatePod(logger, pod, lifecycle.AddOperation)
				if err != nil {
					t.Fatalf("could not allocate pod: %v", err)
				}
			} else {
				// Allocate resources (Container Scope / Legacy)
				for i := range pod.Spec.Containers {
					container := &pod.Spec.Containers[i]
					err = mgr.Allocate(tCtx, pod, container, lifecycle.AddOperation)
					if err != nil {
						t.Fatalf("could not allocate container %s: %v", container.Name, err)
					}
				}
			}

			// Add containers (simulate running)
			for i := range pod.Spec.Containers {
				container := &pod.Spec.Containers[i]
				mgr.AddContainer(logger, pod, container, container.Name)
			}

			// Verify state before restart
			podCPUSet, _ := mgr.State().GetPodCPUSet(string(pod.UID))
			if tc.expectPodCPUSet && podCPUSet.IsEmpty() {
				t.Errorf("expected pod cpu set to be present")
			} else if !tc.expectPodCPUSet && !podCPUSet.IsEmpty() {
				t.Errorf("expected no pod cpu set, but got some")
			}

			// Re-create manager to simulate restart
			mgr2, err := newRestoreTestManager(logger, 4, sDir)
			if err != nil {
				t.Fatalf("could not create manager 2: %v", err)
			}

			err = mgr2.Start(tCtx, func() []*v1.Pod { return []*v1.Pod{pod} }, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
			if err != nil {
				t.Fatalf("could not start manager 2: %v", err)
			}

			// Verify state restored
			podCPUSetRestored, _ := mgr2.State().GetPodCPUSet(string(pod.UID))
			if tc.expectPodCPUSet {
				if podCPUSetRestored.IsEmpty() {
					t.Errorf("expected pod cpu set to be present after restore")
				}
				if !podCPUSetRestored.Equals(podCPUSet) {
					t.Errorf("expected pod cpu set to be %q, got %q", podCPUSet, podCPUSetRestored)
				}
			} else if !podCPUSetRestored.IsEmpty() {
				t.Errorf("expected no pod cpu set after restore, but got some")
			}

			for i := range pod.Spec.Containers {
				container := &pod.Spec.Containers[i]
				containerCPUSet, ok := mgr2.State().GetCPUSet(string(pod.UID), container.Name)
				// If allocation was skipped (PLR enabled but Manager disabled), no cpuset.
				allocationSkipped := tc.podLevelResourcesEnabled && !tc.podLevelResourceManagersEnabled && resourcehelper.IsPodLevelResourcesSet(pod)

				if allocationSkipped {
					if ok {
						t.Errorf("expected no container cpu set for %s after restore (allocation skipped), but got some", container.Name)
					}
				} else {
					if !ok {
						t.Errorf("expected container cpu set to be present for %s after restore", container.Name)
					}
					// Only check size if we expect assignments (guaranteed)
					if p := mgr2.(*manager).policy.(*staticPolicy); p.guaranteedCPUs(logger, pod, container) > 0 {
						if containerCPUSet.IsEmpty() {
							t.Errorf("expected container cpu set to be non-empty for guaranteed container %s", container.Name)
						}
					}
				}
			}
		})
	}
}

// TestPodLevelResourcesReallocationAfterRestart reproduces the scenario from
// https://github.com/kubernetes/kubernetes/issues/140989: when a pod with
// pod-level resources is re-admitted after a kubelet restart, the pod-level
// CPU allocation restored from the state checkpoint must be preserved.
//
// Before the fix, allocatePodForAdd unconditionally re-allocated the pod:
//   - on a 4 CPU node (1 reserved) the pod was rejected with "not enough cpus
//     available to satisfy request: requested=2, available=1"
//   - on a 6 CPU node the pod CPU set silently changed (e.g. 1-2 -> 3-4),
//     leaking the originally assigned exclusive CPUs
func TestPodLevelResourcesReallocationAfterRestart(t *testing.T) {
	if runtime.GOOS == "windows" {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.WindowsCPUAndMemoryAffinity, true)
	}

	testCases := []struct {
		description    string
		numCPUs        int
		podRequest     string
		containerSpecs []*containerOptions
	}{
		{
			description: "4 CPU node, pod requesting 2 CPUs",
			numCPUs:     4,
			podRequest:  "2",
			containerSpecs: []*containerOptions{
				{name: "container1", request: "2", limit: "2"},
			},
		},
		{
			description: "6 CPU node, pod requesting 2 CPUs",
			numCPUs:     6,
			podRequest:  "2",
			containerSpecs: []*containerOptions{
				{name: "container1", request: "2", limit: "2"},
			},
		},
		{
			description: "6 CPU node, pod requesting 4 CPUs split between two containers",
			numCPUs:     6,
			podRequest:  "4",
			containerSpecs: []*containerOptions{
				{name: "container1", request: "2", limit: "2"},
				{name: "container2", request: "2", limit: "2"},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

			sDir := t.TempDir()
			logger, tCtx := ktesting.NewTestContext(t)

			// First manager instance - initial allocation
			mgr, err := newRestoreTestManager(logger, tc.numCPUs, sDir)
			if err != nil {
				t.Fatalf("could not create manager: %v", err)
			}

			// Create pod with pod-level resources
			pod := makeMultiContainerPodWithOptionsAndPodLevelResources(tc.podRequest, nil, tc.containerSpecs)
			pod.Name = "test-pod"
			pod.UID = types.UID("test-pod")

			// Start manager
			err = mgr.Start(tCtx, func() []*v1.Pod { return []*v1.Pod{pod} }, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
			if err != nil {
				t.Fatalf("could not start manager: %v", err)
			}

			// Initial pod-level allocation
			err = mgr.AllocatePod(logger, pod, lifecycle.AddOperation)
			if err != nil {
				t.Fatalf("initial pod allocation failed: %v", err)
			}

			// Record the initial pod-level CPU assignment
			initialPodCPUSet, ok := mgr.State().GetPodCPUSet(string(pod.UID))
			if !ok {
				t.Fatalf("expected pod cpu set to be present after initial allocation")
			}
			if initialPodCPUSet.IsEmpty() {
				t.Fatalf("expected non-empty pod cpu set after initial allocation")
			}

			// Record container assignments
			initialContainerAssignments := make(map[string]cpuset.CPUSet)
			for i := range pod.Spec.Containers {
				container := &pod.Spec.Containers[i]
				cset, ok := mgr.State().GetCPUSet(string(pod.UID), container.Name)
				if !ok {
					t.Fatalf("expected container cpu set to be present for %s", container.Name)
				}
				initialContainerAssignments[container.Name] = cset
			}

			// Record default CPU set after allocation
			defaultCPUSetAfterAlloc := mgr.State().GetDefaultCPUSet()

			// Simulate kubelet restart by creating a new manager instance
			// reading the same state directory.
			mgr2, err := newRestoreTestManager(logger, tc.numCPUs, sDir)
			if err != nil {
				t.Fatalf("could not create manager 2: %v", err)
			}

			// Start the new manager (should restore state from checkpoint)
			err = mgr2.Start(tCtx, func() []*v1.Pod { return []*v1.Pod{pod} }, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
			if err != nil {
				t.Fatalf("could not start manager 2: %v", err)
			}

			// The pod is re-admitted after the restart. This must restore the
			// checkpointed allocation, not re-allocate it and not fail.
			err = mgr2.AllocatePod(logger, pod, lifecycle.AddOperation)
			if err != nil {
				t.Fatalf("pod allocation after restart failed: %v", err)
			}

			// Verify that restored pod-level CPU set matches the initial one
			restoredPodCPUSet, ok := mgr2.State().GetPodCPUSet(string(pod.UID))
			if !ok {
				t.Fatalf("expected pod cpu set to be present after restore")
			}
			if !restoredPodCPUSet.Equals(initialPodCPUSet) {
				t.Errorf("pod cpu set changed after restart: before=%s, after=%s", initialPodCPUSet.String(), restoredPodCPUSet.String())
			}

			// Verify that container assignments match
			for i := range pod.Spec.Containers {
				container := &pod.Spec.Containers[i]
				restoredCPUSet, ok := mgr2.State().GetCPUSet(string(pod.UID), container.Name)
				if !ok {
					t.Fatalf("expected container cpu set to be present for %s after restore", container.Name)
				}
				if !restoredCPUSet.Equals(initialContainerAssignments[container.Name]) {
					t.Errorf("container %s cpu set changed after restart: before=%s, after=%s",
						container.Name, initialContainerAssignments[container.Name].String(), restoredCPUSet.String())
				}
			}

			// Verify that default CPU set was not modified
			defaultCPUSetAfterRestore := mgr2.State().GetDefaultCPUSet()
			if !defaultCPUSetAfterRestore.Equals(defaultCPUSetAfterAlloc) {
				t.Errorf("default cpu set changed after restart: before=%s, after=%s",
					defaultCPUSetAfterAlloc.String(), defaultCPUSetAfterRestore.String())
			}

			// Verify that pod-level CPUs + default CPUs cover all CPUs: nothing
			// may leak.
			coveredCPUs := restoredPodCPUSet.Union(defaultCPUSetAfterRestore)
			if !coveredCPUs.Equals(allTestCPUs(tc.numCPUs)) {
				t.Errorf("pod CPUs + default CPUs do not cover all CPUs: %s != %s",
					coveredCPUs.String(), allTestCPUs(tc.numCPUs).String())
			}

			// Verify that container assignments are subset of pod-level CPUs
			for i := range pod.Spec.Containers {
				container := &pod.Spec.Containers[i]
				if !initialContainerAssignments[container.Name].IsSubsetOf(initialPodCPUSet) {
					t.Errorf("container %s assignments %s are not a subset of pod-level CPUs %s",
						container.Name, initialContainerAssignments[container.Name].String(), initialPodCPUSet.String())
				}
			}

		})
	}
}

// topoRestoreUniformSingleSocket is a simple topology without SMT, so
// takeByTopology always picks the lowest-numbered free CPUs and the test
// expectations are deterministic.
var topoRestoreUniformSingleSocket = &topology.CPUTopology{
	NumCPUs:    8,
	NumSockets: 1,
	NumCores:   8,
	CPUDetails: map[int]topology.CPUInfo{
		0: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
		1: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
		2: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
		3: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
		4: {CoreID: 4, SocketID: 0, NUMANodeID: 0},
		5: {CoreID: 5, SocketID: 0, NUMANodeID: 0},
		6: {CoreID: 6, SocketID: 0, NUMANodeID: 0},
		7: {CoreID: 7, SocketID: 0, NUMANodeID: 0},
	},
}

// TestStaticPolicyRestorePodAllocation exercises AllocatePod against
// hand-crafted state contents, simulating all the intermediate checkpoint
// states the kubelet may leave behind when it stops while admitting a pod
// with pod-level resources:
//   - the pod-level CPU set and all the container assignments were saved,
//   - the pod-level CPU set was saved with only some container assignments,
//   - the pod-level CPU set was saved with no container assignment at all.
//
// Everything found in the state must be restored untouched, only the missing
// assignments may be computed, carved out of the restored pod-level CPU set.
// If the missing assignments cannot be satisfied, the pod allocation must
// fail with a regular allocation error and every CPU held by the pod must be
// released back to the default CPU set.
func TestStaticPolicyRestorePodAllocation(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	allCPUs := cpuset.New(0, 1, 2, 3, 4, 5, 6, 7)
	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways

	testCases := []struct {
		description     string
		pod             *v1.Pod
		stDefaultCPUSet cpuset.CPUSet
		stPodCPUSet     cpuset.CPUSet
		stAssignments   map[string]cpuset.CPUSet
		repeatAllocate  bool
		expErr          bool
		expAssignments  map[string]cpuset.CPUSet
		expPodCPUSet    cpuset.CPUSet
		// expDefaultCPUSet is checked in both the success and the error case;
		// on error it must equal the pre-allocation default plus all the CPUs
		// the pod was holding.
		expDefaultCPUSet cpuset.CPUSet
	}{
		{
			description: "complete checkpoint is restored as-is",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-full", "4", "4", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
				{name: "c2", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3, 4),
			stAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
				"c2": cpuset.New(3, 4),
			},
			expAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
				"c2": cpuset.New(3, 4),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3, 4),
			expDefaultCPUSet: cpuset.New(0, 5, 6, 7),
		},
		{
			description: "complete checkpoint is restored as-is even when a fresh allocation would pick different CPUs",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-moved", "4", "4", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
				{name: "c2", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 1, 2, 7),
			stPodCPUSet:     cpuset.New(3, 4, 5, 6),
			stAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(3, 4),
				"c2": cpuset.New(5, 6),
			},
			expAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(3, 4),
				"c2": cpuset.New(5, 6),
			},
			expPodCPUSet:     cpuset.New(3, 4, 5, 6),
			expDefaultCPUSet: cpuset.New(0, 1, 2, 7),
		},
		{
			description: "partial checkpoint: the missing container assignment is carved out of the pod CPU set",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-partial", "4", "4", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
				{name: "c2", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3, 4),
			stAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
			},
			expAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
				"c2": cpuset.New(3, 4),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3, 4),
			expDefaultCPUSet: cpuset.New(0, 5, 6, 7),
		},
		{
			description: "partial checkpoint: no container assignments at all, everything is carved out of the pod CPU set",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-none", "4", "4", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
				{name: "c2", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3, 4),
			stAssignments:   map[string]cpuset.CPUSet{},
			expAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
				"c2": cpuset.New(3, 4),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3, 4),
			expDefaultCPUSet: cpuset.New(0, 5, 6, 7),
		},
		{
			description: "partial checkpoint: the missing non-guaranteed container gets the recomputed pod shared pool",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-shared", "3", "3", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
				{name: "c2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 4, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3),
			stAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
			},
			expAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
				"c2": cpuset.New(3),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3),
			expDefaultCPUSet: cpuset.New(0, 4, 5, 6, 7),
		},
		{
			description: "partial checkpoint: the missing guaranteed app container reuses the CPUs of a restored regular init container",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-init-reuse", "2", "2", []containerSpec{
				{name: "i1", request: "2", limit: "2"},
			}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 3, 4, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2),
			stAssignments: map[string]cpuset.CPUSet{
				"i1": cpuset.New(1, 2),
			},
			expAssignments: map[string]cpuset.CPUSet{
				"i1": cpuset.New(1, 2),
				"c1": cpuset.New(1, 2),
			},
			expPodCPUSet:     cpuset.New(1, 2),
			expDefaultCPUSet: cpuset.New(0, 3, 4, 5, 6, 7),
		},
		{
			description: "partial checkpoint: the missing guaranteed app container does not collide with a restored sidecar",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-sidecar", "4", "4", []containerSpec{
				{name: "s1", request: "2", limit: "2", restartPolicy: &containerRestartPolicyAlways},
			}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3, 4),
			stAssignments: map[string]cpuset.CPUSet{
				"s1": cpuset.New(1, 2),
			},
			expAssignments: map[string]cpuset.CPUSet{
				"s1": cpuset.New(1, 2),
				"c1": cpuset.New(3, 4),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3, 4),
			expDefaultCPUSet: cpuset.New(0, 5, 6, 7),
		},
		{
			description: "partial checkpoint: the missing sidecar does not collide with a restored sidecar declared earlier in the spec",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-two-sidecars", "6", "6", []containerSpec{
				{name: "s1", request: "2", limit: "2", restartPolicy: &containerRestartPolicyAlways},
				{name: "s2", request: "2", limit: "2", restartPolicy: &containerRestartPolicyAlways},
			}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3, 4, 5, 6),
			stAssignments: map[string]cpuset.CPUSet{
				"s1": cpuset.New(1, 2),
			},
			expAssignments: map[string]cpuset.CPUSet{
				"s1": cpuset.New(1, 2),
				"s2": cpuset.New(3, 4),
				"c1": cpuset.New(5, 6),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3, 4, 5, 6),
			expDefaultCPUSet: cpuset.New(0, 7),
		},
		{
			description: "non-prefix leftovers: a hole among the app containers releases the stale allocation and the pod is allocated from scratch",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-nonprefix-app", "4", "4", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
				{name: "c2", request: "2", limit: "2"},
			}),
			// Interrupted cleanup of a previous pod instance: c2 still has an
			// assignment while its predecessor c1 does not. The assignments
			// are written in spec order, so this shape cannot come from a
			// partially checkpointed admission.
			stDefaultCPUSet: cpuset.New(0, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3, 4),
			stAssignments: map[string]cpuset.CPUSet{
				"c2": cpuset.New(2, 4),
			},
			expAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
				"c2": cpuset.New(3, 4),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3, 4),
			expDefaultCPUSet: cpuset.New(0, 5, 6, 7),
		},
		{
			description: "non-prefix leftovers: a hole between init and app containers releases the stale allocation and the pod is allocated from scratch",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-nonprefix-init", "2", "2", []containerSpec{
				{name: "i1", request: "2", limit: "2"},
			}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 3, 4, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2),
			stAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(2),
			},
			expAssignments: map[string]cpuset.CPUSet{
				"i1": cpuset.New(1, 2),
				"c1": cpuset.New(1, 2),
			},
			expPodCPUSet:     cpuset.New(1, 2),
			expDefaultCPUSet: cpuset.New(0, 3, 4, 5, 6, 7),
		},
		{
			description: "stale pod-level CPU set overlapping the default CPU set is dropped and the pod is allocated from scratch",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-stale-overlap", "2", "2", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
			}),
			// Interrupted release: the CPUs already went back to the default
			// CPU set, but the kubelet stopped before the pod-level entry was
			// dropped from the state. Restoring the entry would hand out CPUs
			// which are also in the shared pool.
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			stPodCPUSet:     cpuset.New(5, 6),
			stAssignments:   map[string]cpuset.CPUSet{},
			expAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
			},
			expPodCPUSet:     cpuset.New(1, 2),
			expDefaultCPUSet: cpuset.New(0, 3, 4, 5, 6, 7),
		},
		{
			description: "partial checkpoint: the missing non-guaranteed sidecar gets the recomputed pod shared pool",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-ngu-sidecar", "3", "3", []containerSpec{
				{name: "s1", restartPolicy: &containerRestartPolicyAlways},
			}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 4, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3),
			stAssignments:   map[string]cpuset.CPUSet{},
			expAssignments: map[string]cpuset.CPUSet{
				"s1": cpuset.New(3),
				"c1": cpuset.New(1, 2),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3),
			expDefaultCPUSet: cpuset.New(0, 4, 5, 6, 7),
		},
		{
			description: "partial checkpoint: the missing non-guaranteed regular init container gets the pod CPU set minus the sidecar CPUs",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-ngu-init", "4", "4", []containerSpec{
				{name: "s1", request: "2", limit: "2", restartPolicy: &containerRestartPolicyAlways},
				{name: "i1"},
			}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3, 4),
			stAssignments: map[string]cpuset.CPUSet{
				"s1": cpuset.New(1, 2),
			},
			// The regular init container may overlap the app container: it
			// runs and terminates before the app containers start.
			expAssignments: map[string]cpuset.CPUSet{
				"s1": cpuset.New(1, 2),
				"i1": cpuset.New(3, 4),
				"c1": cpuset.New(3, 4),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3, 4),
			expDefaultCPUSet: cpuset.New(0, 5, 6, 7),
		},
		{
			description: "partial checkpoint: the restored non-guaranteed regular init container assignment is preserved as-is",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-ngu-init-restored", "4", "4", []containerSpec{
				{name: "s1", request: "2", limit: "2", restartPolicy: &containerRestartPolicyAlways},
				{name: "i1"},
			}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3, 4),
			stAssignments: map[string]cpuset.CPUSet{
				"s1": cpuset.New(1, 2),
				// Deliberately narrower than the bubble-minus-sidecars value
				// the current partitioning would compute ({3,4}), e.g. a
				// checkpoint taken by a previous kubelet version: it must be
				// preserved untouched, not recomputed.
				"i1": cpuset.New(3),
			},
			expAssignments: map[string]cpuset.CPUSet{
				"s1": cpuset.New(1, 2),
				"i1": cpuset.New(3),
				"c1": cpuset.New(3, 4),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3, 4),
			expDefaultCPUSet: cpuset.New(0, 5, 6, 7),
		},
		{
			description: "partial checkpoint: missing regular init and app containers reuse the CPUs of a restored regular init container",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-multi-init-reuse", "2", "2", []containerSpec{
				{name: "i1", request: "2", limit: "2"},
				{name: "i2", request: "2", limit: "2"},
			}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 3, 4, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2),
			stAssignments: map[string]cpuset.CPUSet{
				"i1": cpuset.New(1, 2),
			},
			// Regular init containers run sequentially and terminate before
			// the app containers start, so they all share the pod CPU set.
			expAssignments: map[string]cpuset.CPUSet{
				"i1": cpuset.New(1, 2),
				"i2": cpuset.New(1, 2),
				"c1": cpuset.New(1, 2),
			},
			expPodCPUSet:     cpuset.New(1, 2),
			expDefaultCPUSet: cpuset.New(0, 3, 4, 5, 6, 7),
		},
		{
			description: "restoring is idempotent: a second AllocatePod call leaves the state unchanged",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-idempotent", "4", "4", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
				{name: "c2", request: "2", limit: "2"},
			}),
			stDefaultCPUSet: cpuset.New(0, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2, 3, 4),
			stAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
			},
			repeatAllocate: true,
			expAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
				"c2": cpuset.New(3, 4),
			},
			expPodCPUSet:     cpuset.New(1, 2, 3, 4),
			expDefaultCPUSet: cpuset.New(0, 5, 6, 7),
		},
		{
			description: "failed completion releases every CPU held by the pod back to the default CPU set",
			pod: makePodWithContainersAndPodLevelResources("pod-restore-error", "4", "4", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
				{name: "c2", request: "2", limit: "2"},
			}),
			// Inconsistent checkpoint: the pod-level CPU set is too small to
			// fit the assignment of c2 next to the restored assignment of c1.
			stDefaultCPUSet: cpuset.New(0, 3, 4, 5, 6, 7),
			stPodCPUSet:     cpuset.New(1, 2),
			stAssignments: map[string]cpuset.CPUSet{
				"c1": cpuset.New(1, 2),
			},
			expErr:           true,
			expDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

			policy, err := NewStaticPolicy(logger, topoRestoreUniformSingleSocket, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}

			podUID := string(tc.pod.UID)
			st := &mockState{
				assignments:    state.ContainerCPUAssignments{podUID: tc.stAssignments},
				defaultCPUSet:  tc.stDefaultCPUSet,
				podAssignments: state.PodCPUAssignments{podUID: {CPUSet: tc.stPodCPUSet}},
			}

			err = policy.AllocatePod(logger, st, tc.pod, lifecycle.AddOperation)
			if tc.expErr {
				if err == nil {
					t.Fatalf("expected an allocation error, got none")
				}
				// The rejected pod must not hold any resource.
				if cset, ok := st.GetPodCPUSet(podUID); ok {
					t.Errorf("expected the pod-level CPU set to be removed from the state, got %q", cset)
				}
				if got := st.GetCPUAssignments()[podUID]; len(got) != 0 {
					t.Errorf("expected the container assignments to be removed from the state, got %v", got)
				}
				if !st.GetDefaultCPUSet().Equals(tc.expDefaultCPUSet) {
					t.Errorf("expected default cpuset %q, got %q", tc.expDefaultCPUSet, st.GetDefaultCPUSet())
				}
				return
			}
			if err != nil {
				t.Fatalf("AllocatePod() failed: %v", err)
			}
			if tc.repeatAllocate {
				if err := policy.AllocatePod(logger, st, tc.pod, lifecycle.AddOperation); err != nil {
					t.Fatalf("second AllocatePod() failed: %v", err)
				}
			}

			podCPUSet, ok := st.GetPodCPUSet(podUID)
			if !ok {
				t.Fatalf("expected a pod-level CPU set in the state")
			}
			if !podCPUSet.Equals(tc.expPodCPUSet) {
				t.Errorf("expected pod-level cpuset %q, got %q", tc.expPodCPUSet, podCPUSet)
			}
			if !st.GetDefaultCPUSet().Equals(tc.expDefaultCPUSet) {
				t.Errorf("expected default cpuset %q, got %q", tc.expDefaultCPUSet, st.GetDefaultCPUSet())
			}
			expAssignments := state.ContainerCPUAssignments{podUID: tc.expAssignments}
			if !reflect.DeepEqual(st.GetCPUAssignments(), expAssignments) {
				t.Errorf("expected assignments %v, got %v", expAssignments, st.GetCPUAssignments())
			}

			// Global accounting invariants: no CPU may be leaked or double-booked.
			if !st.GetDefaultCPUSet().Intersection(podCPUSet).IsEmpty() {
				t.Errorf("default cpuset %q overlaps with the pod cpuset %q", st.GetDefaultCPUSet(), podCPUSet)
			}
			if covered := st.GetDefaultCPUSet().Union(podCPUSet); !covered.Equals(allCPUs) {
				t.Errorf("default cpuset and pod cpuset %q do not cover all CPUs %q", covered, allCPUs)
			}
			for name, cset := range st.GetCPUAssignments()[podUID] {
				if !cset.IsSubsetOf(podCPUSet) {
					t.Errorf("assignment %q of container %q is not a subset of the pod cpuset %q", cset, name, podCPUSet)
				}
			}
		})
	}
}

// topoRestoreDualNUMA is a dual-NUMA-node topology without SMT: CPUs 0-3 live
// on NUMA node 0, CPUs 4-7 on NUMA node 1. It makes the hint regeneration
// observable: hints computed from the restored pod-level CPU set point at the
// NUMA node of the restored CPUs, while hints computed from the default CPU
// set would (also) prefer the other node.
var topoRestoreDualNUMA = &topology.CPUTopology{
	NumCPUs:      8,
	NumSockets:   1,
	NumCores:     8,
	NumNUMANodes: 2,
	CPUDetails: map[int]topology.CPUInfo{
		0: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
		1: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
		2: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
		3: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
		4: {CoreID: 4, SocketID: 0, NUMANodeID: 1},
		5: {CoreID: 5, SocketID: 0, NUMANodeID: 1},
		6: {CoreID: 6, SocketID: 0, NUMANodeID: 1},
		7: {CoreID: 7, SocketID: 0, NUMANodeID: 1},
	},
}

// TestGetPodTopologyHintsWithRestoredPodCPUSet verifies that the pod-scope
// hint generation regenerates the hints from a pod-level CPU set restored
// from the checkpoint, even when the container assignments were not
// checkpointed. Without this, the Topology Manager could reject the pod
// during re-admission after a kubelet restart before the CPU manager gets a
// chance to restore the allocation.
func TestGetPodTopologyHintsWithRestoredPodCPUSet(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	testCases := []struct {
		description   string
		stPodCPUSet   cpuset.CPUSet
		expEmptyHints bool
		// expNUMAAffinityBits is the NUMA affinity every preferred hint must
		// carry; only checked when expEmptyHints is false.
		expNUMAAffinityBits []int
	}{
		{
			description: "hints are regenerated from the restored pod-level CPU set",
			// The restored CPUs live on NUMA node 1. A fresh hint computation
			// from the default CPU set would prefer NUMA node 0 as well, so a
			// preferred hint for node 0 means the restored pod-level CPU set
			// was ignored.
			stPodCPUSet:         cpuset.New(5, 6),
			expNUMAAffinityBits: []int{1},
		},
		{
			description:   "unsatisfiable hints when the restored pod-level CPU set does not match the request",
			stPodCPUSet:   cpuset.New(1, 2, 3),
			expEmptyHints: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

			policy, err := NewStaticPolicy(logger, topoRestoreDualNUMA, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}

			pod := makePodWithContainersAndPodLevelResources("pod-restore-hints", "2", "2", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
			})

			// The pod-level CPU set was checkpointed, the container assignments were not.
			st := &mockState{
				assignments:    state.ContainerCPUAssignments{},
				defaultCPUSet:  cpuset.New(0, 1, 2, 3, 4, 5, 6, 7).Difference(tc.stPodCPUSet),
				podAssignments: state.PodCPUAssignments{string(pod.UID): {CPUSet: tc.stPodCPUSet}},
			}

			hints := policy.GetPodTopologyHints(logger, st, pod, lifecycle.AddOperation)
			cpuHints, ok := hints[string(v1.ResourceCPU)]
			if !ok {
				t.Fatalf("expected hints for %q, got %v", v1.ResourceCPU, hints)
			}
			if tc.expEmptyHints {
				if len(cpuHints) != 0 {
					t.Errorf("expected unsatisfiable (empty) hints, got %v", cpuHints)
				}
				return
			}
			// Every preferred hint must carry the NUMA affinity of the
			// restored CPUs, proving the hints were regenerated from the
			// restored pod-level CPU set and not computed from the default
			// CPU set.
			expAffinity := newNUMAAffinity(tc.expNUMAAffinityBits...)
			preferred := 0
			for _, hint := range cpuHints {
				if !hint.Preferred {
					continue
				}
				preferred++
				if !hint.NUMANodeAffinity.IsEqual(expAffinity) {
					t.Errorf("preferred hint %v does not match the NUMA affinity %v of the restored pod-level CPU set", hint, expAffinity)
				}
			}
			if preferred == 0 {
				t.Errorf("expected a preferred hint regenerated from the restored pod-level CPU set, got %v", cpuHints)
			}
		})
	}
}

// TestGetPodTopologyHintsStalePodCPUSet verifies that the pod-scope hint
// generation does NOT regenerate hints from a stale pod-level CPU set (one
// that overlaps the default CPU set, left behind by an interrupted release).
// Instead, fresh hints must be generated from the available CPUs, so the
// Topology Manager does not reject the pod under the restricted policy before
// allocatePodForAdd gets a chance to clean up the stale state.
func TestGetPodTopologyHintsStalePodCPUSet(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

	policy, err := NewStaticPolicy(logger, topoRestoreDualNUMA, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
	if err != nil {
		t.Fatalf("NewStaticPolicy() failed: %v", err)
	}

	pod := makePodWithContainersAndPodLevelResources("pod-stale-hints", "2", "2", []containerSpec{}, []containerSpec{
		{name: "c1", request: "2", limit: "2"},
	})

	// Simulate an interrupted release: the pod-level CPU set (CPUs 5,6) was
	// not dropped, but the CPUs were already returned to the default CPU set.
	stalePodCPUSet := cpuset.New(5, 6)
	st := &mockState{
		assignments:    state.ContainerCPUAssignments{},
		defaultCPUSet:  cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		podAssignments: state.PodCPUAssignments{string(pod.UID): {CPUSet: stalePodCPUSet}},
	}

	hints := policy.GetPodTopologyHints(logger, st, pod, lifecycle.AddOperation)
	cpuHints, ok := hints[string(v1.ResourceCPU)]
	if !ok {
		t.Fatalf("expected hints for %q, got %v", v1.ResourceCPU, hints)
	}

	// The stale pod-level CPU set lives on NUMA node 1. If hints were
	// regenerated from it, every preferred hint would carry NUMA node 1
	// affinity only. Instead, fresh hints from the full default CPU set
	// should prefer both NUMA nodes (each has 4 CPUs, enough for 2).
	// Verify that at least one preferred hint prefers NUMA node 0, proving
	// the stale pod-level CPU set was not used.
	node0Preferred := false
	for _, hint := range cpuHints {
		if !hint.Preferred {
			continue
		}
		if hint.NUMANodeAffinity.IsSet(0) {
			node0Preferred = true
		}
	}
	if !node0Preferred {
		t.Errorf("expected a preferred hint for NUMA node 0 (fresh hints from the default CPU set), got %v", cpuHints)
	}
}

// TestGetPodTopologyHintsStaleNonPrefixPodCPUSet verifies the hint generation
// for the other stale shape: the container assignments do not form a prefix of
// the pod spec (leftovers of an interrupted cleanup), so the pod-level CPU set
// is still held and will be released back to the default CPU set during the
// allocation. The fresh hints must account for those CPUs, and the stale
// container assignments must not be taken into account: otherwise the Topology
// Manager rejects the pod before allocatePodForAdd can clean the state up, and
// since the cleanup only runs during allocation, the rejection repeats on
// every retry.
func TestGetPodTopologyHintsStaleNonPrefixPodCPUSet(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	// CPUs 0-3 live on NUMA node 0, CPUs 4-7 on NUMA node 1, CPU 0 is reserved.
	testCases := []struct {
		description string
		pod         *v1.Pod
		// stale leftovers: pod-level CPU set held, assignments with a hole
		stPodCPUSet     cpuset.CPUSet
		stAssignments   map[string]cpuset.CPUSet
		stDefaultCPUSet cpuset.CPUSet
		// expNUMANodePreferred is the NUMA node which must carry a preferred
		// hint once the CPUs about to be released are accounted for.
		expNUMANodePreferred int
	}{
		{
			description: "the CPUs to be released are accounted for in the fresh hints",
			pod: makePodWithContainersAndPodLevelResources("pod-stale-nonprefix-hints", "4", "4", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
				{name: "c2", request: "2", limit: "2"},
			}),
			// Only NUMA node 1 can satisfy the 4 CPU request, and only once the
			// stale pod-level CPU set goes back to the default CPU set: the
			// default CPU set alone holds 3 assignable CPUs.
			stPodCPUSet:          cpuset.New(4, 5, 6, 7),
			stAssignments:        map[string]cpuset.CPUSet{"c2": cpuset.New(6, 7)},
			stDefaultCPUSet:      cpuset.New(0, 1, 2, 3),
			expNUMANodePreferred: 1,
		},
		{
			description: "the stale assignment of a non-guaranteed container does not make the hints unsatisfiable",
			pod: makePodWithContainersAndPodLevelResources("pod-stale-nonprefix-ngu-hints", "3", "3", []containerSpec{}, []containerSpec{
				{name: "c1", request: "2", limit: "2"},
				{name: "c2"},
			}),
			// The leftover assignment of the non-guaranteed container c2 holds a
			// shared pool CPU, which never matches its zero exclusive request.
			stPodCPUSet:          cpuset.New(5, 6, 7),
			stAssignments:        map[string]cpuset.CPUSet{"c2": cpuset.New(7)},
			stDefaultCPUSet:      cpuset.New(0, 1, 2, 3, 4),
			expNUMANodePreferred: 1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

			policy, err := NewStaticPolicy(logger, topoRestoreDualNUMA, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}

			st := &mockState{
				assignments:    state.ContainerCPUAssignments{string(tc.pod.UID): tc.stAssignments},
				defaultCPUSet:  tc.stDefaultCPUSet,
				podAssignments: state.PodCPUAssignments{string(tc.pod.UID): {CPUSet: tc.stPodCPUSet}},
			}

			hints := policy.GetPodTopologyHints(logger, st, tc.pod, lifecycle.AddOperation)
			cpuHints, ok := hints[string(v1.ResourceCPU)]
			if !ok {
				t.Fatalf("expected hints for %q, got %v", v1.ResourceCPU, hints)
			}
			if len(cpuHints) == 0 {
				t.Fatalf("expected satisfiable hints accounting for the CPUs to be released, got none")
			}

			// A preferred hint for the expected NUMA node proves both that the
			// CPUs about to be released were accounted for and that the stale
			// container assignments did not constrain the affinity.
			found := false
			for _, hint := range cpuHints {
				if hint.Preferred && hint.NUMANodeAffinity.IsSet(tc.expNUMANodePreferred) {
					found = true
				}
			}
			if !found {
				t.Errorf("expected a preferred hint for NUMA node %d, got %v", tc.expNUMANodePreferred, cpuHints)
			}
		})
	}
}

// TestPodLevelResourcesRestoreCPUIntegrity tests that CPU accounting stays
// consistent across restarts with multiple pods and no CPUs are lost or
// duplicated.
func TestPodLevelResourcesRestoreCPUIntegrity(t *testing.T) {
	if runtime.GOOS == "windows" {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.WindowsCPUAndMemoryAffinity, true)
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

	const numCPUs = 8

	sDir := t.TempDir()
	logger, tCtx := ktesting.NewTestContext(t)

	// Create the pods, each with a single guaranteed container filling the
	// whole pod-level request.
	podsSpecs := []struct{ name, request string }{
		{"pod1", "2"},
		{"pod2", "2"},
	}
	pods := make([]*v1.Pod, len(podsSpecs))
	for i, spec := range podsSpecs {
		containers := []*containerOptions{
			{name: fmt.Sprintf("container-%d", i), request: spec.request, limit: spec.request},
		}
		pod := makeMultiContainerPodWithOptionsAndPodLevelResources(spec.request, nil, containers)
		pod.Name = spec.name
		pod.UID = types.UID(spec.name)
		pods[i] = pod
	}

	mgr, err := newRestoreTestManager(logger, numCPUs, sDir)
	if err != nil {
		t.Fatalf("could not create manager: %v", err)
	}

	err = mgr.Start(tCtx, func() []*v1.Pod { return pods }, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
	if err != nil {
		t.Fatalf("could not start manager: %v", err)
	}

	// Allocate all pods
	for _, pod := range pods {
		err = mgr.AllocatePod(logger, pod, lifecycle.AddOperation)
		if err != nil {
			t.Fatalf("pod allocation failed for %s: %v", pod.Name, err)
		}
	}

	// Record state
	initialDefaultCPUSet := mgr.State().GetDefaultCPUSet()
	initialPodCPUSets := make(map[string]cpuset.CPUSet)
	for _, pod := range pods {
		cset, _ := mgr.State().GetPodCPUSet(string(pod.UID))
		initialPodCPUSets[string(pod.UID)] = cset
	}

	// Verify CPU integrity before restart
	totalBefore := initialDefaultCPUSet
	for _, cset := range initialPodCPUSets {
		totalBefore = totalBefore.Union(cset)
	}
	if !totalBefore.Equals(allTestCPUs(numCPUs)) {
		t.Errorf("CPU integrity check failed before restart: expected=%s, got=%s", allTestCPUs(numCPUs).String(), totalBefore.String())
	}

	// Second manager instance (restart)
	mgr2, err := newRestoreTestManager(logger, numCPUs, sDir)
	if err != nil {
		t.Fatalf("could not create manager 2: %v", err)
	}

	err = mgr2.Start(tCtx, func() []*v1.Pod { return pods }, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
	if err != nil {
		t.Fatalf("could not start manager 2: %v", err)
	}

	// Re-admit all pods
	for _, pod := range pods {
		err = mgr2.AllocatePod(logger, pod, lifecycle.AddOperation)
		if err != nil {
			t.Fatalf("pod allocation after restart failed for %s: %v", pod.Name, err)
		}
	}

	// Verify CPU integrity after restart
	restoredDefaultCPUSet := mgr2.State().GetDefaultCPUSet()
	totalAfter := restoredDefaultCPUSet
	for _, pod := range pods {
		cset, _ := mgr2.State().GetPodCPUSet(string(pod.UID))
		totalAfter = totalAfter.Union(cset)
	}
	if !totalAfter.Equals(allTestCPUs(numCPUs)) {
		t.Errorf("CPU integrity check failed after restart: expected=%s, got=%s", allTestCPUs(numCPUs).String(), totalAfter.String())
	}

	// Verify that default CPU set was properly restored
	if !restoredDefaultCPUSet.Equals(initialDefaultCPUSet) {
		t.Errorf("default cpuset changed after restart: before=%s, after=%s", initialDefaultCPUSet.String(), restoredDefaultCPUSet.String())
	}

	// Verify all pods have same allocations
	for _, pod := range pods {
		restoredCPUSet, _ := mgr2.State().GetPodCPUSet(string(pod.UID))
		if !restoredCPUSet.Equals(initialPodCPUSets[string(pod.UID)]) {
			t.Errorf("pod %s cpuset changed: before=%s, after=%s", pod.Name, initialPodCPUSets[string(pod.UID)].String(), restoredCPUSet.String())
		}
	}
}

// TestMixedScopeRestoreCPUIntegrity verifies the CPU accounting invariant
// across a kubelet restart when a pod with pod-level resources shares the
// node with a classic guaranteed pod managed at container scope: both pods
// keep their exact CPU assignments, and the default CPU set, the pod-level
// CPU set and the container-scope assignments together cover all the CPUs.
func TestMixedScopeRestoreCPUIntegrity(t *testing.T) {
	if runtime.GOOS == "windows" {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.WindowsCPUAndMemoryAffinity, true)
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

	const numCPUs = 8

	sDir := t.TempDir()
	logger, tCtx := ktesting.NewTestContext(t)

	// A pod with pod-level resources, allocated at pod scope...
	podLevelPod := makeMultiContainerPodWithOptionsAndPodLevelResources("2", nil, []*containerOptions{
		{name: "container-0", request: "2", limit: "2"},
	})
	podLevelPod.Name = "pod-level-pod"
	podLevelPod.UID = types.UID("pod-level-pod")
	podLevelContainer := &podLevelPod.Spec.Containers[0]

	// ...and a classic guaranteed pod, allocated at container scope.
	containerScopePod := makeMultiContainerPodWithOptions(nil, []*containerOptions{
		{name: "container-0", request: "2", limit: "2"},
	})
	containerScopePod.Name = "container-scope-pod"
	containerScopePod.UID = types.UID("container-scope-pod")
	containerScopeContainer := &containerScopePod.Spec.Containers[0]

	pods := []*v1.Pod{podLevelPod, containerScopePod}

	mgr, err := newRestoreTestManager(logger, numCPUs, sDir)
	if err != nil {
		t.Fatalf("could not create manager: %v", err)
	}

	err = mgr.Start(tCtx, func() []*v1.Pod { return pods }, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
	if err != nil {
		t.Fatalf("could not start manager: %v", err)
	}

	if err := mgr.AllocatePod(logger, podLevelPod, lifecycle.AddOperation); err != nil {
		t.Fatalf("pod-level pod allocation failed: %v", err)
	}
	if err := mgr.Allocate(tCtx, containerScopePod, containerScopeContainer, lifecycle.AddOperation); err != nil {
		t.Fatalf("container-scope pod allocation failed: %v", err)
	}

	// Record the state before the simulated restart and check the accounting.
	initialPodCPUSet, ok := mgr.State().GetPodCPUSet(string(podLevelPod.UID))
	if !ok {
		t.Fatalf("expected a pod-level cpu set for %s", podLevelPod.Name)
	}
	initialPodLevelContainerCPUSet, ok := mgr.State().GetCPUSet(string(podLevelPod.UID), podLevelContainer.Name)
	if !ok {
		t.Fatalf("expected a container cpu set for %s in %s", podLevelContainer.Name, podLevelPod.Name)
	}
	initialContainerScopeCPUSet, ok := mgr.State().GetCPUSet(string(containerScopePod.UID), containerScopeContainer.Name)
	if !ok {
		t.Fatalf("expected a container cpu set for %s in %s", containerScopeContainer.Name, containerScopePod.Name)
	}
	initialDefaultCPUSet := mgr.State().GetDefaultCPUSet()

	if covered := initialDefaultCPUSet.Union(initialPodCPUSet).Union(initialContainerScopeCPUSet); !covered.Equals(allTestCPUs(numCPUs)) {
		t.Fatalf("CPU accounting broken before restart: covered %q, expected %q", covered, allTestCPUs(numCPUs))
	}

	// Simulate a kubelet restart and re-admit both pods.
	mgr2, err := newRestoreTestManager(logger, numCPUs, sDir)
	if err != nil {
		t.Fatalf("could not create manager 2: %v", err)
	}

	err = mgr2.Start(tCtx, func() []*v1.Pod { return pods }, &sourcesReadyStub{}, mockPodStatusProvider{}, mockRuntimeService{}, containermap.NewContainerMap())
	if err != nil {
		t.Fatalf("could not start manager 2: %v", err)
	}
	if err := mgr2.AllocatePod(logger, podLevelPod, lifecycle.AddOperation); err != nil {
		t.Fatalf("pod-level pod allocation after restart failed: %v", err)
	}
	if err := mgr2.Allocate(tCtx, containerScopePod, containerScopeContainer, lifecycle.AddOperation); err != nil {
		t.Fatalf("container-scope pod allocation after restart failed: %v", err)
	}

	// Both pods must keep their exact assignments, and the accounting must
	// still cover all the CPUs without overlaps.
	restoredPodCPUSet, _ := mgr2.State().GetPodCPUSet(string(podLevelPod.UID))
	if !restoredPodCPUSet.Equals(initialPodCPUSet) {
		t.Errorf("pod-level cpu set changed after restart: before=%q, after=%q", initialPodCPUSet, restoredPodCPUSet)
	}
	restoredPodLevelContainerCPUSet, _ := mgr2.State().GetCPUSet(string(podLevelPod.UID), podLevelContainer.Name)
	if !restoredPodLevelContainerCPUSet.Equals(initialPodLevelContainerCPUSet) {
		t.Errorf("pod-level container cpu set changed after restart: before=%q, after=%q", initialPodLevelContainerCPUSet, restoredPodLevelContainerCPUSet)
	}
	restoredContainerScopeCPUSet, _ := mgr2.State().GetCPUSet(string(containerScopePod.UID), containerScopeContainer.Name)
	if !restoredContainerScopeCPUSet.Equals(initialContainerScopeCPUSet) {
		t.Errorf("container-scope cpu set changed after restart: before=%q, after=%q", initialContainerScopeCPUSet, restoredContainerScopeCPUSet)
	}
	restoredDefaultCPUSet := mgr2.State().GetDefaultCPUSet()
	if !restoredDefaultCPUSet.Equals(initialDefaultCPUSet) {
		t.Errorf("default cpu set changed after restart: before=%q, after=%q", initialDefaultCPUSet, restoredDefaultCPUSet)
	}
	if covered := restoredDefaultCPUSet.Union(restoredPodCPUSet).Union(restoredContainerScopeCPUSet); !covered.Equals(allTestCPUs(numCPUs)) {
		t.Errorf("CPU accounting broken after restart: covered %q, expected %q", covered, allTestCPUs(numCPUs))
	}
	if !restoredDefaultCPUSet.Intersection(restoredPodCPUSet).IsEmpty() || !restoredDefaultCPUSet.Intersection(restoredContainerScopeCPUSet).IsEmpty() || !restoredPodCPUSet.Intersection(restoredContainerScopeCPUSet).IsEmpty() {
		t.Errorf("CPU sets overlap after restart: default=%q, podLevel=%q, containerScope=%q", restoredDefaultCPUSet, restoredPodCPUSet, restoredContainerScopeCPUSet)
	}
}

// newRestoreTestManager creates a CPU manager with the static policy on a
// simple single-socket, single-NUMA-node topology with numCPUs cores (no SMT)
// and 1 reserved CPU, backed by a checkpoint in stateDir.
func newRestoreTestManager(logger klog.Logger, numCPUs int, stateDir string) (Manager, error) {
	machineInfo := &cadvisorapi.MachineInfo{
		NumCores: numCPUs,
		Topology: []cadvisorapi.Node{
			{
				Cores: buildCoresTopology(numCPUs),
			},
		},
	}
	return NewManager(
		logger,
		"static",
		nil,
		5*time.Second,
		machineInfo,
		cpuset.New(),
		v1.ResourceList{v1.ResourceCPU: *resource.NewQuantity(1, resource.DecimalSI)},
		stateDir,
		topologymanager.NewFakeManager(logger),
	)
}

// buildCoresTopology creates a simple single-socket topology for testing
func buildCoresTopology(numCPUs int) []cadvisorapi.Core {
	cores := make([]cadvisorapi.Core, numCPUs)
	for i := range numCPUs {
		cores[i] = cadvisorapi.Core{
			Id:      i,
			Threads: []int{i},
		}
	}
	return cores
}

// allTestCPUs returns the set of all the CPU IDs of a node with numCPUs CPUs.
func allTestCPUs(numCPUs int) cpuset.CPUSet {
	ids := make([]int, numCPUs)
	for i := range ids {
		ids[i] = i
	}
	return cpuset.New(ids...)
}
