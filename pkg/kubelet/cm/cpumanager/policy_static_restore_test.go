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
