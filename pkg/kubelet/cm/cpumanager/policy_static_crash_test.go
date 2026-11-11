/*
Copyright The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/utils/cpuset"
)

// newLegacyReleaseState seeds a checkpoint of a pod-level allocation and then
// replays, step by step, the release performed by a kubelet that predates the
// atomic pod release. The container assignments are dropped first, so an
// interrupted release always leaves the pod-level entry behind. When
// withDefaultUpdate is true the CPUs are also returned to the default CPU set,
// which is the crash window such kubelets had after that step.
func newLegacyReleaseState(t *testing.T, st state.State, allCPUs, podCPUs cpuset.CPUSet, podUID string, withDefaultUpdate bool) {
	t.Helper()
	st.SetDefaultCPUSet(allCPUs)
	st.SetCPUSet(podUID, "container", podCPUs)
	st.SetDefaultCPUSet(allCPUs.Difference(podCPUs))
	st.SetPodCPUSet(podUID, podCPUs)
	for containerName := range st.GetCPUAssignments()[podUID] {
		st.Delete(podUID, containerName)
	}
	if withDefaultUpdate {
		st.SetDefaultCPUSet(st.GetDefaultCPUSet().Union(podCPUs))
	}
}

func newCrashTestManager(policy Policy, st state.State) *manager {
	return &manager{
		policy:       policy,
		state:        st,
		sourcesReady: &sourcesReadyStub{},
		activePods:   func() []*v1.Pod { return nil },
		containerMap: containermap.NewContainerMap(),
	}
}

// A legacy checkpoint can hold only the pod-level entry: the kubelet crashed
// after dropping the container assignments. The startup cleanup must not touch
// it, and the stale state GC reclaims the entry once the pod is gone while
// preserving the allocation of active pods.
func TestRemoveStaleStateInterruptedPodCPURelease(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)
	logger, _ := ktesting.NewTestContext(t)
	stateDir := t.TempDir()
	newState := func() state.State {
		st, err := state.NewCheckpointState(logger, stateDir, "cpu_manager_state", "static", containermap.NewContainerMap())
		require.NoError(t, err)
		return st
	}
	policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
	require.NoError(t, err)

	allocatedCPUs := cpuset.New(1, 2)
	allCPUs := topoSingleSocketHT.CPUDetails.CPUs()
	st := newState()
	newLegacyReleaseState(t, st, allCPUs, allocatedCPUs, "pod", false)

	restored := newState()
	require.NoError(t, policy.Start(logger, restored))
	podCPUs, exists := restored.GetPodCPUSet("pod")
	require.True(t, exists, "legacy pod entry holding its CPUs must be preserved")
	require.Equal(t, allocatedCPUs, podCPUs)
	require.Empty(t, restored.GetCPUAssignments())
	require.Equal(t, allCPUs.Difference(allocatedCPUs), restored.GetDefaultCPUSet())

	mgr := newCrashTestManager(policy, restored)
	mgr.activePods = func() []*v1.Pod { return []*v1.Pod{{ObjectMeta: metav1.ObjectMeta{UID: "pod"}}} }
	mgr.removeStaleState(logger)
	_, exists = restored.GetPodCPUSet("pod")
	require.True(t, exists, "active pod allocation must be preserved")

	mgr.activePods = func() []*v1.Pod { return nil }
	mgr.removeStaleState(logger)
	require.Equal(t, allCPUs, restored.GetDefaultCPUSet())
	require.Empty(t, restored.GetPodCPUAssignments())

	mgr.removeStaleState(logger)
	reloaded := newState()
	require.NoError(t, policy.Start(logger, reloaded))
	require.Equal(t, allCPUs, reloaded.GetDefaultCPUSet())
	require.Empty(t, reloaded.GetPodCPUAssignments())
}

// A legacy crash after the CPUs went back to the default CPU set but before the
// pod entry was dropped leaves them allocatable. The startup cleanup drops the
// stale entry so no ownership has to be guessed later.
func TestStartReclaimsLegacyFullyReturnedRelease(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)
	logger, _ := ktesting.NewTestContext(t)
	stateDir := t.TempDir()

	allocatedCPUs := cpuset.New(1, 2)
	allCPUs := topoSingleSocketHT.CPUDetails.CPUs()
	st, err := state.NewCheckpointState(logger, stateDir, "cpu_manager_state", "static", containermap.NewContainerMap())
	require.NoError(t, err)
	newLegacyReleaseState(t, st, allCPUs, allocatedCPUs, "pod", true)

	policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
	require.NoError(t, err)
	require.NoError(t, policy.Start(logger, st))
	require.Equal(t, allCPUs, st.GetDefaultCPUSet())
	require.Empty(t, st.GetPodCPUAssignments())

	reloaded, err := state.NewCheckpointState(logger, stateDir, "cpu_manager_state", "static", containermap.NewContainerMap())
	require.NoError(t, err)
	require.NoError(t, policy.Start(logger, reloaded))
	require.Equal(t, allCPUs, reloaded.GetDefaultCPUSet())
	require.Empty(t, reloaded.GetPodCPUAssignments())
}

// The startup cleanup is conservative: a legacy pod entry which is not fully
// back in the default CPU set is left alone until the stale state GC reclaims
// it, once the pod is known to be gone.
func TestStartPreservesLegacyOrphanUntilStaleGC(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)
	logger, _ := ktesting.NewTestContext(t)
	stateDir := t.TempDir()

	allocatedCPUs := cpuset.New(1, 2)
	allCPUs := topoSingleSocketHT.CPUDetails.CPUs()
	st, err := state.NewCheckpointState(logger, stateDir, "cpu_manager_state", "static", containermap.NewContainerMap())
	require.NoError(t, err)
	newLegacyReleaseState(t, st, allCPUs, allocatedCPUs, "pod", false)

	policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
	require.NoError(t, err)
	require.NoError(t, policy.Start(logger, st))
	_, exists := st.GetPodCPUSet("pod")
	require.True(t, exists, "orphan not returned to the default CPU set must wait for the stale state GC")
	require.Equal(t, allCPUs.Difference(allocatedCPUs), st.GetDefaultCPUSet())

	mgr := newCrashTestManager(policy, st)
	mgr.removeStaleState(logger)
	require.Equal(t, allCPUs, st.GetDefaultCPUSet())
	require.Empty(t, st.GetPodCPUAssignments())
}

// The startup cleanup must not reclaim a live allocation whose containers were
// not checkpointed yet.
func TestStartPreservesActivePodAllocation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)
	logger, _ := ktesting.NewTestContext(t)
	stateDir := t.TempDir()

	allocatedCPUs := cpuset.New(1, 2)
	allCPUs := topoSingleSocketHT.CPUDetails.CPUs()
	st, err := state.NewCheckpointState(logger, stateDir, "cpu_manager_state", "static", containermap.NewContainerMap())
	require.NoError(t, err)
	st.SetDefaultCPUSet(allCPUs.Difference(allocatedCPUs))
	st.SetPodCPUSet("pod", allocatedCPUs)
	st.SetCPUSet("pod", "container", allocatedCPUs)

	reloaded, err := state.NewCheckpointState(logger, stateDir, "cpu_manager_state", "static", containermap.NewContainerMap())
	require.NoError(t, err)
	policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
	require.NoError(t, err)
	require.NoError(t, policy.Start(logger, reloaded))
	podCPUs, exists := reloaded.GetPodCPUSet("pod")
	require.True(t, exists, "active allocation must be preserved")
	require.Equal(t, allocatedCPUs, podCPUs)
	require.Equal(t, allocatedCPUs, reloaded.GetCPUAssignments()["pod"]["container"])
	require.Equal(t, allCPUs.Difference(allocatedCPUs), reloaded.GetDefaultCPUSet())
}

// A legacy crash after the CPUs went back to the default CPU set leaves them
// allocatable. The CPUs can be handed to another pod on the next start; the
// stale state GC must never hand them back to the shared pool a second time,
// not even on the following restart.
func TestRemoveStaleStateInterruptedPodCPUReleaseReallocated(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)
	logger, _ := ktesting.NewTestContext(t)
	stateDir := t.TempDir()
	newState := func() state.State {
		st, err := state.NewCheckpointState(logger, stateDir, "cpu_manager_state", "static", containermap.NewContainerMap())
		require.NoError(t, err)
		return st
	}
	policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
	require.NoError(t, err)

	allocatedCPUs := cpuset.New(1, 2)
	allCPUs := topoSingleSocketHT.CPUDetails.CPUs()
	st := newState()
	newLegacyReleaseState(t, st, allCPUs, allocatedCPUs, "pod1", true)

	restored := newState()
	require.NoError(t, policy.Start(logger, restored))
	require.True(t, allocatedCPUs.IsSubsetOf(restored.GetDefaultCPUSet()), "the interrupted release left the CPUs allocatable")

	// Before the pod sources are ready removeStaleState is a no-op, so a pod
	// admitted on restart can take the CPUs the stale entry still lists.
	restored.SetDefaultCPUSet(restored.GetDefaultCPUSet().Difference(allocatedCPUs))
	restored.SetPodCPUSet("pod2", allocatedCPUs)
	restored.SetCPUSet("pod2", "container", allocatedCPUs)

	mgr := newCrashTestManager(policy, restored)
	mgr.activePods = func() []*v1.Pod {
		return []*v1.Pod{{
			ObjectMeta: metav1.ObjectMeta{UID: "pod2"},
			Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "container"}}},
		}}
	}
	mgr.removeStaleState(logger)

	_, exists := restored.GetPodCPUSet("pod1")
	require.False(t, exists, "stale pod entry must be dropped")
	require.Equal(t, allCPUs.Difference(allocatedCPUs), restored.GetDefaultCPUSet(), "CPUs owned by pod2 must not go back to the shared pool")
	pod2CPUs, exists := restored.GetPodCPUSet("pod2")
	require.True(t, exists)
	require.Equal(t, allocatedCPUs, pod2CPUs)

	reloaded := newState()
	require.NoError(t, policy.Start(logger, reloaded), "state must stay consistent across the next restart")
	require.Equal(t, allCPUs.Difference(allocatedCPUs), reloaded.GetDefaultCPUSet())
	_, exists = reloaded.GetPodCPUSet("pod1")
	require.False(t, exists)
}

// Removing a non-last container of a pod-level allocation only drops that
// container and keeps the CPUs held by the pod.
func TestPodLevelRemoveNonLastContainer(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)
	logger, _ := ktesting.NewTestContext(t)

	allocatedCPUs := cpuset.New(1, 2)
	allCPUs := topoSingleSocketHT.CPUDetails.CPUs()
	st := &mockState{
		assignments: state.ContainerCPUAssignments{
			"pod": {"c1": cpuset.New(1, 2), "c2": cpuset.New(1, 2)},
		},
		podAssignments: state.PodCPUAssignments{"pod": {CPUSet: allocatedCPUs}},
		defaultCPUSet:  allCPUs.Difference(allocatedCPUs),
	}
	policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
	require.NoError(t, err)
	require.NoError(t, policy.RemoveContainer(logger, st, "pod", "c1"))
	// Repeated removal must not release the CPUs still used by c2.
	require.NoError(t, policy.RemoveContainer(logger, st, "pod", "c1"))

	require.Equal(t, state.ContainerCPUAssignments{"pod": {"c2": cpuset.New(1, 2)}}, st.GetCPUAssignments())
	podCPUs, exists := st.GetPodCPUSet("pod")
	require.True(t, exists)
	require.Equal(t, allocatedCPUs, podCPUs)
	require.Equal(t, allCPUs.Difference(allocatedCPUs), st.GetDefaultCPUSet())
}

// Removing the last container releases the whole pod-level allocation at once:
// the container assignment, the pod entry and the CPUs held by the pod.
func TestPodLevelRemoveLastContainer(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)
	logger, _ := ktesting.NewTestContext(t)

	allocatedCPUs := cpuset.New(1, 2)
	allCPUs := topoSingleSocketHT.CPUDetails.CPUs()
	st := &mockState{
		assignments: state.ContainerCPUAssignments{
			"pod": {"c1": cpuset.New(1, 2), "c2": cpuset.New(1, 2)},
		},
		podAssignments: state.PodCPUAssignments{"pod": {CPUSet: allocatedCPUs}},
		defaultCPUSet:  allCPUs.Difference(allocatedCPUs),
	}
	policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
	require.NoError(t, err)
	require.NoError(t, policy.RemoveContainer(logger, st, "pod", "c1"))
	require.NoError(t, policy.RemoveContainer(logger, st, "pod", "c2"))

	require.Empty(t, st.GetCPUAssignments())
	require.Empty(t, st.GetPodCPUAssignments())
	require.Equal(t, allCPUs, st.GetDefaultCPUSet())
}
