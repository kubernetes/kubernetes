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
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/utils/cpuset"
)

type interruptedCPURelease struct {
	state.State
	afterDefaultUpdate bool
}

func (s interruptedCPURelease) Delete(podUID, containerName string) {
	s.State.Delete(podUID, containerName)
	if !s.afterDefaultUpdate {
		panic("interrupted CPU release")
	}
}

func (s interruptedCPURelease) SetDefaultCPUSet(cset cpuset.CPUSet) {
	s.State.SetDefaultCPUSet(cset)
	panic("interrupted CPU release")
}

func TestRemoveStaleStateInterruptedPodCPURelease(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	for _, afterDefaultUpdate := range []bool{false, true} {
		t.Run(fmt.Sprintf("afterDefaultUpdate=%t", afterDefaultUpdate), func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			stateDir := t.TempDir()
			newState := func() state.State {
				st, err := state.NewCheckpointState(logger, stateDir, "cpu_manager_state", "static", containermap.NewContainerMap())
				require.NoError(t, err)
				return st
			}
			st := newState()
			allocatedCPUs := cpuset.New(1, 2)
			allCPUs := topoSingleSocketHT.CPUDetails.CPUs()
			st.SetDefaultCPUSet(allCPUs.Difference(allocatedCPUs))
			st.SetPodCPUSet("pod", allocatedCPUs)
			st.SetCPUSet("pod", "container", allocatedCPUs)
			policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
			require.NoError(t, err)
			require.PanicsWithValue(t, "interrupted CPU release", func() {
				_ = policy.RemoveContainer(logger, interruptedCPURelease{State: st, afterDefaultUpdate: afterDefaultUpdate}, "pod", "container")
			})
			restored := newState()
			require.Empty(t, restored.GetCPUAssignments())
			require.NoError(t, policy.Start(logger, restored))
			mgr := &manager{
				policy:          policy,
				state:           restored,
				lastUpdateState: state.NewMemoryState(logger),
				sourcesReady:    &sourcesReadyStub{},
				activePods:      func() []*v1.Pod { return []*v1.Pod{{ObjectMeta: metav1.ObjectMeta{UID: "pod"}}} },
				containerMap:    containermap.NewContainerMap(),
			}
			// An active pod may not have checkpointed its first container yet.
			mgr.removeStaleState(logger)
			_, exists := restored.GetPodCPUSet("pod")
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
		})
	}
}

// A crash after the CPUs went back to the default CPU set but before the pod
// entry was dropped leaves them allocatable. Reclaiming the stale pod entry
// after a restart must not hand back CPUs another pod got in the meantime.
func TestRemoveStaleStateInterruptedPodCPUReleaseReallocated(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
	logger, _ := ktesting.NewTestContext(t)
	stateDir := t.TempDir()
	newState := func() state.State {
		st, err := state.NewCheckpointState(logger, stateDir, "cpu_manager_state", "static", containermap.NewContainerMap())
		require.NoError(t, err)
		return st
	}
	st := newState()
	allocatedCPUs := cpuset.New(1, 2)
	allCPUs := topoSingleSocketHT.CPUDetails.CPUs()
	st.SetDefaultCPUSet(allCPUs.Difference(allocatedCPUs))
	st.SetPodCPUSet("pod1", allocatedCPUs)
	st.SetCPUSet("pod1", "container", allocatedCPUs)
	policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(0), topologymanager.NewFakeManager(logger), nil)
	require.NoError(t, err)
	require.PanicsWithValue(t, "interrupted CPU release", func() {
		_ = policy.RemoveContainer(logger, interruptedCPURelease{State: st, afterDefaultUpdate: true}, "pod1", "container")
	})

	restored := newState()
	require.NoError(t, policy.Start(logger, restored))
	require.True(t, allocatedCPUs.IsSubsetOf(restored.GetDefaultCPUSet()), "the interrupted release left the CPUs allocatable")

	// Before the pod sources are ready removeStaleState is a no-op, so a pod
	// admitted on restart can take the CPUs the stale entry still lists.
	restored.SetDefaultCPUSet(restored.GetDefaultCPUSet().Difference(allocatedCPUs))
	restored.SetPodCPUSet("pod2", allocatedCPUs)
	restored.SetCPUSet("pod2", "container", allocatedCPUs)

	mgr := &manager{
		policy:          policy,
		state:           restored,
		lastUpdateState: state.NewMemoryState(logger),
		sourcesReady:    &sourcesReadyStub{},
		activePods: func() []*v1.Pod {
			return []*v1.Pod{{
				ObjectMeta: metav1.ObjectMeta{UID: "pod2"},
				Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "container"}}},
			}}
		},
		containerMap: containermap.NewContainerMap(),
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
}
