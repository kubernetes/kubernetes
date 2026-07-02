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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/utils/cpuset"
)

type operationRetainingPodStatusProvider struct {
	mockPodStatusProvider
	retained map[types.UID]bool
}

func (p *operationRetainingPodStatusProvider) PodOperationMayUsePodResources(uid types.UID) bool {
	return p.retained[uid]
}

func TestRemoveStaleStateRetainsPodOperation(t *testing.T) {
	for _, inventory := range []string{"assignment only", "mapping only", "assignment and mapping"} {
		for _, providerKind := range []string{"pending", "completed", "unaware"} {
			t.Run(inventory+"/"+providerKind, func(t *testing.T) {
				logger, _ := ktesting.NewTestContext(t)
				policy, err := NewStaticPolicy(logger, &topology.CPUTopology{
					NumCPUs: 4, NumSockets: 1, NumCores: 4, NumNUMANodes: 1,
					CPUDetails: topology.CPUDetails{
						0: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
						1: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
						2: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
						3: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					},
				}, 0, cpuset.New(), topologymanager.NewFakeManager(logger), nil)
				require.NoError(t, err)
				assignments, mappings := inventory != "mapping only", inventory != "assignment only"
				current, last := state.NewMemoryState(logger), state.NewMemoryState(logger)
				pool := cpuset.New(0, 3)
				if assignments {
					current.SetCPUSet("pending", "app", cpuset.New(1))
				} else {
					pool = pool.Union(cpuset.New(1))
				}
				current.SetCPUSet("unrelated", "app", cpuset.New(2))
				current.SetDefaultCPUSet(pool)
				last.SetCPUSet("pending", "app", cpuset.New(1))
				last.SetCPUSet("unrelated", "app", cpuset.New(2))
				require.NoError(t, policy.Start(logger, current), "the seeded allocation must be valid static-policy state")
				containers := containermap.NewContainerMap()
				if mappings {
					containers.Add("pending", "app", "pending-container")
				}
				containers.Add("unrelated", "app", "unrelated-container")
				retained := map[types.UID]bool{"pending": providerKind == "pending"}
				var provider status.PodStatusProvider = mockPodStatusProvider{}
				if providerKind != "unaware" {
					provider = &operationRetainingPodStatusProvider{retained: retained}
				}
				manager := &manager{
					policy: policy, state: current, lastUpdateState: last,
					activePods:   func() []*v1.Pod { return nil },
					sourcesReady: &sourcesReadyStub{}, podStatusProvider: provider, containerMap: containers,
				}
				manager.removeStaleState(logger)
				_, allocated := current.GetCPUSet("pending", "app")
				_, updated := last.GetCPUSet("pending", "app")
				_, mappedErr := containers.GetContainerID("pending", "app")
				if providerKind == "pending" {
					require.Equal(t, assignments, allocated)
					require.True(t, updated, "last-update tracking still belongs to the pending operation")
					require.Equal(t, mappings, mappedErr == nil)
					require.Equal(t, pool.Union(cpuset.New(2)), current.GetDefaultCPUSet(), "only unrelated CPUs can return to the shared pool")
				} else {
					require.False(t, allocated)
					require.False(t, updated)
					require.Error(t, mappedErr)
					require.Equal(t, cpuset.New(0, 1, 2, 3), current.GetDefaultCPUSet())
				}
				_, allocated = current.GetCPUSet("unrelated", "app")
				require.False(t, allocated, "an unrelated Pod must still be cleaned up")
				_, err = containers.GetContainerID("unrelated", "app")
				require.Error(t, err)
				retained["pending"] = false
				manager.removeStaleState(logger)
				require.Empty(t, current.GetCPUAssignments())
				require.Empty(t, last.GetCPUAssignments())
				require.Empty(t, containers)
				require.Equal(t, cpuset.New(0, 1, 2, 3), current.GetDefaultCPUSet())
			})
		}
	}
}
