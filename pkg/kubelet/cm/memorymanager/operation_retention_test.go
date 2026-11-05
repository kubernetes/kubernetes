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

package memorymanager

import (
	"testing"

	cadvisorapi "github.com/google/cadvisor/lib/model"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/status"
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
				machineInfo := &cadvisorapi.MachineInfo{Topology: []cadvisorapi.Node{{Id: 0, Memory: 4 * gb}}}
				policy, err := NewPolicyStatic(logger, machineInfo, systemReservedMemory{0: {v1.ResourceMemory: gb}}, topologymanager.NewFakeManager(logger))
				require.NoError(t, err)
				assignments, mappings := inventory != "mapping only", inventory != "assignment only"
				current := state.NewMemoryState(logger)
				block := []state.Block{{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: gb}}
				reserved, count := uint64(gb), 1
				if assignments {
					current.SetMemoryBlocks("pending", "app", block)
					reserved += gb
					count++
				}
				current.SetMemoryBlocks("unrelated", "app", block)
				current.SetMachineState(state.NUMANodeMap{0: {
					Cells: []int{0}, NumberOfAssignments: count,
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{v1.ResourceMemory: {
						TotalMemSize: 4 * gb, SystemReserved: gb, Allocatable: 3 * gb, Free: 3*gb - reserved, Reserved: reserved,
					}},
				}})
				require.NoError(t, policy.Start(logger, current), "the seeded reservation must be valid static-policy state")
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
					policy: policy, state: current, activePods: func() []*v1.Pod { return nil },
					sourcesReady: &sourcesReadyStub{}, podStatusProvider: provider, containerMap: containers,
				}
				manager.removeStaleState(logger)
				_, mappedErr := containers.GetContainerID("pending", "app")
				memory := current.GetMachineState()[0].MemoryMap[v1.ResourceMemory]
				if providerKind == "pending" {
					if assignments {
						require.Equal(t, block, current.GetMemoryBlocks("pending", "app"))
					} else {
						require.Empty(t, current.GetMemoryBlocks("pending", "app"))
					}
					require.Equal(t, mappings, mappedErr == nil)
					require.Equal(t, reserved-gb, memory.Reserved, "only unrelated reservations may be released")
					require.Equal(t, uint64(3*gb)-(reserved-gb), memory.Free)
				} else {
					require.Empty(t, current.GetMemoryBlocks("pending", "app"))
					require.Error(t, mappedErr)
					require.Zero(t, memory.Reserved)
					require.Equal(t, uint64(3*gb), memory.Free)
				}
				require.Empty(t, current.GetMemoryBlocks("unrelated", "app"), "an unrelated Pod must still be cleaned up")
				_, err = containers.GetContainerID("unrelated", "app")
				require.Error(t, err)
				retained["pending"] = false
				manager.removeStaleState(logger)
				require.Empty(t, current.GetMemoryAssignments())
				require.Empty(t, containers)
				memory = current.GetMachineState()[0].MemoryMap[v1.ResourceMemory]
				require.Zero(t, memory.Reserved)
				require.Equal(t, uint64(3*gb), memory.Free)
			})
		}
	}
}
