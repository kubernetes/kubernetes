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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	calls    map[types.UID]int
}

func (p *operationRetainingPodStatusProvider) PodOperationMayUsePodResources(uid types.UID) bool {
	if p.calls == nil {
		p.calls = make(map[types.UID]int)
	}
	p.calls[uid]++
	return p.retained[uid]
}

func TestRemoveStaleStateRetainsPodOperation(t *testing.T) {
	for _, podState := range []string{"undiscovered", "active"} {
		for _, inventory := range []string{"assignment only", "mapping only", "assignment and mapping"} {
			for _, providerKind := range []string{"pending", "completed", "unaware"} {
				t.Run(podState+"/"+inventory+"/"+providerKind, func(t *testing.T) {
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
						current.SetMemoryBlocks("pending", "sidecar", block)
						reserved += 2 * gb
						count += 2
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
					operationProvider := &operationRetainingPodStatusProvider{retained: retained}
					var provider status.PodStatusProvider = mockPodStatusProvider{}
					if providerKind != "unaware" {
						provider = operationProvider
					}
					pendingPod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pending"}, Spec: v1.PodSpec{
						Containers: []v1.Container{{Name: "app"}, {Name: "sidecar"}},
					}}
					var activePods []*v1.Pod
					if podState == "active" {
						activePods = []*v1.Pod{pendingPod}
					}
					manager := &manager{
						policy: policy, state: current, activePods: func() []*v1.Pod { return activePods },
						sourcesReady: &sourcesReadyStub{}, podStatusProvider: provider, containerMap: containers,
					}
					pendingAssignments := current.GetMemoryAssignments()["pending"]
					if podState == "active" {
						manager.removeStaleState(logger)
						require.Zero(t, operationProvider.calls["pending"], "active containers do not need an operation lookup")
						require.Equal(t, pendingAssignments, current.GetMemoryAssignments()["pending"])
						_, err := containers.GetContainerID("pending", "app")
						require.Equal(t, mappings, err == nil)
						pendingPod.Spec.Containers = []v1.Container{{Name: "replacement"}}
					}
					manager.removeStaleState(logger)
					_, mappedErr := containers.GetContainerID("pending", "app")
					memory := current.GetMachineState()[0].MemoryMap[v1.ResourceMemory]
					if providerKind == "pending" {
						require.Equal(t, pendingAssignments, current.GetMemoryAssignments()["pending"])
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
					if providerKind != "unaware" {
						wantCalls := 1
						if providerKind == "pending" && assignments && mappings {
							wantCalls++
						}
						require.Equal(t, wantCalls, operationProvider.calls["pending"], "assignment cleanup checks ownership once per Pod; stale mappings check before removal")
					}
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
}
