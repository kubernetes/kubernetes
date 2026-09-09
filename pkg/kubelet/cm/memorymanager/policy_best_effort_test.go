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
	"context"
	"reflect"
	"testing"

	cadvisorapi "github.com/google/cadvisor/lib/model"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"

	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

// fakeBestEffortAffinity is a topologymanager.AuthoritativeStore that returns a
// preset hint.
type fakeBestEffortAffinity struct {
	hint              topologymanager.TopologyHint
	hintAuthoritative bool
}

func (f *fakeBestEffortAffinity) GetAffinity(_ klog.Logger, podUID, containerName string) topologymanager.TopologyHint {
	return f.hint
}
func (f *fakeBestEffortAffinity) GetPolicy() topologymanager.Policy { return nil }
func (f *fakeBestEffortAffinity) Name() string                      { return "fake" }

// IsHintAuthoritative mirrors the Windows cpuFollowingStore: when the container
// follows the CPU manager's NUMA decision, its hint is authoritative and the
// memory manager must not extend it to additional NUMA nodes.
func (f *fakeBestEffortAffinity) IsHintAuthoritative(_, _ string) bool { return f.hintAuthoritative }

// bestEffortTestMachineState returns a fresh 2-NUMA-node state: node 0 has only
// 1Gi free (too little for the 4Gi request), node 1 has 10Gi free.
func bestEffortTestMachineState() state.NUMANodeMap {
	return state.NUMANodeMap{
		0: &state.NUMANodeState{
			Cells: []int{0},
			MemoryMap: map[v1.ResourceName]*state.MemoryTable{
				v1.ResourceMemory: {
					Allocatable:    1 * gb,
					Free:           1 * gb,
					Reserved:       0,
					SystemReserved: 512 * mb,
					TotalMemSize:   1*gb + 512*mb,
				},
			},
		},
		1: &state.NUMANodeState{
			Cells: []int{1},
			MemoryMap: map[v1.ResourceName]*state.MemoryTable{
				v1.ResourceMemory: {
					Allocatable:    10 * gb,
					Free:           10 * gb,
					Reserved:       0,
					SystemReserved: 0,
					TotalMemSize:   10 * gb,
				},
			},
		},
	}
}

// TestBestEffortPolicyAllocateFollowsCPUManager verifies that the Windows
// BestEffort policy skips hint extension when the container follows the CPU
// manager's NUMA decision (it owns exclusive CPUs), and otherwise extends like
// the static policy.
func TestBestEffortPolicyAllocateFollowsCPUManager(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	machineInfo := &cadvisorapi.MachineInfo{
		Topology: []cadvisorapi.Node{
			{Id: 0, Memory: 1*gb + 512*mb},
			{Id: 1, Memory: 10 * gb},
		},
	}
	systemReserved := systemReservedMemory{
		0: map[v1.ResourceName]uint64{v1.ResourceMemory: 512 * mb},
	}
	// Guaranteed container requesting 4Gi; the affinity hint is node 0, which has
	// only 1Gi free, so the hint does not satisfy the request and the extend
	// branch is reachable.
	requirements := &v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("2"),
			v1.ResourceMemory: resource.MustParse("4Gi"),
		},
		Limits: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("2"),
			v1.ResourceMemory: resource.MustParse("4Gi"),
		},
	}

	testCases := []struct {
		name             string
		hasExclusiveCPUs bool
		expectedNUMA     []int
	}{
		{
			name:             "has exclusive CPUs: follows CPU manager, no hint extension",
			hasExclusiveCPUs: true,
			expectedNUMA:     []int{0},
		},
		{
			name:             "no exclusive CPUs: extends like the static policy",
			hasExclusiveCPUs: false,
			expectedNUMA:     []int{0, 1},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			affinity := &fakeBestEffortAffinity{
				hint:              topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: false},
				hintAuthoritative: tc.hasExclusiveCPUs,
			}
			p, err := NewPolicyBestEffort(logger, machineInfo, systemReserved, affinity)
			if err != nil {
				t.Fatalf("NewPolicyBestEffort() failed: %v", err)
			}

			s := state.NewMemoryState(logger)
			s.SetMachineState(bestEffortTestMachineState())

			pod := getPod("pod1", "container1", requirements)
			if err := p.Allocate(context.Background(), s, pod, &pod.Spec.Containers[0], lifecycle.AddOperation); err != nil {
				t.Fatalf("Allocate() failed: %v", err)
			}

			blocks := s.GetMemoryBlocks("pod1", "container1")
			if len(blocks) != 1 {
				t.Fatalf("expected exactly 1 memory block, got %d: %+v", len(blocks), blocks)
			}
			if got := blocks[0].NUMAAffinity; !reflect.DeepEqual(got, tc.expectedNUMA) {
				t.Errorf("expected memory block on NUMA nodes %v, got %v", tc.expectedNUMA, got)
			}
		})
	}
}
