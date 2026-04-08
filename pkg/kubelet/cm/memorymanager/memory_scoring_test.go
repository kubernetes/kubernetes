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

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func memBlock(numa int, sz uint64) state.Block {
	return state.Block{
		NUMAAffinity: []int{numa},
		Type:         v1.ResourceMemory,
		Size:         sz,
	}
}

func symmetricMachineState(allocatable uint64) state.NUMANodeMap {
	return state.NUMANodeMap{
		0: &state.NUMANodeState{
			MemoryMap: map[v1.ResourceName]*state.MemoryTable{
				v1.ResourceMemory: {Allocatable: allocatable},
			},
			Cells: []int{0},
		},
		1: &state.NUMANodeState{
			MemoryMap: map[v1.ResourceName]*state.MemoryTable{
				v1.ResourceMemory: {Allocatable: allocatable},
			},
			Cells: []int{1},
		},
	}
}

func TestGetNUMAUtilizationScores_MemoryScoring(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	mi := returnMachineInfo()
	reserved := systemReservedMemory{
		0: {v1.ResourceMemory: 1 * gb},
		1: {v1.ResourceMemory: 1 * gb},
	}
	staticPol, err := NewPolicyStatic(logger, &mi, reserved, topologymanager.NewFakeManager())
	if err != nil {
		t.Fatal(err)
	}

	m0, err := bitmask.NewBitMask(0)
	if err != nil {
		t.Fatal(err)
	}
	m1, err := bitmask.NewBitMask(1)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("equal utilization returns equal scores", func(t *testing.T) {
		// 2GB/10GB on each NUMA → score 20 vs 20
		st := state.NewMemoryState(logger)
		st.SetMachineState(symmetricMachineState(10 * gb))
		st.SetMemoryAssignments(state.ContainerMemoryAssignments{
			"p1": {"c": {memBlock(0, 2*gb)}},
			"p2": {"c": {memBlock(1, 2*gb)}},
		})
		m := &manager{policy: staticPol, state: st, containerMap: containermap.NewContainerMap()}
		s0, s1 := m.GetNUMAUtilizationScores(m0, m1)
		if s0 != s1 {
			t.Fatalf("got s0=%d s1=%d want equal scores", s0, s1)
		}
	})

	t.Run("higher utilization on candidate returns higher candidate score", func(t *testing.T) {
		// NUMA 0: 1GB/10GB → score 10, NUMA 1: 4GB/10GB → score 40
		st := state.NewMemoryState(logger)
		st.SetMachineState(symmetricMachineState(10 * gb))
		st.SetMemoryAssignments(state.ContainerMemoryAssignments{
			"p1": {"c": {memBlock(0, 1*gb)}},
			"p2": {"c": {memBlock(1, 4*gb)}},
		})
		m := &manager{policy: staticPol, state: st, containerMap: containermap.NewContainerMap()}
		s0, s1 := m.GetNUMAUtilizationScores(m0, m1)
		if s1 <= s0 {
			t.Fatalf("got s0=%d s1=%d want s1 > s0", s0, s1)
		}
	})

	t.Run("higher utilization on current returns higher current score", func(t *testing.T) {
		// NUMA 0: 5GB/10GB → score 50, NUMA 1: 1GB/10GB → score 10
		st := state.NewMemoryState(logger)
		st.SetMachineState(symmetricMachineState(10 * gb))
		st.SetMemoryAssignments(state.ContainerMemoryAssignments{
			"p1": {"c": {memBlock(0, 5*gb)}},
			"p2": {"c": {memBlock(1, 1*gb)}},
		})
		m := &manager{policy: staticPol, state: st, containerMap: containermap.NewContainerMap()}
		s0, s1 := m.GetNUMAUtilizationScores(m0, m1)
		if s0 <= s1 {
			t.Fatalf("got s0=%d s1=%d want s0 > s1", s0, s1)
		}
	})

	t.Run("non-static policy returns zero scores", func(t *testing.T) {
		st := state.NewMemoryState(logger)
		st.SetMachineState(symmetricMachineState(10 * gb))
		st.SetMemoryAssignments(state.ContainerMemoryAssignments{
			"p1": {"c": {memBlock(0, 100)}},
		})
		m := &manager{policy: &mockPolicy{}, state: st, containerMap: containermap.NewContainerMap()}
		s0, s1 := m.GetNUMAUtilizationScores(m0, m1)
		if s0 != 0 || s1 != 0 {
			t.Fatalf("got s0=%d s1=%d want both 0", s0, s1)
		}
	})

	t.Run("asymmetric allocatable changes scores", func(t *testing.T) {
		// NUMA 0 allocatable = 8 GB, NUMA 1 allocatable = 4 GB.
		// Both have 2 GB assigned:
		//   NUMA 0 score = (2*100)/8 = 25
		//   NUMA 1 score = (2*100)/4 = 50
		st := state.NewMemoryState(logger)
		st.SetMachineState(state.NUMANodeMap{
			0: &state.NUMANodeState{
				MemoryMap: map[v1.ResourceName]*state.MemoryTable{
					v1.ResourceMemory: {Allocatable: 8 * gb},
				},
				Cells: []int{0},
			},
			1: &state.NUMANodeState{
				MemoryMap: map[v1.ResourceName]*state.MemoryTable{
					v1.ResourceMemory: {Allocatable: 4 * gb},
				},
				Cells: []int{1},
			},
		})
		st.SetMemoryAssignments(state.ContainerMemoryAssignments{
			"p1": {"c": {memBlock(0, 2*gb)}},
			"p2": {"c": {memBlock(1, 2*gb)}},
		})
		m := &manager{policy: staticPol, state: st, containerMap: containermap.NewContainerMap()}
		s0, s1 := m.GetNUMAUtilizationScores(m0, m1)
		if s1 <= s0 {
			t.Fatalf("got s0=%d s1=%d want s1 > s0 (NUMA 1 more utilized)", s0, s1)
		}
	})
}
