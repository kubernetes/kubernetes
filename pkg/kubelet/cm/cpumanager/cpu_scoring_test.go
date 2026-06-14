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

	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/cpuset"
)

// returnMachineInfo topology (from topology_hints_test.go):
//   NUMA 0: CPUs {0,1,2,6,7,8}   (6 CPUs)
//   NUMA 1: CPUs {3,4,5,9,10,11} (6 CPUs)

func TestGetNUMAUtilizationScores_CPUScoring(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	mi := returnMachineInfo()
	topo, err := topology.Discover(logger, &mi)
	if err != nil {
		t.Fatal(err)
	}
	staticPol, err := NewStaticPolicy(logger, topo, 0, cpuset.New(), topologymanager.NewFakeManager(), nil)
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
		// 1/6 on each NUMA → score 16 vs 16
		st := &mockState{assignments: state.ContainerCPUAssignments{
			"p1": {"c": mustParseCPUSet(t, "0")},
			"p2": {"c": mustParseCPUSet(t, "3")},
		}}
		m := &manager{policy: staticPol, state: st}
		s0, s1 := m.GetNUMAUtilizationScores(m0, m1)
		if s0 != s1 {
			t.Fatalf("got s0=%d s1=%d want equal scores", s0, s1)
		}
	})

	t.Run("higher utilization on candidate returns higher candidate score", func(t *testing.T) {
		// NUMA 0: 1/6 → score 16, NUMA 1: 3/6 → score 50
		st := &mockState{assignments: state.ContainerCPUAssignments{
			"p1": {"c": mustParseCPUSet(t, "0")},
			"p2": {"c": mustParseCPUSet(t, "3")},
			"p3": {"c": mustParseCPUSet(t, "4")},
			"p4": {"c": mustParseCPUSet(t, "5")},
		}}
		m := &manager{policy: staticPol, state: st}
		s0, s1 := m.GetNUMAUtilizationScores(m0, m1)
		if s1 <= s0 {
			t.Fatalf("got s0=%d s1=%d want s1 > s0", s0, s1)
		}
	})

	t.Run("higher utilization on current returns higher current score", func(t *testing.T) {
		// NUMA 0: 3/6 → score 50, NUMA 1: 1/6 → score 16
		st := &mockState{assignments: state.ContainerCPUAssignments{
			"p1": {"c": mustParseCPUSet(t, "0")},
			"p2": {"c": mustParseCPUSet(t, "1")},
			"p3": {"c": mustParseCPUSet(t, "2")},
			"p4": {"c": mustParseCPUSet(t, "3")},
		}}
		m := &manager{policy: staticPol, state: st}
		s0, s1 := m.GetNUMAUtilizationScores(m0, m1)
		if s0 <= s1 {
			t.Fatalf("got s0=%d s1=%d want s0 > s1", s0, s1)
		}
	})

	t.Run("non-static policy returns zero scores", func(t *testing.T) {
		st := &mockState{assignments: state.ContainerCPUAssignments{
			"p1": {"c": mustParseCPUSet(t, "0")},
		}}
		m := &manager{policy: &mockPolicy{}, state: st}
		s0, s1 := m.GetNUMAUtilizationScores(m0, m1)
		if s0 != 0 || s1 != 0 {
			t.Fatalf("got s0=%d s1=%d want both 0", s0, s1)
		}
	})

	t.Run("asymmetric reservation changes scores", func(t *testing.T) {
		// Reserve CPUs {0,6} on NUMA 0 → allocatable: NUMA 0 = 4, NUMA 1 = 6.
		// Assign 2 exclusive CPUs on each: NUMA 0 score = (2*100)/4 = 50,
		// NUMA 1 score = (2*100)/6 = 33.
		asymPol, err := NewStaticPolicy(logger, topo, 2, cpuset.New(0, 6), topologymanager.NewFakeManager(), nil)
		if err != nil {
			t.Fatal(err)
		}
		st := &mockState{assignments: state.ContainerCPUAssignments{
			"p1": {"c": mustParseCPUSet(t, "1")},
			"p2": {"c": mustParseCPUSet(t, "2")},
			"p3": {"c": mustParseCPUSet(t, "3")},
			"p4": {"c": mustParseCPUSet(t, "4")},
		}}
		m := &manager{policy: asymPol, state: st}
		s0, s1 := m.GetNUMAUtilizationScores(m0, m1)
		if s0 <= s1 {
			t.Fatalf("got s0=%d s1=%d want s0 > s1 (NUMA 0 more utilized due to reservation)", s0, s1)
		}
	})
}
