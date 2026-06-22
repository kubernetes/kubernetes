//go:build windows

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

package cm

import (
	"testing"

	cadvisorapi "github.com/google/cadvisor/info/v1"

	"k8s.io/utils/cpuset"

	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

// cpuFollowingFakeCPUManager implements cpumanager.Manager, but only GetExclusiveCPUs is
// meaningful. The remaining methods are satisfied by the embedded (nil) interface
// and must not be called by the code under test (they would panic).
type cpuFollowingFakeCPUManager struct {
	cpumanager.Manager
	exclusive map[string]cpuset.CPUSet
}

func (f *cpuFollowingFakeCPUManager) GetExclusiveCPUs(podUID, containerName string) cpuset.CPUSet {
	if cpus, ok := f.exclusive[podUID+"/"+containerName]; ok {
		return cpus
	}
	return cpuset.New()
}

// cpuFollowingFakeStore is a minimal topologymanager.Store that always returns a preset hint.
type cpuFollowingFakeStore struct {
	base topologymanager.TopologyHint
}

func (f *cpuFollowingFakeStore) GetAffinity(podUID, containerName string) topologymanager.TopologyHint {
	return f.base
}
func (f *cpuFollowingFakeStore) GetPolicy() topologymanager.Policy { return nil }
func (f *cpuFollowingFakeStore) Name() string                      { return "fake" }

func cpuFollowingMustBitMask(t *testing.T, bits ...int) bitmask.BitMask {
	t.Helper()
	m, err := bitmask.NewBitMask(bits...)
	if err != nil {
		t.Fatalf("NewBitMask(%v): %v", bits, err)
	}
	return m
}

// cpuFollowingTestMachineInfo describes 2 NUMA nodes: CPUs 0-3 on node 0, CPUs 4-7 on node 1.
func cpuFollowingTestMachineInfo() *cadvisorapi.MachineInfo {
	return &cadvisorapi.MachineInfo{
		Topology: []cadvisorapi.Node{
			{Id: 0, Cores: []cadvisorapi.Core{
				{Id: 0, Threads: []int{0, 1}},
				{Id: 1, Threads: []int{2, 3}},
			}},
			{Id: 1, Cores: []cadvisorapi.Core{
				{Id: 2, Threads: []int{4, 5}},
				{Id: 3, Threads: []int{6, 7}},
			}},
		},
	}
}

func TestCPUFollowingStoreGetAffinity(t *testing.T) {
	// base is the Topology Manager's own hint; deliberately a different node ({1})
	// than the exclusive-CPU-derived masks so "follow" and "fall back" are
	// distinguishable.
	baseHint := topologymanager.TopologyHint{NUMANodeAffinity: cpuFollowingMustBitMask(t, 1), Preferred: true}

	testCases := []struct {
		name          string
		exclusiveCPUs cpuset.CPUSet
		expectFollow  bool  // true → expect derived mask; false → expect base returned
		expectedNodes []int // only used when expectFollow is true
	}{
		{
			name:          "exclusive CPUs on a single node follows that node",
			exclusiveCPUs: cpuset.New(0, 1),
			expectFollow:  true,
			expectedNodes: []int{0},
		},
		{
			name:          "exclusive CPUs spanning two nodes follows both",
			exclusiveCPUs: cpuset.New(1, 4),
			expectFollow:  true,
			expectedNodes: []int{0, 1},
		},
		{
			name:          "exclusive CPUs partially mapped follows only the mapped node(s)",
			exclusiveCPUs: cpuset.New(0, 99), // 0 -> node 0; 99 has no NUMA node and is dropped
			expectFollow:  true,
			expectedNodes: []int{0},
		},
		{
			name:          "no exclusive CPUs (e.g. CPU policy none) falls back to base",
			exclusiveCPUs: cpuset.New(),
			expectFollow:  false,
		},
		{
			name:          "exclusive CPUs not mapped to any NUMA node falls back to base",
			exclusiveCPUs: cpuset.New(99),
			expectFollow:  false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cpuMgr := &cpuFollowingFakeCPUManager{exclusive: map[string]cpuset.CPUSet{"pod/ctr": tc.exclusiveCPUs}}
			store := newCPUFollowingStore(&cpuFollowingFakeStore{base: baseHint}, cpuMgr, cpuFollowingTestMachineInfo())

			got := store.GetAffinity("pod", "ctr")

			if !tc.expectFollow {
				if !got.NUMANodeAffinity.IsEqual(baseHint.NUMANodeAffinity) {
					t.Errorf("expected fall back to base %v, got %v", baseHint.NUMANodeAffinity, got.NUMANodeAffinity)
				}
				return
			}

			want := cpuFollowingMustBitMask(t, tc.expectedNodes...)
			if got.NUMANodeAffinity == nil || !got.NUMANodeAffinity.IsEqual(want) {
				t.Errorf("expected NUMA affinity %v, got %v", want, got.NUMANodeAffinity)
			}
			if got.Preferred != baseHint.Preferred {
				t.Errorf("expected Preferred %v (carried from base), got %v", baseHint.Preferred, got.Preferred)
			}
		})
	}
}

func TestCPUFollowingStoreHasExclusiveCPUs(t *testing.T) {
	cpuMgr := &cpuFollowingFakeCPUManager{exclusive: map[string]cpuset.CPUSet{
		"pod/with":    cpuset.New(0, 1),
		"pod/without": cpuset.New(),
	}}
	store := newCPUFollowingStore(&cpuFollowingFakeStore{}, cpuMgr, cpuFollowingTestMachineInfo())

	if !store.HasExclusiveCPUs("pod", "with") {
		t.Errorf("expected HasExclusiveCPUs=true for a container with exclusive CPUs")
	}
	if store.HasExclusiveCPUs("pod", "without") {
		t.Errorf("expected HasExclusiveCPUs=false for a container with an empty CPU set")
	}
	if store.HasExclusiveCPUs("pod", "missing") {
		t.Errorf("expected HasExclusiveCPUs=false for an unknown container")
	}
}

func TestCPUFollowingStoreNumaMaskForCPUs(t *testing.T) {
	store := newCPUFollowingStore(&cpuFollowingFakeStore{}, &cpuFollowingFakeCPUManager{}, cpuFollowingTestMachineInfo())

	testCases := []struct {
		name     string
		cpus     cpuset.CPUSet
		wantNil  bool
		wantBits []int
	}{
		{name: "cpus within a single node", cpus: cpuset.New(0, 2), wantBits: []int{0}},
		{name: "cpus across two nodes", cpus: cpuset.New(3, 6), wantBits: []int{0, 1}},
		{name: "partially mapped cpus drop the unmapped one", cpus: cpuset.New(2, 99), wantBits: []int{0}},
		{name: "unmapped cpu yields nil", cpus: cpuset.New(99), wantNil: true},
		{name: "empty set yields nil", cpus: cpuset.New(), wantNil: true},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := store.numaMaskForCPUs(tc.cpus)
			if tc.wantNil {
				if got != nil {
					t.Errorf("expected nil mask, got %v", got)
				}
				return
			}
			want := cpuFollowingMustBitMask(t, tc.wantBits...)
			if got == nil || !got.IsEqual(want) {
				t.Errorf("expected mask %v, got %v", want, got)
			}
		})
	}
}
