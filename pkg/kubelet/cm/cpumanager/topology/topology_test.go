/*
Copyright 2017 The Kubernetes Authors.

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

package topology

import (
	"reflect"
	"testing"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

func Test_Discover(t *testing.T) {

	tests := []struct {
		name        string
		machineInfo cadvisorapi.MachineInfo
		want        *CPUTopology
		wantErr     bool
	}{
		{
			name: "FailNumCores",
			machineInfo: cadvisorapi.MachineInfo{
				NumCores: 0,
			},
			want:    &CPUTopology{},
			wantErr: true,
		},
		{
			name: "OneSocketHT",
			machineInfo: cadvisorapi.MachineInfo{
				NumCores:   8,
				NumSockets: 1,
				Topology: []cadvisorapi.Node{
					{Id: 0,
						Cores: []cadvisorapi.Core{
							{SocketID: 0, Id: 0, Threads: []int{0, 4}},
							{SocketID: 0, Id: 1, Threads: []int{1, 5}},
							{SocketID: 0, Id: 2, Threads: []int{2, 6}},
							{SocketID: 0, Id: 3, Threads: []int{3, 7}},
						},
					},
				},
			},
			want: &CPUTopology{
				NumCPUs:    8,
				NumSockets: 1,
				NumCores:   4,
				CPUDetails: map[int]CPUInfo{
					0: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					1: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					2: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					3: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					4: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					5: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					6: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					7: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
				},
			},
			wantErr: false,
		},
		{
			name: "DualSocketNoHT",
			machineInfo: cadvisorapi.MachineInfo{
				NumCores:   4,
				NumSockets: 2,
				Topology: []cadvisorapi.Node{
					{Id: 0,
						Cores: []cadvisorapi.Core{
							{SocketID: 0, Id: 0, Threads: []int{0}},
							{SocketID: 0, Id: 2, Threads: []int{2}},
						},
					},
					{Id: 1,
						Cores: []cadvisorapi.Core{
							{SocketID: 1, Id: 1, Threads: []int{1}},
							{SocketID: 1, Id: 3, Threads: []int{3}},
						},
					},
				},
			},
			want: &CPUTopology{
				NumCPUs:    4,
				NumSockets: 2,
				NumCores:   4,
				CPUDetails: map[int]CPUInfo{
					0: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					1: {CoreID: 1, SocketID: 1, NUMANodeID: 1},
					2: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					3: {CoreID: 3, SocketID: 1, NUMANodeID: 1},
				},
			},
			wantErr: false,
		},
		{
			name: "DualSocketHT - non unique Core'ID's",
			machineInfo: cadvisorapi.MachineInfo{
				NumCores:   12,
				NumSockets: 2,
				Topology: []cadvisorapi.Node{
					{Id: 0,
						Cores: []cadvisorapi.Core{
							{SocketID: 0, Id: 0, Threads: []int{0, 6}},
							{SocketID: 0, Id: 1, Threads: []int{1, 7}},
							{SocketID: 0, Id: 2, Threads: []int{2, 8}},
						},
					},
					{Id: 1,
						Cores: []cadvisorapi.Core{
							{SocketID: 1, Id: 0, Threads: []int{3, 9}},
							{SocketID: 1, Id: 1, Threads: []int{4, 10}},
							{SocketID: 1, Id: 2, Threads: []int{5, 11}},
						},
					},
				},
			},
			want: &CPUTopology{
				NumCPUs:    12,
				NumSockets: 2,
				NumCores:   6,
				CPUDetails: map[int]CPUInfo{
					0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					1:  {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					3:  {CoreID: 3, SocketID: 1, NUMANodeID: 1},
					4:  {CoreID: 4, SocketID: 1, NUMANodeID: 1},
					5:  {CoreID: 5, SocketID: 1, NUMANodeID: 1},
					6:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					7:  {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					8:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					9:  {CoreID: 3, SocketID: 1, NUMANodeID: 1},
					10: {CoreID: 4, SocketID: 1, NUMANodeID: 1},
					11: {CoreID: 5, SocketID: 1, NUMANodeID: 1},
				},
			},
			wantErr: false,
		},
		{
			name: "OneSocketHT fail",
			machineInfo: cadvisorapi.MachineInfo{
				NumCores:   8,
				NumSockets: 1,
				Topology: []cadvisorapi.Node{
					{Id: 0,
						Cores: []cadvisorapi.Core{
							{SocketID: 0, Id: 0, Threads: []int{0, 4}},
							{SocketID: 0, Id: 1, Threads: []int{1, 5}},
							{SocketID: 0, Id: 2, Threads: []int{2, 2}}, // Wrong case - should fail here
							{SocketID: 0, Id: 3, Threads: []int{3, 7}},
						},
					},
				},
			},
			want:    &CPUTopology{},
			wantErr: true,
		},
		{
			name: "OneSocketHT fail",
			machineInfo: cadvisorapi.MachineInfo{
				NumCores:   8,
				NumSockets: 1,
				Topology: []cadvisorapi.Node{
					{Id: 0,
						Cores: []cadvisorapi.Core{
							{SocketID: 0, Id: 0, Threads: []int{0, 4}},
							{SocketID: 0, Id: 1, Threads: []int{1, 5}},
							{SocketID: 0, Id: 2, Threads: []int{2, 6}},
							{SocketID: 0, Id: 3, Threads: []int{}}, // Wrong case - should fail here
						},
					},
				},
			},
			want:    &CPUTopology{},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Discover(&tt.machineInfo)
			if err != nil {
				if tt.wantErr {
					t.Logf("Discover() expected error = %v", err)
				} else {
					t.Errorf("Discover() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Discover() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsKeepOnly(t *testing.T) {

	var details CPUDetails
	details = map[int]CPUInfo{
		0: {},
		1: {},
		2: {},
	}

	tests := []struct {
		name string
		cpus cpuset.CPUSet
		want CPUDetails
	}{{
		name: "cpus is in CPUDetails.",
		cpus: cpuset.NewCPUSet(0, 1),
		want: map[int]CPUInfo{
			0: {},
			1: {},
		},
	}, {
		name: "cpus is not in CPUDetails.",
		cpus: cpuset.NewCPUSet(3),
		want: CPUDetails{},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := details.KeepOnly(tt.cpus)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("KeepOnly() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsNUMANodes(t *testing.T) {

	tests := []struct {
		name    string
		details CPUDetails
		want    cpuset.CPUSet
	}{{
		name: "Get CPUset of NUMANode IDs",
		details: map[int]CPUInfo{
			0: {NUMANodeID: 0},
			1: {NUMANodeID: 0},
			2: {NUMANodeID: 1},
			3: {NUMANodeID: 1},
		},
		want: cpuset.NewCPUSet(0, 1),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.details.NUMANodes()
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NUMANodes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsNUMANodesInSockets(t *testing.T) {

	var details1 CPUDetails
	details1 = map[int]CPUInfo{
		0: {SocketID: 0, NUMANodeID: 0},
		1: {SocketID: 1, NUMANodeID: 0},
		2: {SocketID: 2, NUMANodeID: 1},
		3: {SocketID: 3, NUMANodeID: 1},
	}

	// poorly designed mainboards
	var details2 CPUDetails
	details2 = map[int]CPUInfo{
		0: {SocketID: 0, NUMANodeID: 0},
		1: {SocketID: 0, NUMANodeID: 1},
		2: {SocketID: 1, NUMANodeID: 2},
		3: {SocketID: 1, NUMANodeID: 3},
	}

	tests := []struct {
		name    string
		details CPUDetails
		ids     []int
		want    cpuset.CPUSet
	}{{
		name:    "Socket IDs is in CPUDetails.",
		details: details1,
		ids:     []int{0, 1, 2},
		want:    cpuset.NewCPUSet(0, 1),
	}, {
		name:    "Socket IDs is not in CPUDetails.",
		details: details1,
		ids:     []int{4},
		want:    cpuset.NewCPUSet(),
	}, {
		name:    "Socket IDs is in CPUDetails. (poorly designed mainboards)",
		details: details2,
		ids:     []int{0},
		want:    cpuset.NewCPUSet(0, 1),
	}, {
		name:    "Socket IDs is not in CPUDetails. (poorly designed mainboards)",
		details: details2,
		ids:     []int{3},
		want:    cpuset.NewCPUSet(),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.details.NUMANodesInSockets(tt.ids...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NUMANodesInSockets() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsSockets(t *testing.T) {

	tests := []struct {
		name    string
		details CPUDetails
		want    cpuset.CPUSet
	}{{
		name: "Get CPUset of Socket IDs",
		details: map[int]CPUInfo{
			0: {SocketID: 0},
			1: {SocketID: 0},
			2: {SocketID: 1},
			3: {SocketID: 1},
		},
		want: cpuset.NewCPUSet(0, 1),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.details.Sockets()
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Sockets() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsCPUsInSockets(t *testing.T) {

	var details CPUDetails
	details = map[int]CPUInfo{
		0: {SocketID: 0},
		1: {SocketID: 0},
		2: {SocketID: 1},
		3: {SocketID: 2},
	}

	tests := []struct {
		name string
		ids  []int
		want cpuset.CPUSet
	}{{
		name: "Socket IDs is in CPUDetails.",
		ids:  []int{0, 1},
		want: cpuset.NewCPUSet(0, 1, 2),
	}, {
		name: "Socket IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.NewCPUSet(),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := details.CPUsInSockets(tt.ids...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CPUsInSockets() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsSocketsInNUMANodes(t *testing.T) {

	var details CPUDetails
	details = map[int]CPUInfo{
		0: {NUMANodeID: 0, SocketID: 0},
		1: {NUMANodeID: 0, SocketID: 1},
		2: {NUMANodeID: 1, SocketID: 2},
		3: {NUMANodeID: 2, SocketID: 3},
	}

	tests := []struct {
		name string
		ids  []int
		want cpuset.CPUSet
	}{{
		name: "NUMANodes IDs is in CPUDetails.",
		ids:  []int{0, 1},
		want: cpuset.NewCPUSet(0, 1, 2),
	}, {
		name: "NUMANodes IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.NewCPUSet(),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := details.SocketsInNUMANodes(tt.ids...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("SocketsInNUMANodes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsCores(t *testing.T) {

	tests := []struct {
		name    string
		details CPUDetails
		want    cpuset.CPUSet
	}{{
		name: "Get CPUset of Cores",
		details: map[int]CPUInfo{
			0: {CoreID: 0},
			1: {CoreID: 0},
			2: {CoreID: 1},
			3: {CoreID: 1},
		},
		want: cpuset.NewCPUSet(0, 1),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.details.Cores()
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Cores() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsCoresInNUMANodes(t *testing.T) {

	var details CPUDetails
	details = map[int]CPUInfo{
		0: {NUMANodeID: 0, CoreID: 0},
		1: {NUMANodeID: 0, CoreID: 1},
		2: {NUMANodeID: 1, CoreID: 2},
		3: {NUMANodeID: 2, CoreID: 3},
	}

	tests := []struct {
		name string
		ids  []int
		want cpuset.CPUSet
	}{{
		name: "NUMANodes IDs is in CPUDetails.",
		ids:  []int{0, 1},
		want: cpuset.NewCPUSet(0, 1, 2),
	}, {
		name: "NUMANodes IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.NewCPUSet(),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := details.CoresInNUMANodes(tt.ids...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CoresInNUMANodes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsCoresInSockets(t *testing.T) {

	var details CPUDetails
	details = map[int]CPUInfo{
		0: {SocketID: 0, CoreID: 0},
		1: {SocketID: 0, CoreID: 1},
		2: {SocketID: 1, CoreID: 2},
		3: {SocketID: 2, CoreID: 3},
	}

	tests := []struct {
		name string
		ids  []int
		want cpuset.CPUSet
	}{{
		name: "Socket IDs is in CPUDetails.",
		ids:  []int{0, 1},
		want: cpuset.NewCPUSet(0, 1, 2),
	}, {
		name: "Socket IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.NewCPUSet(),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := details.CoresInSockets(tt.ids...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CoresInSockets() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsCPUs(t *testing.T) {

	tests := []struct {
		name    string
		details CPUDetails
		want    cpuset.CPUSet
	}{{
		name: "Get CPUset of CPUs",
		details: map[int]CPUInfo{
			0: {},
			1: {},
		},
		want: cpuset.NewCPUSet(0, 1),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.details.CPUs()
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CPUs() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsCPUsInNUMANodes(t *testing.T) {

	var details CPUDetails
	details = map[int]CPUInfo{
		0: {NUMANodeID: 0},
		1: {NUMANodeID: 0},
		2: {NUMANodeID: 1},
		3: {NUMANodeID: 2},
	}

	tests := []struct {
		name string
		ids  []int
		want cpuset.CPUSet
	}{{
		name: "NUMANode IDs is in CPUDetails.",
		ids:  []int{0, 1},
		want: cpuset.NewCPUSet(0, 1, 2),
	}, {
		name: "NUMANode IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.NewCPUSet(),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := details.CPUsInNUMANodes(tt.ids...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CPUsInNUMANodes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUDetailsCPUsInCores(t *testing.T) {

	var details CPUDetails
	details = map[int]CPUInfo{
		0: {CoreID: 0},
		1: {CoreID: 0},
		2: {CoreID: 1},
		3: {CoreID: 2},
	}

	tests := []struct {
		name string
		ids  []int
		want cpuset.CPUSet
	}{{
		name: "Core IDs is in CPUDetails.",
		ids:  []int{0, 1},
		want: cpuset.NewCPUSet(0, 1, 2),
	}, {
		name: "Core IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.NewCPUSet(),
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := details.CPUsInCores(tt.ids...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CPUsInCores() = %v, want %v", got, tt.want)
			}
		})
	}
}
