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
				NumCores: 8,
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
				NumCores: 4,
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
				NumCores: 12,
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
				NumCores: 8,
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
				NumCores: 8,
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
