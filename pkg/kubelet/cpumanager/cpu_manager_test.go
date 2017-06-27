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

package cpumanager

import (
	"reflect"
	"testing"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/topo"
)

func Test_discoverTopology(t *testing.T) {

	tests := []struct {
		name    string
		args    *cadvisorapi.MachineInfo
		want    *topo.CPUTopology
		wantErr bool
	}{
		{
			name: "FailNumCores",
			args: &cadvisorapi.MachineInfo{
				NumCores: 0,
			},
			want:    &topo.CPUTopology{},
			wantErr: true,
		},
		{
			name: "OneSocketHT",
			args: &cadvisorapi.MachineInfo{
				NumCores: 8,
				Topology: []cadvisorapi.Node{
					{Id: 0,
						Cores: []cadvisorapi.Core{
							{Id: 0, Threads: []int{0, 4}},
							{Id: 1, Threads: []int{1, 5}},
							{Id: 2, Threads: []int{2, 6}},
							{Id: 3, Threads: []int{3, 7}},
						},
					},
				},
			},
			want: &topo.CPUTopology{
				NumCPUs:        8,
				NumSockets:     1,
				NumCores:       4,
				HyperThreading: true,
				CPUtopoDetails: map[int]topo.CPUInfo{
					0: {CoreId: 0, SocketId: 0},
					1: {CoreId: 1, SocketId: 0},
					2: {CoreId: 2, SocketId: 0},
					3: {CoreId: 3, SocketId: 0},
					4: {CoreId: 0, SocketId: 0},
					5: {CoreId: 1, SocketId: 0},
					6: {CoreId: 2, SocketId: 0},
					7: {CoreId: 3, SocketId: 0},
				},
			},
			wantErr: false,
		},
		{
			name: "DualSocketNoHT",
			args: &cadvisorapi.MachineInfo{
				NumCores: 4,
				Topology: []cadvisorapi.Node{
					{Id: 0,
						Cores: []cadvisorapi.Core{
							{Id: 0, Threads: []int{0}},
							{Id: 2, Threads: []int{2}},
						},
					},
					{Id: 1,
						Cores: []cadvisorapi.Core{
							{Id: 1, Threads: []int{1}},
							{Id: 3, Threads: []int{3}},
						},
					},
				},
			},
			want: &topo.CPUTopology{
				NumCPUs:        4,
				NumSockets:     2,
				NumCores:       4,
				HyperThreading: false,
				CPUtopoDetails: map[int]topo.CPUInfo{
					0: {CoreId: 0, SocketId: 0},
					1: {CoreId: 1, SocketId: 1},
					2: {CoreId: 2, SocketId: 0},
					3: {CoreId: 3, SocketId: 1},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := discoverTopology(tt.args)
			if err != nil {
				if tt.wantErr {
					t.Logf("discoverTopology() expected error = %v", err)
				} else {
					t.Errorf("discoverTopology() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("discoverTopology() = %v, want %v", got, tt.want)
			}
		})
	}
}
