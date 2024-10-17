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
	"github.com/google/go-cmp/cmp"
	"k8s.io/utils/cpuset"
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
				NumCPUs:      8,
				NumSockets:   1,
				NumCores:     4,
				NumNUMANodes: 1,
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
			// dual xeon gold 6230
			name: "DualSocketMultiNumaPerSocketHT",
			machineInfo: cadvisorapi.MachineInfo{
				NumCores:   80,
				NumSockets: 2,
				Topology: []cadvisorapi.Node{
					{Id: 0,
						Cores: []cadvisorapi.Core{
							{SocketID: 0, Id: 0, Threads: []int{0, 40}},
							{SocketID: 0, Id: 1, Threads: []int{1, 41}},
							{SocketID: 0, Id: 2, Threads: []int{2, 42}},
							{SocketID: 0, Id: 8, Threads: []int{3, 43}},
							{SocketID: 0, Id: 9, Threads: []int{4, 44}},
							{SocketID: 0, Id: 16, Threads: []int{5, 45}},
							{SocketID: 0, Id: 17, Threads: []int{6, 46}},
							{SocketID: 0, Id: 18, Threads: []int{7, 47}},
							{SocketID: 0, Id: 24, Threads: []int{8, 48}},
							{SocketID: 0, Id: 25, Threads: []int{9, 49}},
						},
					},
					{Id: 1,
						Cores: []cadvisorapi.Core{
							{SocketID: 0, Id: 3, Threads: []int{10, 50}},
							{SocketID: 0, Id: 4, Threads: []int{11, 51}},
							{SocketID: 0, Id: 10, Threads: []int{12, 52}},
							{SocketID: 0, Id: 11, Threads: []int{13, 53}},
							{SocketID: 0, Id: 12, Threads: []int{14, 54}},
							{SocketID: 0, Id: 19, Threads: []int{15, 55}},
							{SocketID: 0, Id: 20, Threads: []int{16, 56}},
							{SocketID: 0, Id: 26, Threads: []int{17, 57}},
							{SocketID: 0, Id: 27, Threads: []int{18, 58}},
							{SocketID: 0, Id: 28, Threads: []int{19, 59}},
						},
					},
					{Id: 2,
						Cores: []cadvisorapi.Core{
							{SocketID: 1, Id: 0, Threads: []int{20, 60}},
							{SocketID: 1, Id: 1, Threads: []int{21, 61}},
							{SocketID: 1, Id: 2, Threads: []int{22, 62}},
							{SocketID: 1, Id: 8, Threads: []int{23, 63}},
							{SocketID: 1, Id: 9, Threads: []int{24, 64}},
							{SocketID: 1, Id: 16, Threads: []int{25, 65}},
							{SocketID: 1, Id: 17, Threads: []int{26, 66}},
							{SocketID: 1, Id: 18, Threads: []int{27, 67}},
							{SocketID: 1, Id: 24, Threads: []int{28, 68}},
							{SocketID: 1, Id: 25, Threads: []int{29, 69}},
						},
					},
					{Id: 3,
						Cores: []cadvisorapi.Core{
							{SocketID: 1, Id: 3, Threads: []int{30, 70}},
							{SocketID: 1, Id: 4, Threads: []int{31, 71}},
							{SocketID: 1, Id: 10, Threads: []int{32, 72}},
							{SocketID: 1, Id: 11, Threads: []int{33, 73}},
							{SocketID: 1, Id: 12, Threads: []int{34, 74}},
							{SocketID: 1, Id: 19, Threads: []int{35, 75}},
							{SocketID: 1, Id: 20, Threads: []int{36, 76}},
							{SocketID: 1, Id: 26, Threads: []int{37, 77}},
							{SocketID: 1, Id: 27, Threads: []int{38, 78}},
							{SocketID: 1, Id: 28, Threads: []int{39, 79}},
						},
					},
				},
			},
			want: &CPUTopology{
				NumCPUs:      80,
				NumSockets:   2,
				NumCores:     40,
				NumNUMANodes: 4,
				CPUDetails: map[int]CPUInfo{
					0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					1:  {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					3:  {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0},
					5:  {CoreID: 5, SocketID: 0, NUMANodeID: 0},
					6:  {CoreID: 6, SocketID: 0, NUMANodeID: 0},
					7:  {CoreID: 7, SocketID: 0, NUMANodeID: 0},
					8:  {CoreID: 8, SocketID: 0, NUMANodeID: 0},
					9:  {CoreID: 9, SocketID: 0, NUMANodeID: 0},
					10: {CoreID: 10, SocketID: 0, NUMANodeID: 1},
					11: {CoreID: 11, SocketID: 0, NUMANodeID: 1},
					12: {CoreID: 12, SocketID: 0, NUMANodeID: 1},
					13: {CoreID: 13, SocketID: 0, NUMANodeID: 1},
					14: {CoreID: 14, SocketID: 0, NUMANodeID: 1},
					15: {CoreID: 15, SocketID: 0, NUMANodeID: 1},
					16: {CoreID: 16, SocketID: 0, NUMANodeID: 1},
					17: {CoreID: 17, SocketID: 0, NUMANodeID: 1},
					18: {CoreID: 18, SocketID: 0, NUMANodeID: 1},
					19: {CoreID: 19, SocketID: 0, NUMANodeID: 1},
					20: {CoreID: 20, SocketID: 1, NUMANodeID: 2},
					21: {CoreID: 21, SocketID: 1, NUMANodeID: 2},
					22: {CoreID: 22, SocketID: 1, NUMANodeID: 2},
					23: {CoreID: 23, SocketID: 1, NUMANodeID: 2},
					24: {CoreID: 24, SocketID: 1, NUMANodeID: 2},
					25: {CoreID: 25, SocketID: 1, NUMANodeID: 2},
					26: {CoreID: 26, SocketID: 1, NUMANodeID: 2},
					27: {CoreID: 27, SocketID: 1, NUMANodeID: 2},
					28: {CoreID: 28, SocketID: 1, NUMANodeID: 2},
					29: {CoreID: 29, SocketID: 1, NUMANodeID: 2},
					30: {CoreID: 30, SocketID: 1, NUMANodeID: 3},
					31: {CoreID: 31, SocketID: 1, NUMANodeID: 3},
					32: {CoreID: 32, SocketID: 1, NUMANodeID: 3},
					33: {CoreID: 33, SocketID: 1, NUMANodeID: 3},
					34: {CoreID: 34, SocketID: 1, NUMANodeID: 3},
					35: {CoreID: 35, SocketID: 1, NUMANodeID: 3},
					36: {CoreID: 36, SocketID: 1, NUMANodeID: 3},
					37: {CoreID: 37, SocketID: 1, NUMANodeID: 3},
					38: {CoreID: 38, SocketID: 1, NUMANodeID: 3},
					39: {CoreID: 39, SocketID: 1, NUMANodeID: 3},
					40: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					41: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					42: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					43: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					44: {CoreID: 4, SocketID: 0, NUMANodeID: 0},
					45: {CoreID: 5, SocketID: 0, NUMANodeID: 0},
					46: {CoreID: 6, SocketID: 0, NUMANodeID: 0},
					47: {CoreID: 7, SocketID: 0, NUMANodeID: 0},
					48: {CoreID: 8, SocketID: 0, NUMANodeID: 0},
					49: {CoreID: 9, SocketID: 0, NUMANodeID: 0},
					50: {CoreID: 10, SocketID: 0, NUMANodeID: 1},
					51: {CoreID: 11, SocketID: 0, NUMANodeID: 1},
					52: {CoreID: 12, SocketID: 0, NUMANodeID: 1},
					53: {CoreID: 13, SocketID: 0, NUMANodeID: 1},
					54: {CoreID: 14, SocketID: 0, NUMANodeID: 1},
					55: {CoreID: 15, SocketID: 0, NUMANodeID: 1},
					56: {CoreID: 16, SocketID: 0, NUMANodeID: 1},
					57: {CoreID: 17, SocketID: 0, NUMANodeID: 1},
					58: {CoreID: 18, SocketID: 0, NUMANodeID: 1},
					59: {CoreID: 19, SocketID: 0, NUMANodeID: 1},
					60: {CoreID: 20, SocketID: 1, NUMANodeID: 2},
					61: {CoreID: 21, SocketID: 1, NUMANodeID: 2},
					62: {CoreID: 22, SocketID: 1, NUMANodeID: 2},
					63: {CoreID: 23, SocketID: 1, NUMANodeID: 2},
					64: {CoreID: 24, SocketID: 1, NUMANodeID: 2},
					65: {CoreID: 25, SocketID: 1, NUMANodeID: 2},
					66: {CoreID: 26, SocketID: 1, NUMANodeID: 2},
					67: {CoreID: 27, SocketID: 1, NUMANodeID: 2},
					68: {CoreID: 28, SocketID: 1, NUMANodeID: 2},
					69: {CoreID: 29, SocketID: 1, NUMANodeID: 2},
					70: {CoreID: 30, SocketID: 1, NUMANodeID: 3},
					71: {CoreID: 31, SocketID: 1, NUMANodeID: 3},
					72: {CoreID: 32, SocketID: 1, NUMANodeID: 3},
					73: {CoreID: 33, SocketID: 1, NUMANodeID: 3},
					74: {CoreID: 34, SocketID: 1, NUMANodeID: 3},
					75: {CoreID: 35, SocketID: 1, NUMANodeID: 3},
					76: {CoreID: 36, SocketID: 1, NUMANodeID: 3},
					77: {CoreID: 37, SocketID: 1, NUMANodeID: 3},
					78: {CoreID: 38, SocketID: 1, NUMANodeID: 3},
					79: {CoreID: 39, SocketID: 1, NUMANodeID: 3},
				},
			},
			wantErr: false,
		},
		{

			// FAKE Topology from dual xeon gold 6230
			// (see: dual xeon gold 6230).
			// We flip NUMA cells and Sockets to exercise the code.
			// TODO(fromanirh): replace with a real-world topology
			// once we find a suitable one.
			// Note: this is a fake topology. Thus, there is not a "correct"
			// representation. This one was created following the these concepts:
			// 1. be internally consistent (most important rule)
			// 2. be as close as possible as existing HW topologies
			// 3. if possible, minimize chances wrt existing HW topologies.
			name: "DualNumaMultiSocketPerNumaHT",
			machineInfo: cadvisorapi.MachineInfo{
				NumCores:   80,
				NumSockets: 4,
				Topology: []cadvisorapi.Node{
					{Id: 0,
						Cores: []cadvisorapi.Core{
							{SocketID: 0, Id: 0, Threads: []int{0, 40}},
							{SocketID: 0, Id: 1, Threads: []int{1, 41}},
							{SocketID: 0, Id: 2, Threads: []int{2, 42}},
							{SocketID: 0, Id: 8, Threads: []int{3, 43}},
							{SocketID: 0, Id: 9, Threads: []int{4, 44}},
							{SocketID: 0, Id: 16, Threads: []int{5, 45}},
							{SocketID: 0, Id: 17, Threads: []int{6, 46}},
							{SocketID: 0, Id: 18, Threads: []int{7, 47}},
							{SocketID: 0, Id: 24, Threads: []int{8, 48}},
							{SocketID: 0, Id: 25, Threads: []int{9, 49}},
							{SocketID: 1, Id: 3, Threads: []int{10, 50}},
							{SocketID: 1, Id: 4, Threads: []int{11, 51}},
							{SocketID: 1, Id: 10, Threads: []int{12, 52}},
							{SocketID: 1, Id: 11, Threads: []int{13, 53}},
							{SocketID: 1, Id: 12, Threads: []int{14, 54}},
							{SocketID: 1, Id: 19, Threads: []int{15, 55}},
							{SocketID: 1, Id: 20, Threads: []int{16, 56}},
							{SocketID: 1, Id: 26, Threads: []int{17, 57}},
							{SocketID: 1, Id: 27, Threads: []int{18, 58}},
							{SocketID: 1, Id: 28, Threads: []int{19, 59}},
						},
					},
					{Id: 1,
						Cores: []cadvisorapi.Core{
							{SocketID: 2, Id: 0, Threads: []int{20, 60}},
							{SocketID: 2, Id: 1, Threads: []int{21, 61}},
							{SocketID: 2, Id: 2, Threads: []int{22, 62}},
							{SocketID: 2, Id: 8, Threads: []int{23, 63}},
							{SocketID: 2, Id: 9, Threads: []int{24, 64}},
							{SocketID: 2, Id: 16, Threads: []int{25, 65}},
							{SocketID: 2, Id: 17, Threads: []int{26, 66}},
							{SocketID: 2, Id: 18, Threads: []int{27, 67}},
							{SocketID: 2, Id: 24, Threads: []int{28, 68}},
							{SocketID: 2, Id: 25, Threads: []int{29, 69}},
							{SocketID: 3, Id: 3, Threads: []int{30, 70}},
							{SocketID: 3, Id: 4, Threads: []int{31, 71}},
							{SocketID: 3, Id: 10, Threads: []int{32, 72}},
							{SocketID: 3, Id: 11, Threads: []int{33, 73}},
							{SocketID: 3, Id: 12, Threads: []int{34, 74}},
							{SocketID: 3, Id: 19, Threads: []int{35, 75}},
							{SocketID: 3, Id: 20, Threads: []int{36, 76}},
							{SocketID: 3, Id: 26, Threads: []int{37, 77}},
							{SocketID: 3, Id: 27, Threads: []int{38, 78}},
							{SocketID: 3, Id: 28, Threads: []int{39, 79}},
						},
					},
				},
			},
			want: &CPUTopology{
				NumCPUs:      80,
				NumSockets:   4,
				NumCores:     40,
				NumNUMANodes: 2,
				CPUDetails: map[int]CPUInfo{
					0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					1:  {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					3:  {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0},
					5:  {CoreID: 5, SocketID: 0, NUMANodeID: 0},
					6:  {CoreID: 6, SocketID: 0, NUMANodeID: 0},
					7:  {CoreID: 7, SocketID: 0, NUMANodeID: 0},
					8:  {CoreID: 8, SocketID: 0, NUMANodeID: 0},
					9:  {CoreID: 9, SocketID: 0, NUMANodeID: 0},
					10: {CoreID: 10, SocketID: 1, NUMANodeID: 0},
					11: {CoreID: 11, SocketID: 1, NUMANodeID: 0},
					12: {CoreID: 12, SocketID: 1, NUMANodeID: 0},
					13: {CoreID: 13, SocketID: 1, NUMANodeID: 0},
					14: {CoreID: 14, SocketID: 1, NUMANodeID: 0},
					15: {CoreID: 15, SocketID: 1, NUMANodeID: 0},
					16: {CoreID: 16, SocketID: 1, NUMANodeID: 0},
					17: {CoreID: 17, SocketID: 1, NUMANodeID: 0},
					18: {CoreID: 18, SocketID: 1, NUMANodeID: 0},
					19: {CoreID: 19, SocketID: 1, NUMANodeID: 0},
					20: {CoreID: 20, SocketID: 2, NUMANodeID: 1},
					21: {CoreID: 21, SocketID: 2, NUMANodeID: 1},
					22: {CoreID: 22, SocketID: 2, NUMANodeID: 1},
					23: {CoreID: 23, SocketID: 2, NUMANodeID: 1},
					24: {CoreID: 24, SocketID: 2, NUMANodeID: 1},
					25: {CoreID: 25, SocketID: 2, NUMANodeID: 1},
					26: {CoreID: 26, SocketID: 2, NUMANodeID: 1},
					27: {CoreID: 27, SocketID: 2, NUMANodeID: 1},
					28: {CoreID: 28, SocketID: 2, NUMANodeID: 1},
					29: {CoreID: 29, SocketID: 2, NUMANodeID: 1},
					30: {CoreID: 30, SocketID: 3, NUMANodeID: 1},
					31: {CoreID: 31, SocketID: 3, NUMANodeID: 1},
					32: {CoreID: 32, SocketID: 3, NUMANodeID: 1},
					33: {CoreID: 33, SocketID: 3, NUMANodeID: 1},
					34: {CoreID: 34, SocketID: 3, NUMANodeID: 1},
					35: {CoreID: 35, SocketID: 3, NUMANodeID: 1},
					36: {CoreID: 36, SocketID: 3, NUMANodeID: 1},
					37: {CoreID: 37, SocketID: 3, NUMANodeID: 1},
					38: {CoreID: 38, SocketID: 3, NUMANodeID: 1},
					39: {CoreID: 39, SocketID: 3, NUMANodeID: 1},
					40: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					41: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					42: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					43: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					44: {CoreID: 4, SocketID: 0, NUMANodeID: 0},
					45: {CoreID: 5, SocketID: 0, NUMANodeID: 0},
					46: {CoreID: 6, SocketID: 0, NUMANodeID: 0},
					47: {CoreID: 7, SocketID: 0, NUMANodeID: 0},
					48: {CoreID: 8, SocketID: 0, NUMANodeID: 0},
					49: {CoreID: 9, SocketID: 0, NUMANodeID: 0},
					50: {CoreID: 10, SocketID: 1, NUMANodeID: 0},
					51: {CoreID: 11, SocketID: 1, NUMANodeID: 0},
					52: {CoreID: 12, SocketID: 1, NUMANodeID: 0},
					53: {CoreID: 13, SocketID: 1, NUMANodeID: 0},
					54: {CoreID: 14, SocketID: 1, NUMANodeID: 0},
					55: {CoreID: 15, SocketID: 1, NUMANodeID: 0},
					56: {CoreID: 16, SocketID: 1, NUMANodeID: 0},
					57: {CoreID: 17, SocketID: 1, NUMANodeID: 0},
					58: {CoreID: 18, SocketID: 1, NUMANodeID: 0},
					59: {CoreID: 19, SocketID: 1, NUMANodeID: 0},
					60: {CoreID: 20, SocketID: 2, NUMANodeID: 1},
					61: {CoreID: 21, SocketID: 2, NUMANodeID: 1},
					62: {CoreID: 22, SocketID: 2, NUMANodeID: 1},
					63: {CoreID: 23, SocketID: 2, NUMANodeID: 1},
					64: {CoreID: 24, SocketID: 2, NUMANodeID: 1},
					65: {CoreID: 25, SocketID: 2, NUMANodeID: 1},
					66: {CoreID: 26, SocketID: 2, NUMANodeID: 1},
					67: {CoreID: 27, SocketID: 2, NUMANodeID: 1},
					68: {CoreID: 28, SocketID: 2, NUMANodeID: 1},
					69: {CoreID: 29, SocketID: 2, NUMANodeID: 1},
					70: {CoreID: 30, SocketID: 3, NUMANodeID: 1},
					71: {CoreID: 31, SocketID: 3, NUMANodeID: 1},
					72: {CoreID: 32, SocketID: 3, NUMANodeID: 1},
					73: {CoreID: 33, SocketID: 3, NUMANodeID: 1},
					74: {CoreID: 34, SocketID: 3, NUMANodeID: 1},
					75: {CoreID: 35, SocketID: 3, NUMANodeID: 1},
					76: {CoreID: 36, SocketID: 3, NUMANodeID: 1},
					77: {CoreID: 37, SocketID: 3, NUMANodeID: 1},
					78: {CoreID: 38, SocketID: 3, NUMANodeID: 1},
					79: {CoreID: 39, SocketID: 3, NUMANodeID: 1},
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
				NumCPUs:      4,
				NumSockets:   2,
				NumCores:     4,
				NumNUMANodes: 2,
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
				NumCPUs:      12,
				NumSockets:   2,
				NumCores:     6,
				NumNUMANodes: 2,
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
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("Discover() = %v, want %v diff=%s", got, tt.want, diff)
			}
		})
	}
}

func TestCPUDetailsKeepOnly(t *testing.T) {
	details := CPUDetails{
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
		cpus: cpuset.New(0, 1),
		want: map[int]CPUInfo{
			0: {},
			1: {},
		},
	}, {
		name: "cpus is not in CPUDetails.",
		cpus: cpuset.New(3),
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
		want: cpuset.New(0, 1),
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
	details1 := CPUDetails{
		0: {SocketID: 0, NUMANodeID: 0},
		1: {SocketID: 1, NUMANodeID: 0},
		2: {SocketID: 2, NUMANodeID: 1},
		3: {SocketID: 3, NUMANodeID: 1},
	}

	// poorly designed mainboards
	details2 := CPUDetails{
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
		want:    cpuset.New(0, 1),
	}, {
		name:    "Socket IDs is not in CPUDetails.",
		details: details1,
		ids:     []int{4},
		want:    cpuset.New(),
	}, {
		name:    "Socket IDs is in CPUDetails. (poorly designed mainboards)",
		details: details2,
		ids:     []int{0},
		want:    cpuset.New(0, 1),
	}, {
		name:    "Socket IDs is not in CPUDetails. (poorly designed mainboards)",
		details: details2,
		ids:     []int{3},
		want:    cpuset.New(),
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
		want: cpuset.New(0, 1),
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
	details := CPUDetails{
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
		want: cpuset.New(0, 1, 2),
	}, {
		name: "Socket IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.New(),
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
	details := CPUDetails{
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
		want: cpuset.New(0, 1, 2),
	}, {
		name: "NUMANodes IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.New(),
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
		want: cpuset.New(0, 1),
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
	details := CPUDetails{
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
		want: cpuset.New(0, 1, 2),
	}, {
		name: "NUMANodes IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.New(),
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
	details := CPUDetails{
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
		want: cpuset.New(0, 1, 2),
	}, {
		name: "Socket IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.New(),
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
		want: cpuset.New(0, 1),
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
	details := CPUDetails{
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
		want: cpuset.New(0, 1, 2),
	}, {
		name: "NUMANode IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.New(),
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
	details := CPUDetails{
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
		want: cpuset.New(0, 1, 2),
	}, {
		name: "Core IDs is not in CPUDetails.",
		ids:  []int{3},
		want: cpuset.New(),
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

func TestCPUCoreID(t *testing.T) {
	topoDualSocketHT := &CPUTopology{
		NumCPUs:    12,
		NumSockets: 2,
		NumCores:   6,
		CPUDetails: map[int]CPUInfo{
			0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
			1:  {CoreID: 1, SocketID: 1, NUMANodeID: 1},
			2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
			3:  {CoreID: 3, SocketID: 1, NUMANodeID: 1},
			4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0},
			5:  {CoreID: 5, SocketID: 1, NUMANodeID: 1},
			6:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
			7:  {CoreID: 1, SocketID: 1, NUMANodeID: 1},
			8:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
			9:  {CoreID: 3, SocketID: 1, NUMANodeID: 1},
			10: {CoreID: 4, SocketID: 0, NUMANodeID: 0},
			11: {CoreID: 5, SocketID: 1, NUMANodeID: 1},
		},
	}

	tests := []struct {
		name    string
		topo    *CPUTopology
		id      int
		want    int
		wantErr bool
	}{{
		name: "Known Core ID",
		topo: topoDualSocketHT,
		id:   2,
		want: 2,
	}, {
		name: "Known Core ID (core sibling).",
		topo: topoDualSocketHT,
		id:   8,
		want: 2,
	}, {
		name:    "Unknown Core ID.",
		topo:    topoDualSocketHT,
		id:      -2,
		want:    -1,
		wantErr: true,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.topo.CPUCoreID(tt.id)
			gotErr := (err != nil)
			if gotErr != tt.wantErr {
				t.Errorf("CPUCoreID() returned err %v, want %v", gotErr, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("CPUCoreID() returned %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUSocketID(t *testing.T) {
	topoDualSocketHT := &CPUTopology{
		NumCPUs:    12,
		NumSockets: 2,
		NumCores:   6,
		CPUDetails: map[int]CPUInfo{
			0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
			1:  {CoreID: 1, SocketID: 1, NUMANodeID: 1},
			2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
			3:  {CoreID: 3, SocketID: 1, NUMANodeID: 1},
			4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0},
			5:  {CoreID: 5, SocketID: 1, NUMANodeID: 1},
			6:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
			7:  {CoreID: 1, SocketID: 1, NUMANodeID: 1},
			8:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
			9:  {CoreID: 3, SocketID: 1, NUMANodeID: 1},
			10: {CoreID: 4, SocketID: 0, NUMANodeID: 0},
			11: {CoreID: 5, SocketID: 1, NUMANodeID: 1},
		},
	}

	tests := []struct {
		name    string
		topo    *CPUTopology
		id      int
		want    int
		wantErr bool
	}{{
		name: "Known Core ID",
		topo: topoDualSocketHT,
		id:   3,
		want: 1,
	}, {
		name: "Known Core ID (core sibling).",
		topo: topoDualSocketHT,
		id:   9,
		want: 1,
	}, {
		name:    "Unknown Core ID.",
		topo:    topoDualSocketHT,
		id:      1000,
		want:    -1,
		wantErr: true,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.topo.CPUSocketID(tt.id)
			gotErr := (err != nil)
			if gotErr != tt.wantErr {
				t.Errorf("CPUSocketID() returned err %v, want %v", gotErr, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("CPUSocketID() returned %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCPUNUMANodeID(t *testing.T) {
	topoDualSocketHT := &CPUTopology{
		NumCPUs:    12,
		NumSockets: 2,
		NumCores:   6,
		CPUDetails: map[int]CPUInfo{
			0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
			1:  {CoreID: 1, SocketID: 1, NUMANodeID: 1},
			2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
			3:  {CoreID: 3, SocketID: 1, NUMANodeID: 1},
			4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0},
			5:  {CoreID: 5, SocketID: 1, NUMANodeID: 1},
			6:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
			7:  {CoreID: 1, SocketID: 1, NUMANodeID: 1},
			8:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
			9:  {CoreID: 3, SocketID: 1, NUMANodeID: 1},
			10: {CoreID: 4, SocketID: 0, NUMANodeID: 0},
			11: {CoreID: 5, SocketID: 1, NUMANodeID: 1},
		},
	}

	tests := []struct {
		name    string
		topo    *CPUTopology
		id      int
		want    int
		wantErr bool
	}{{
		name: "Known Core ID",
		topo: topoDualSocketHT,
		id:   0,
		want: 0,
	}, {
		name: "Known Core ID (core sibling).",
		topo: topoDualSocketHT,
		id:   6,
		want: 0,
	}, {
		name:    "Unknown Core ID.",
		topo:    topoDualSocketHT,
		id:      1000,
		want:    -1,
		wantErr: true,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.topo.CPUNUMANodeID(tt.id)
			gotErr := (err != nil)
			if gotErr != tt.wantErr {
				t.Errorf("CPUSocketID() returned err %v, want %v", gotErr, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("CPUSocketID() returned %v, want %v", got, tt.want)
			}
		})
	}
}
