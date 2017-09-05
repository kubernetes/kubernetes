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
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
)

var (
	topoSingleSocketHT = &topology.CPUTopology{
		NumCPUs:    8,
		NumSockets: 1,
		NumCores:   4,
		CPUDetails: map[int]topology.CPUInfo{
			0: {CoreID: 0, SocketID: 0},
			1: {CoreID: 1, SocketID: 0},
			2: {CoreID: 2, SocketID: 0},
			3: {CoreID: 3, SocketID: 0},
			4: {CoreID: 0, SocketID: 0},
			5: {CoreID: 1, SocketID: 0},
			6: {CoreID: 2, SocketID: 0},
			7: {CoreID: 3, SocketID: 0},
		},
	}

	topoDualSocketHT = &topology.CPUTopology{
		NumCPUs:    12,
		NumSockets: 2,
		NumCores:   6,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0},
			1:  {CoreID: 1, SocketID: 1},
			2:  {CoreID: 2, SocketID: 0},
			3:  {CoreID: 3, SocketID: 1},
			4:  {CoreID: 4, SocketID: 0},
			5:  {CoreID: 5, SocketID: 1},
			6:  {CoreID: 0, SocketID: 0},
			7:  {CoreID: 1, SocketID: 1},
			8:  {CoreID: 2, SocketID: 0},
			9:  {CoreID: 3, SocketID: 1},
			10: {CoreID: 4, SocketID: 0},
			11: {CoreID: 5, SocketID: 1},
		},
	}

	topoDualSocketNoHT = &topology.CPUTopology{
		NumCPUs:    8,
		NumSockets: 2,
		NumCores:   8,
		CPUDetails: map[int]topology.CPUInfo{
			0: {CoreID: 0, SocketID: 0},
			1: {CoreID: 1, SocketID: 0},
			2: {CoreID: 2, SocketID: 0},
			3: {CoreID: 3, SocketID: 0},
			4: {CoreID: 4, SocketID: 1},
			5: {CoreID: 5, SocketID: 1},
			6: {CoreID: 6, SocketID: 1},
			7: {CoreID: 7, SocketID: 1},
		},
	}
)
