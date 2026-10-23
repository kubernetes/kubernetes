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
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"testing"

	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/utils/cpuset"
)

func TestCPUAccumulatorFreeSockets(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description   string
		topo          *topology.CPUTopology
		availableCPUs cpuset.CPUSet
		expect        []int
	}{
		{
			"single socket HT, 1 socket free",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			[]int{0},
		},
		{
			"single socket HT, 0 sockets free",
			topoSingleSocketHT,
			cpuset.New(1, 2, 3, 4, 5, 6, 7),
			[]int{},
		},
		{
			"dual socket HT, 2 sockets free",
			topoDualSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			[]int{0, 1},
		},
		{
			"dual socket HT, 1 socket free",
			topoDualSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
			[]int{1},
		},
		{
			"dual socket HT, 0 sockets free",
			topoDualSocketHT,
			cpuset.New(0, 2, 3, 4, 5, 6, 7, 8, 9, 11),
			[]int{},
		},
		{
			"dual socket, multi numa per socket, HT, 2 sockets free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "0-79"),
			[]int{0, 1},
		},
		{
			"dual socket, multi numa per socket, HT, 1 sockets free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "1-79"),
			[]int{1},
		},
		{
			"dual socket, multi numa per socket, HT, 0 sockets free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "1-78"),
			[]int{},
		},
		{
			"dual socket, non-uniform CPUs per socket, HT, 1 socket free",
			topoDualSocketMultiNumaPerSocketMixedChips,
			mustParseCPUSet(t, "4-47,52-95"),
			[]int{1},
		},
		{
			"dual socket, non-uniform CPUs per socket, HT, 0 socket free",
			topoDualSocketMultiNumaPerSocketMixedChips,
			mustParseCPUSet(t, "4-15,17-47,52-63,65-95"),
			[]int{},
		},
		{
			"dual numa, multi socket per per socket, HT, 4 sockets free",
			fakeTopoMultiSocketDualSocketPerNumaHT,
			mustParseCPUSet(t, "0-79"),
			[]int{0, 1, 2, 3},
		},
		{
			"dual numa, multi socket per per socket, HT, 3 sockets free",
			fakeTopoMultiSocketDualSocketPerNumaHT,
			mustParseCPUSet(t, "0-19,21-79"),
			[]int{0, 1, 3},
		},
		{
			"dual numa, multi socket per per socket, HT, 2 sockets free",
			fakeTopoMultiSocketDualSocketPerNumaHT,
			mustParseCPUSet(t, "0-59,61-78"),
			[]int{0, 1},
		},
		{
			"dual numa, multi socket per per socket, HT, 1 sockets free",
			fakeTopoMultiSocketDualSocketPerNumaHT,
			mustParseCPUSet(t, "1-19,21-38,41-60,61-78"),
			[]int{1},
		},
		{
			"dual numa, multi socket per per socket, HT, 0 sockets free",
			fakeTopoMultiSocketDualSocketPerNumaHT,
			mustParseCPUSet(t, "0-40,42-49,51-68,71-79"),
			[]int{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			acc := newCPUAccumulator(logger, tc.topo, tc.availableCPUs, 0, CPUSortingStrategyPacked)
			result := acc.freeSockets()
			sort.Ints(result)
			if !reflect.DeepEqual(result, tc.expect) {
				t.Errorf("expected %v to equal %v", result, tc.expect)

			}
		})
	}
}

func TestCPUAccumulatorFreeNUMANodes(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description   string
		topo          *topology.CPUTopology
		availableCPUs cpuset.CPUSet
		expect        []int
	}{
		{
			"single socket HT, 1 NUMA node free",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			[]int{0},
		},
		{
			"single socket HT, 0 NUMA Node free",
			topoSingleSocketHT,
			cpuset.New(1, 2, 3, 4, 5, 6, 7),
			[]int{},
		},
		{
			"dual socket HT, 2 NUMA Node free",
			topoDualSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			[]int{0, 1},
		},
		{
			"dual socket HT, 1 NUMA Node free",
			topoDualSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
			[]int{1},
		},
		{
			"dual socket HT, 0 NUMA node free",
			topoDualSocketHT,
			cpuset.New(0, 2, 3, 4, 5, 6, 7, 8, 9, 11),
			[]int{},
		},
		{
			"dual socket, multi numa per socket, HT, 4 NUMA Node free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "0-79"),
			[]int{0, 1, 2, 3},
		},
		{
			"dual socket, multi numa per socket, HT, 3 NUMA node free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "1-79"),
			[]int{1, 2, 3},
		},
		{
			"dual socket, multi numa per socket, HT, 2 NUMA node free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "1-9,11-79"),
			[]int{2, 3},
		},
		{
			"dual socket, multi numa per socket, HT, 1 NUMA node free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "1-9,11-59,61-79"),
			[]int{3},
		},
		{
			"dual socket, multi numa per socket, HT, 0 NUMA node free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "1-9,11-59,61-78"),
			[]int{},
		},
		{
			"dual numa, multi socket per per socket, HT, 2 NUMA node free",
			fakeTopoMultiSocketDualSocketPerNumaHT,
			mustParseCPUSet(t, "0-79"),
			[]int{0, 1},
		},
		{
			"dual numa, multi socket per per socket, HT, 1 NUMA node free",
			fakeTopoMultiSocketDualSocketPerNumaHT,
			mustParseCPUSet(t, "0-9,11-79"),
			[]int{1},
		},
		{
			"dual numa, multi socket per per socket, HT, 0 sockets free",
			fakeTopoMultiSocketDualSocketPerNumaHT,
			mustParseCPUSet(t, "0-9,11-59,61-79"),
			[]int{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			acc := newCPUAccumulator(logger, tc.topo, tc.availableCPUs, 0, CPUSortingStrategyPacked)
			result := acc.freeNUMANodes()
			if !reflect.DeepEqual(result, tc.expect) {
				t.Errorf("expected %v to equal %v", result, tc.expect)
			}
		})
	}
}

func TestCPUAccumulatorFreeSocketsAndNUMANodes(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description     string
		topo            *topology.CPUTopology
		availableCPUs   cpuset.CPUSet
		expectSockets   []int
		expectNUMANodes []int
	}{
		{
			"dual socket, multi numa per socket, HT, 2 Socket/4 NUMA Node free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "0-79"),
			[]int{0, 1},
			[]int{0, 1, 2, 3},
		},
		{
			"dual socket, multi numa per socket, HT, 1 Socket/3 NUMA node free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "1-79"),
			[]int{1},
			[]int{1, 2, 3},
		},
		{
			"dual socket, multi numa per socket, HT, 1 Socket/ 2 NUMA node free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "1-9,11-79"),
			[]int{1},
			[]int{2, 3},
		},
		{
			"dual socket, multi numa per socket, HT, 0 Socket/ 2 NUMA node free",
			topoDualSocketMultiNumaPerSocketHT,
			mustParseCPUSet(t, "1-59,61-79"),
			[]int{},
			[]int{1, 3},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			acc := newCPUAccumulator(logger, tc.topo, tc.availableCPUs, 0, CPUSortingStrategyPacked)
			resultNUMANodes := acc.freeNUMANodes()
			if !reflect.DeepEqual(resultNUMANodes, tc.expectNUMANodes) {
				t.Errorf("expected NUMA Nodes %v to equal %v", resultNUMANodes, tc.expectNUMANodes)
			}
			resultSockets := acc.freeSockets()
			if !reflect.DeepEqual(resultSockets, tc.expectSockets) {
				t.Errorf("expected Sockets %v to equal %v", resultSockets, tc.expectSockets)
			}
		})
	}
}

func TestCPUAccumulatorFreeCores(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description   string
		topo          *topology.CPUTopology
		availableCPUs cpuset.CPUSet
		expect        []int
	}{
		{
			"single socket HT, 4 cores free",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			[]int{0, 1, 2, 3},
		},
		{
			"single socket HT, 3 cores free",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 4, 5, 6),
			[]int{0, 1, 2},
		},
		{
			"single socket HT, 3 cores free (1 partially consumed)",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6),
			[]int{0, 1, 2},
		},
		{
			"single socket HT, 0 cores free",
			topoSingleSocketHT,
			cpuset.New(),
			[]int{},
		},
		{
			"single socket HT, 0 cores free (4 partially consumed)",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 3),
			[]int{},
		},
		{
			"single socket, 7 HT cores free (1 partially consumed) + 4 ST cores free",
			topoSingleSocketSingleNumaPerSocketPCoreHTECoreST,
			mustParseCPUSet(t, "1-15,24-27"),
			[]int{40, 41, 42, 43, 4, 8, 12, 16, 20, 24, 28},
		},
		{
			"single socket, 0 HT cores free (8 partially consumed) + 12 ST cores free",
			topoSingleSocketSingleNumaPerSocketPCoreHTECoreST,
			mustParseCPUSet(t, "0,2,4,6,8,10,12,14,16-27"),
			[]int{32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43},
		},
		{
			"dual socket HT, 6 cores free",
			topoDualSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			[]int{0, 2, 4, 1, 3, 5},
		},
		{
			"dual socket HT, 5 cores free (1 consumed from socket 0)",
			topoDualSocketHT,
			cpuset.New(2, 1, 3, 4, 5, 7, 8, 9, 10, 11),
			[]int{2, 4, 1, 3, 5},
		},
		{
			"dual socket HT, 4 cores free (1 consumed from each socket)",
			topoDualSocketHT,
			cpuset.New(2, 3, 4, 5, 8, 9, 10, 11),
			[]int{2, 4, 3, 5},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			acc := newCPUAccumulator(logger, tc.topo, tc.availableCPUs, 0, CPUSortingStrategyPacked)
			result := acc.freeCores()
			if !reflect.DeepEqual(result, tc.expect) {
				t.Errorf("expected %v to equal %v", result, tc.expect)
			}
		})
	}
}

func TestCPUAccumulatorFreeCPUs(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description   string
		topo          *topology.CPUTopology
		availableCPUs cpuset.CPUSet
		expect        []int
	}{
		{
			"single socket HT, 8 cpus free",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			[]int{0, 4, 1, 5, 2, 6, 3, 7},
		},
		{
			"single socket HT, 5 cpus free",
			topoSingleSocketHT,
			cpuset.New(3, 4, 5, 6, 7),
			[]int{4, 5, 6, 3, 7},
		},
		{
			"dual socket HT, 12 cpus free",
			topoDualSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			[]int{0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11},
		},
		{
			"dual socket HT, 11 cpus free",
			topoDualSocketHT,
			cpuset.New(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			[]int{6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11},
		},
		{
			"dual socket HT, 10 cpus free",
			topoDualSocketHT,
			cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			[]int{2, 8, 4, 10, 1, 7, 3, 9, 5, 11},
		},
		{
			"triple socket HT, 12 cpus free",
			topoTripleSocketHT,
			cpuset.New(0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13),
			[]int{12, 13, 0, 1, 2, 3, 6, 7, 8, 9, 10, 11},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			acc := newCPUAccumulator(logger, tc.topo, tc.availableCPUs, 0, CPUSortingStrategyPacked)
			result := acc.freeCPUs()
			if !reflect.DeepEqual(result, tc.expect) {
				t.Errorf("expected %v to equal %v", result, tc.expect)
			}
		})
	}
}

func TestCPUAccumulatorFreeCPUsSpreadRoundRobin(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description   string
		topo          *topology.CPUTopology
		availableCPUs cpuset.CPUSet
		expect        []int
	}{
		// Synthetic topology: contiguous sibling IDs, not representative of known hardware.
		{
			description: "contiguous sibling CPU IDs",
			topo: &topology.CPUTopology{
				NumCPUs:    8,
				NumSockets: 1,
				NumCores:   4,
				CPUDetails: map[int]topology.CPUInfo{
					0: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					1: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					2: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					3: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					4: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					5: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					6: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					7: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
				},
			},
			availableCPUs: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expect:        []int{0, 2, 4, 6, 1, 3, 5, 7},
		},
		{
			description:   "interleaved sibling CPU IDs",
			topo:          topoSingleSocketHT,
			availableCPUs: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expect:        []int{0, 1, 2, 3, 4, 5, 6, 7},
		},
		// Synthetic topology: arbitrary CPU numbering, does not correspond to known hardware.
		{
			description: "arbitrary logical CPU numbering",
			topo: &topology.CPUTopology{
				NumCPUs:    8,
				NumSockets: 1,
				NumCores:   4,
				CPUDetails: map[int]topology.CPUInfo{
					0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					7:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					2:  {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					11: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					5:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					9:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					1:  {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					14: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
				},
			},
			availableCPUs: cpuset.New(0, 1, 2, 5, 7, 9, 11, 14),
			expect:        []int{0, 2, 5, 1, 7, 11, 9, 14},
		},
		// Synthetic topology: dual-socket layout synthesized for spread verification, not a specific product.
		{
			description: "multiple sockets",
			topo: &topology.CPUTopology{
				NumCPUs:    16,
				NumSockets: 2,
				NumCores:   8,
				CPUDetails: map[int]topology.CPUInfo{
					0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					1:  {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					2:  {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					3:  {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					4:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					5:  {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					6:  {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					7:  {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					8:  {CoreID: 4, SocketID: 1, NUMANodeID: 1},
					9:  {CoreID: 4, SocketID: 1, NUMANodeID: 1},
					10: {CoreID: 5, SocketID: 1, NUMANodeID: 1},
					11: {CoreID: 5, SocketID: 1, NUMANodeID: 1},
					12: {CoreID: 6, SocketID: 1, NUMANodeID: 1},
					13: {CoreID: 6, SocketID: 1, NUMANodeID: 1},
					14: {CoreID: 7, SocketID: 1, NUMANodeID: 1},
					15: {CoreID: 7, SocketID: 1, NUMANodeID: 1},
				},
			},
			availableCPUs: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
			expect:        []int{0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15},
		},
		// Synthetic topology: partially populated cores on contiguous sibling layout, not modeled on real hardware.
		{
			description: "partially allocated cores",
			topo: &topology.CPUTopology{
				NumCPUs:    8,
				NumSockets: 1,
				NumCores:   4,
				CPUDetails: map[int]topology.CPUInfo{
					0: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					1: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					2: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					3: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					4: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					5: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					6: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
					7: {CoreID: 3, SocketID: 0, NUMANodeID: 0},
				},
			},
			availableCPUs: cpuset.New(0, 2, 3, 4, 6, 7),
			expect:        []int{0, 4, 2, 6, 3, 7},
		},
		// Synthetic topology: 3 threads per core is not common hardware; validates multi-thread spread.
		{
			description: "more than two CPUs per core",
			topo: &topology.CPUTopology{
				NumCPUs:    9,
				NumSockets: 1,
				NumCores:   3,
				CPUDetails: map[int]topology.CPUInfo{
					0: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					1: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					2: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					3: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					4: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					5: {CoreID: 1, SocketID: 0, NUMANodeID: 0},
					6: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					7: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
					8: {CoreID: 2, SocketID: 0, NUMANodeID: 0},
				},
			},
			availableCPUs: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
			expect:        []int{0, 3, 6, 1, 4, 7, 2, 5, 8},
		},
		// Synthetic topology: multi-NUMA per socket, shows spread ignores NUMA hierarchy and still round-robins per core within each socket.
		{
			description: "spread across multi-NUMA sockets round-robins per core",
			topo: &topology.CPUTopology{
				NumCPUs:      8,
				NumSockets:   2,
				NumCores:     4,
				NumNUMANodes: 4,
				CPUDetails: map[int]topology.CPUInfo{
					0: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					1: {CoreID: 0, SocketID: 0, NUMANodeID: 0},
					2: {CoreID: 1, SocketID: 0, NUMANodeID: 1},
					3: {CoreID: 1, SocketID: 0, NUMANodeID: 1},
					4: {CoreID: 2, SocketID: 1, NUMANodeID: 2},
					5: {CoreID: 2, SocketID: 1, NUMANodeID: 2},
					6: {CoreID: 3, SocketID: 1, NUMANodeID: 3},
					7: {CoreID: 3, SocketID: 1, NUMANodeID: 3},
				},
			},
			availableCPUs: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expect:        []int{0, 2, 1, 3, 4, 6, 5, 7},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			acc := newCPUAccumulator(logger, tc.topo, tc.availableCPUs, 0, CPUSortingStrategySpread)
			result := acc.freeCPUs()
			if !reflect.DeepEqual(result, tc.expect) {
				t.Errorf("expected %v to equal %v", result, tc.expect)
			}
		})
	}
}

func TestSortAvailableUncoreCaches(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description   string
		topo          *topology.CPUTopology
		availableCPUs cpuset.CPUSet
		expect        []int
	}{
		{
			description:   "topology with 1 (default) uncore cache, 0 cpus reserved",
			topo:          topoDualSocketMultiNumaPerSocketHT,
			availableCPUs: mustParseCPUSet(t, "0-79"),
			expect:        []int{0},
		},
		{
			description:   "topology with 2 uncore caches, multi numa per uncore, 2 cpus reserved",
			topo:          topoDualSocketSubNumaPerSocketHTMonolithicUncore,
			availableCPUs: mustParseCPUSet(t, "1-119,121-239"), // two cpu(s) from uncore0 reserved
			expect:        []int{0, 1},
		},
		{
			description:   "topology with 24 uncore caches, single numa per uncore, 3 cpus reserved",
			topo:          topoDualSocketSingleNumaPerSocketSMTUncore,
			availableCPUs: mustParseCPUSet(t, "0-90,92-151,153-282,284-383"), // two cpu(s) from uncore11 and one from uncore19 reserved
			expect:        []int{11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			acc := newCPUAccumulator(logger, tc.topo, tc.availableCPUs, 0, CPUSortingStrategyPacked)
			result := acc.sortAvailableUncoreCaches()
			if !reflect.DeepEqual(result, tc.expect) {
				t.Errorf("expected %v to equal %v", result, tc.expect)
			}
		})
	}
}

func TestCPUAccumulatorTake(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description     string
		topo            *topology.CPUTopology
		availableCPUs   cpuset.CPUSet
		takeCPUs        []cpuset.CPUSet
		numCPUs         int
		expectSatisfied bool
		expectFailed    bool
	}{
		{
			"take 0 cpus from a single socket HT, require 1",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			[]cpuset.CPUSet{cpuset.New()},
			1,
			false,
			false,
		},
		{
			"take 0 cpus from a single socket HT, require 1, none available",
			topoSingleSocketHT,
			cpuset.New(),
			[]cpuset.CPUSet{cpuset.New()},
			1,
			false,
			true,
		},
		{
			"take 1 cpu from a single socket HT, require 1",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			[]cpuset.CPUSet{cpuset.New(0)},
			1,
			true,
			false,
		},
		{
			"take 1 cpu from a single socket HT, require 2",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			[]cpuset.CPUSet{cpuset.New(0)},
			2,
			false,
			false,
		},
		{
			"take 2 cpu from a single socket HT, require 4, expect failed",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2),
			[]cpuset.CPUSet{cpuset.New(0), cpuset.New(1)},
			4,
			false,
			true,
		},
		{
			"take all cpus one at a time from a single socket HT, require 8",
			topoSingleSocketHT,
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			[]cpuset.CPUSet{
				cpuset.New(0),
				cpuset.New(1),
				cpuset.New(2),
				cpuset.New(3),
				cpuset.New(4),
				cpuset.New(5),
				cpuset.New(6),
				cpuset.New(7),
			},
			8,
			true,
			false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			acc := newCPUAccumulator(logger, tc.topo, tc.availableCPUs, tc.numCPUs, CPUSortingStrategyPacked)
			totalTaken := 0
			for _, cpus := range tc.takeCPUs {
				acc.take(cpus)
				totalTaken += cpus.Size()
			}
			if tc.expectSatisfied != acc.isSatisfied() {
				t.Errorf("expected acc.isSatisfied() to be %t", tc.expectSatisfied)
			}
			if tc.expectFailed != acc.isFailed() {
				t.Errorf("expected acc.isFailed() to be %t", tc.expectFailed)
			}
			for _, cpus := range tc.takeCPUs {
				availableCPUs := acc.details.CPUs()
				if cpus.Intersection(availableCPUs).Size() > 0 {
					t.Errorf("expected intersection of taken cpus [%s] and acc.details.CPUs() [%s] to be empty", cpus, availableCPUs)
				}
				if !cpus.IsSubsetOf(acc.result) {
					t.Errorf("expected [%s] to be a subset of acc.result [%s]", cpus, acc.result)
				}
			}
			expNumCPUsNeeded := tc.numCPUs - totalTaken
			if acc.numCPUsNeeded != expNumCPUsNeeded {
				t.Errorf("expected acc.numCPUsNeeded to be %d (got %d)", expNumCPUsNeeded, acc.numCPUsNeeded)
			}
		})
	}
}

type takeByTopologyTestCase struct {
	description   string
	topo          *topology.CPUTopology
	opts          StaticPolicyOptions
	availableCPUs cpuset.CPUSet
	numCPUs       int
	expErr        string
	expResult     cpuset.CPUSet
}

func commonTakeByTopologyTestCases(t *testing.T) []takeByTopologyTestCase {
	return []takeByTopologyTestCase{
		{
			"take more cpus than are available from single socket with HT",
			topoSingleSocketHT,
			StaticPolicyOptions{},
			cpuset.New(0, 2, 4, 6),
			5,
			"not enough cpus available to satisfy request: requested=5, available=4",
			cpuset.New(),
		},
		{
			"take zero cpus from single socket with HT",
			topoSingleSocketHT,
			StaticPolicyOptions{},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			0,
			"",
			cpuset.New(),
		},
		{
			"take one cpu from single socket with HT",
			topoSingleSocketHT,
			StaticPolicyOptions{},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			1,
			"",
			cpuset.New(0),
		},
		{
			"take one cpu from single socket with HT, some cpus are taken",
			topoSingleSocketHT,
			StaticPolicyOptions{},
			cpuset.New(1, 3, 5, 6, 7),
			1,
			"",
			cpuset.New(6),
		},
		{
			"take two cpus from single socket with HT",
			topoSingleSocketHT,
			StaticPolicyOptions{},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			2,
			"",
			cpuset.New(0, 4),
		},
		{
			"take all cpus from single socket with HT",
			topoSingleSocketHT,
			StaticPolicyOptions{},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			8,
			"",
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			"take two cpus from single socket with HT, only one core totally free",
			topoSingleSocketHT,
			StaticPolicyOptions{},
			cpuset.New(0, 1, 2, 3, 6),
			2,
			"",
			cpuset.New(2, 6),
		},
		{
			"take a socket of cpus from dual socket with HT",
			topoDualSocketHT,
			StaticPolicyOptions{},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			6,
			"",
			cpuset.New(0, 2, 4, 6, 8, 10),
		},
		{
			"take a socket of cpus from dual socket with multi-numa-per-socket with HT",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-79"),
			40,
			"",
			mustParseCPUSet(t, "0-19,40-59"),
		},
		{
			"take a NUMA node of cpus from dual socket with multi-numa-per-socket with HT",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-79"),
			20,
			"",
			mustParseCPUSet(t, "0-9,40-49"),
		},
		{
			"take a NUMA node of cpus from dual socket with multi-numa-per-socket with HT, with 1 NUMA node already taken",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "10-39,50-79"),
			20,
			"",
			mustParseCPUSet(t, "10-19,50-59"),
		},
		{
			"take a socket and a NUMA node of cpus from dual socket with multi-numa-per-socket with HT",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-79"),
			60,
			"",
			mustParseCPUSet(t, "0-29,40-69"),
		},
		{
			"take a socket and a NUMA node of cpus from dual socket with multi-numa-per-socket with HT, a core taken",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "1-39,41-79"), // reserve the first (phys) core (0,40)
			60,
			"",
			mustParseCPUSet(t, "10-39,50-79"),
		},
	}
}

func TestTakeByTopologyNUMAPacked(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := commonTakeByTopologyTestCases(t)
	testCases = append(testCases, []takeByTopologyTestCase{
		{
			"take one cpu from dual socket with HT - core from Socket 0",
			topoDualSocketHT,
			StaticPolicyOptions{},
			cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			1,
			"",
			cpuset.New(2),
		},
		{
			"allocate 4 full cores with 3 coming from the first NUMA node (filling it up) and 1 coming from the second NUMA node",
			topoDualSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-11"),
			8,
			"",
			mustParseCPUSet(t, "0,6,2,8,4,10,1,7"),
		},
		{
			"allocate 32 full cores with 30 coming from the first 3 NUMA nodes (filling them up) and 2 coming from the fourth NUMA node",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-79"),
			64,
			"",
			mustParseCPUSet(t, "0-29,40-69,30,31,70,71"),
		},
		{
			"allocate 2 cpus from non-uniform topology with 1 cpu reserved",
			topoSingleSocketSingleNumaPerSocketPCoreHTECoreST,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "1-27"), // 0 is reserved
			2,
			"",
			mustParseCPUSet(t, "16-17"),
		},
		// Test cases for PreferAlignByUncoreCache
		{
			"take cpus from two full UncoreCaches and partial from a single UncoreCache",
			topoUncoreSingleSocketNoSMT,
			StaticPolicyOptions{PreferAlignByUncoreCacheOption: true},
			mustParseCPUSet(t, "1-15"),
			10,
			"",
			cpuset.New(1, 2, 4, 5, 6, 7, 8, 9, 10, 11),
		},
		{
			"take one cpu from dual socket with HT - core from Socket 0",
			topoDualSocketHT,
			StaticPolicyOptions{PreferAlignByUncoreCacheOption: true},
			cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			1,
			"",
			cpuset.New(1),
		},
		{
			"take first available UncoreCache from first socket",
			topoUncoreDualSocketNoSMT,
			StaticPolicyOptions{PreferAlignByUncoreCacheOption: true},
			mustParseCPUSet(t, "0-15"),
			4,
			"",
			cpuset.New(0, 1, 2, 3),
		},
		{
			"take all available UncoreCache from first socket",
			topoUncoreDualSocketNoSMT,
			StaticPolicyOptions{PreferAlignByUncoreCacheOption: true},
			mustParseCPUSet(t, "2-15"),
			6,
			"",
			cpuset.New(2, 3, 4, 5, 6, 7),
		},
		{
			"take first available UncoreCache from second socket",
			topoUncoreDualSocketNoSMT,
			StaticPolicyOptions{PreferAlignByUncoreCacheOption: true},
			mustParseCPUSet(t, "8-15"),
			4,
			"",
			cpuset.New(8, 9, 10, 11),
		},
		{
			"take first available UncoreCache from available NUMA",
			topoUncoreSingleSocketMultiNuma,
			StaticPolicyOptions{PreferAlignByUncoreCacheOption: true},
			mustParseCPUSet(t, "3,4-8,12"),
			2,
			"",
			cpuset.New(4, 5),
		},
		{
			"take cpus from best available UncoreCache group of multi uncore cache single socket - SMT enabled",
			topoUncoreSingleSocketSMT,
			StaticPolicyOptions{PreferAlignByUncoreCacheOption: true},
			mustParseCPUSet(t, "2-3,10-11,4-7,12-15"),
			6,
			"",
			cpuset.New(4, 5, 6, 12, 13, 14),
		},
		{
			"take cpus from multiple UncoreCache of single socket - SMT enabled",
			topoUncoreSingleSocketSMT,
			StaticPolicyOptions{PreferAlignByUncoreCacheOption: true},
			mustParseCPUSet(t, "1-7,9-15"),
			10,
			"",
			mustParseCPUSet(t, "4-7,12-15,1,9"),
		},
	}...)

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			strategy := CPUSortingStrategyPacked
			if tc.opts.DistributeCPUsAcrossCores {
				strategy = CPUSortingStrategySpread
			}

			result, err := takeByTopologyNUMAPacked(logger, tc.topo, tc.availableCPUs, tc.numCPUs, strategy, tc.opts.PreferAlignByUncoreCacheOption)
			checkError(t, tc.expErr, err)
			if !result.Equals(tc.expResult) {
				t.Errorf("expected result [%s] to equal [%s]", result, tc.expResult)
			}
		})
	}
}

func TestTakeByTopologyWithSpreadPhysicalCPUsPreferredOption(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description   string
		topo          *topology.CPUTopology
		opts          StaticPolicyOptions
		availableCPUs cpuset.CPUSet
		numCPUs       int
		expErr        string
		expResult     cpuset.CPUSet
	}{
		{
			"take a socket of cpus from single socket with HT, 3 cpus",
			topoSingleSocketHT,
			StaticPolicyOptions{DistributeCPUsAcrossCores: true},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			3,
			"",
			cpuset.New(0, 1, 2),
		},
		{
			"take a socket of cpus from dual socket with HT, 2 cpus",
			topoDualSocketHT,
			StaticPolicyOptions{DistributeCPUsAcrossCores: true},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			2,
			"",
			cpuset.New(0, 2),
		},
		{
			"take a socket of cpus from dual socket with HT, 3 cpus",
			topoDualSocketHT,
			StaticPolicyOptions{DistributeCPUsAcrossCores: true},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			3,
			"",
			cpuset.New(0, 2, 4),
		},
		{
			"take a socket of cpus from dual socket with HT, 6 cpus",
			topoDualSocketHT,
			StaticPolicyOptions{DistributeCPUsAcrossCores: true},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			6,
			"",
			cpuset.New(0, 2, 4, 6, 8, 10),
		},
		{
			"take cpus from dual socket with HT, 8 cpus",
			topoDualSocketHT,
			StaticPolicyOptions{DistributeCPUsAcrossCores: true},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			8,
			"",
			cpuset.New(0, 2, 4, 6, 8, 10, 1, 3),
		},
		{
			"take a socket of cpus from dual socket without HT, 2 cpus",
			topoDualSocketNoHT,
			StaticPolicyOptions{DistributeCPUsAcrossCores: true},
			cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			2,
			"",
			cpuset.New(0, 1),
		},
		{
			// DistributeCPUsAcrossCores doesn't care socket and numa ranking. such setting in test is transparent.
			"take a socket of cpus from dual socket with multi numa per socket and HT, 12 cpus",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{DistributeCPUsAcrossCores: true},
			mustParseCPUSet(t, "0-79"),
			8,
			"",
			mustParseCPUSet(t, "0-7"),
		},
		{
			"take a socket of cpus from quad socket four way with HT, 12 cpus",
			topoQuadSocketFourWayHT,
			StaticPolicyOptions{DistributeCPUsAcrossCores: true},
			mustParseCPUSet(t, "0-287"),
			12,
			"",
			cpuset.New(0, 50, 57, 58, 71, 72, 79, 80, 87, 88, 95, 96),
		},
	}

	for _, tc := range testCases {
		strategy := CPUSortingStrategyPacked
		if tc.opts.DistributeCPUsAcrossCores {
			strategy = CPUSortingStrategySpread
		}
		result, err := takeByTopologyNUMAPacked(logger, tc.topo, tc.availableCPUs, tc.numCPUs, strategy, tc.opts.PreferAlignByUncoreCacheOption)
		checkError(t, tc.expErr, err)
		if !result.Equals(tc.expResult) {
			t.Errorf("testCase %q failed, expected result [%s] to equal [%s]", tc.description, result, tc.expResult)
		}
	}
}

type takeByTopologyExtendedTestCase struct {
	description   string
	topo          *topology.CPUTopology
	opts          StaticPolicyOptions
	availableCPUs cpuset.CPUSet
	numCPUs       int
	cpuGroupSize  int
	expErr        string
	expResult     cpuset.CPUSet
}

func commonTakeByTopologyExtendedTestCases(t *testing.T) []takeByTopologyExtendedTestCase {
	var extendedTestCases []takeByTopologyExtendedTestCase

	testCases := commonTakeByTopologyTestCases(t)
	for _, tc := range testCases {
		extendedTestCases = append(extendedTestCases, takeByTopologyExtendedTestCase{
			tc.description,
			tc.topo,
			tc.opts,
			tc.availableCPUs,
			tc.numCPUs,
			1,
			tc.expErr,
			tc.expResult,
		})
	}

	extendedTestCases = append(extendedTestCases, []takeByTopologyExtendedTestCase{
		{
			"allocate 4 full cores with 2 distributed across each NUMA node",
			topoDualSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-11"),
			8,
			1,
			"",
			mustParseCPUSet(t, "0,6,2,8,1,7,3,9"),
		},
		{
			"allocate 32 full cores with 8 distributed across each NUMA node",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-79"),
			64,
			1,
			"",
			mustParseCPUSet(t, "0-7,10-17,20-27,30-37,40-47,50-57,60-67,70-77"),
		},
		{
			"allocate 24 full cores with 8 distributed across the first 3 NUMA nodes",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-79"),
			48,
			1,
			"",
			mustParseCPUSet(t, "0-7,10-17,20-27,40-47,50-57,60-67"),
		},
		{
			"allocate 24 full cores with 8 distributed across the first 3 NUMA nodes (taking all but 2 from the first NUMA node)",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "1-29,32-39,41-69,72-79"),
			48,
			1,
			"",
			mustParseCPUSet(t, "1-8,10-17,20-27,41-48,50-57,60-67"),
		},
		{
			"allocate 24 full cores with 8 distributed across the last 3 NUMA nodes (even though all 8 could be allocated from the first NUMA node)",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "2-29,31-39,42-69,71-79"),
			48,
			1,
			"",
			mustParseCPUSet(t, "10-17,20-27,31-38,50-57,60-67,71-78"),
		},
		{
			"allocate 8 full cores with 2 distributed across each NUMA node",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-2,10-12,20-22,30-32,40-41,50-51,60-61,70-71"),
			16,
			1,
			"",
			mustParseCPUSet(t, "0-1,10-11,20-21,30-31,40-41,50-51,60-61,70-71"),
		},
		{
			"allocate 8 full cores with 2 distributed across each NUMA node",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-2,10-12,20-22,30-32,40-41,50-51,60-61,70-71"),
			16,
			1,
			"",
			mustParseCPUSet(t, "0-1,10-11,20-21,30-31,40-41,50-51,60-61,70-71"),
		},
	}...)

	return extendedTestCases
}

func TestTakeByTopologyNUMADistributed(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := commonTakeByTopologyExtendedTestCases(t)
	testCases = append(testCases, []takeByTopologyExtendedTestCase{
		{
			"take one cpu from dual socket with HT - core from Socket 0",
			topoDualSocketHT,
			StaticPolicyOptions{},
			cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			1,
			1,
			"",
			cpuset.New(1),
		},
		{
			"take one cpu from dual socket with HT - core from Socket 0 - cpuGroupSize 2",
			topoDualSocketHT,
			StaticPolicyOptions{},
			cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			1,
			2,
			"",
			cpuset.New(2),
		},
		{
			"allocate 13 full cores distributed across the first 2 NUMA nodes",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-79"),
			26,
			1,
			"",
			mustParseCPUSet(t, "0-6,10-16,40-45,50-55"),
		},
		{
			"allocate 13 full cores distributed across the first 2 NUMA nodes (cpuGroupSize 2)",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-79"),
			26,
			2,
			"",
			mustParseCPUSet(t, "0-6,10-15,40-46,50-55"),
		},
		{
			"allocate 31 full cores with 15 CPUs distributed across each NUMA node and 1 CPU spilling over to each of NUMA 0, 1",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-79"),
			62,
			1,
			"",
			mustParseCPUSet(t, "0-7,10-17,20-27,30-37,40-47,50-57,60-66,70-76"),
		},
		{
			"allocate 31 full cores with 14 CPUs distributed across each NUMA node and 2 CPUs spilling over to each of NUMA 0, 1, 2 (cpuGroupSize 2)",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-79"),
			62,
			2,
			"",
			mustParseCPUSet(t, "0-7,10-17,20-27,30-36,40-47,50-57,60-67,70-76"),
		},
		{
			"allocate 31 full cores with 15 CPUs distributed across each NUMA node and 1 CPU spilling over to each of NUMA 2, 3 (to keep balance)",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-8,10-18,20-39,40-48,50-58,60-79"),
			62,
			1,
			"",
			mustParseCPUSet(t, "0-7,10-17,20-27,30-37,40-46,50-56,60-67,70-77"),
		},
		{
			"allocate 31 full cores with 14 CPUs distributed across each NUMA node and 2 CPUs spilling over to each of NUMA 0, 2, 3 (to keep balance with cpuGroupSize 2)",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-8,10-18,20-39,40-48,50-58,60-79"),
			62,
			2,
			"",
			mustParseCPUSet(t, "0-7,10-16,20-27,30-37,40-47,50-56,60-67,70-77"),
		},
		{
			"ensure bestRemainder chosen with NUMA nodes that have enough CPUs to satisfy the request",
			topoDualSocketMultiNumaPerSocketHT,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0-3,10-13,20-23,30-36,40-43,50-53,60-63,70-76"),
			34,
			1,
			"",
			mustParseCPUSet(t, "0-3,10-13,20-23,30-34,40-43,50-53,60-63,70-74"),
		},
		{
			"ensure previous failure encountered on live machine has been fixed (1/1)",
			topoDualSocketMultiNumaPerSocketHTLarge,
			StaticPolicyOptions{},
			mustParseCPUSet(t, "0,128,30,31,158,159,43-47,171-175,62,63,190,191,75-79,203-207,94,96,222,223,101-111,229-239,126,127,254,255"),
			28,
			1,
			"",
			mustParseCPUSet(t, "43-47,75-79,96,101-105,171-174,203-206,229-232"),
		},
		// availableCPUs: Socket 0 has 126 CPUs, Socket 1 has 126 CPUs
		// Socket 0: NUMA 0: CPU 0-15, 128-143   (32 CPUs);  NUMA 1: CPU 16-31, 144-159   (32 CPUs)
		//           NUMA 2: CPU 32-47, 160-175  (32 CPUs);  NUMA 3: CPU 48-62, 176-190   (30 CPUs)
		// Socket 1: NUMA 4: CPU 64-79, 192-207  (32 CPUs);  NUMA 5: CPU 80-95, 208-223   (32 CPUs)
		//           NUMA 6: CPU 96-111, 224-239 (32 CPUs);  NUMA 7: CPU 112-126, 240-254 (30 CPUs)
		//
		// expResult: 120 CPUs from Socket 0 only (30 CPUs per NUMA × 4 NUMA nodes)
		// Socket 0: NUMA 0: CPU 0-14, 128-142   (30 CPUs);  NUMA 1: CPU 16-30, 144-158   (30 CPUs)
		//           NUMA 2: CPU 32-46, 160-174  (30 CPUs);  NUMA 3: CPU 48-62, 176-190   (30 CPUs)
		//
		// With AlignBySocket=true, should prefer NUMA nodes within the same socket.
		// Without deep copy (bestCombo = combo), result will be wrong: CPUs from NUMA 0,1,3,6 (cross-socket)
		// Fix: use deep copy bestCombo = append([]int(nil), combo...)
		{
			"distribute 120 CPUs across 4 NUMA nodes within same Socket 0, with AlignBySocket=true from dual socket multi-NUMA with HT (large)",
			topoDualSocketMultiNumaPerSocketHTLarge,
			StaticPolicyOptions{AlignBySocket: true},
			mustParseCPUSet(t, "0-62,64-126,128-190,192-254"),
			120,
			1,
			"",
			mustParseCPUSet(t, "0-14,16-30,32-46,48-62,128-142,144-158,160-174,176-190"),
		},
		// availableCPUs: All 256 CPUs from both sockets
		// Socket 0: NUMA 0: CPU 0-15, 128-143   (32 CPUs);  NUMA 1: CPU 16-31, 144-159   (32 CPUs)
		//           NUMA 2: CPU 32-47, 160-175  (32 CPUs);  NUMA 3: CPU 48-63, 176-191   (32 CPUs)
		// Socket 1: NUMA 4: CPU 64-79, 192-207  (32 CPUs);  NUMA 5: CPU 80-95, 208-223   (32 CPUs)
		//           NUMA 6: CPU 96-111, 224-239 (32 CPUs);  NUMA 7: CPU 112-127, 240-255 (32 CPUs)
		//
		// expResult: 128 CPUs from Socket 0 only (full socket)
		// Socket 0: CPU 0-63, 128-191 (all CPUs from NUMA 0,1,2,3)
		//
		// With AlignBySocket=true, should prefer NUMA nodes within the same socket.
		// Without deep copy (bestCombo = combo), result will be wrong: CPUs from NUMA 0,1,2,7 (cross-socket)
		// Fix: use deep copy bestCombo = append([]int(nil), combo...)
		{
			"distribute 128 CPUs across all NUMA nodes within same Socket 0 from dual socket multi-NUMA with HT (large)",
			topoDualSocketMultiNumaPerSocketHTLarge,
			StaticPolicyOptions{AlignBySocket: true},
			mustParseCPUSet(t, "0-255"),
			128,
			1,
			"",
			mustParseCPUSet(t, "0-63,128-191"),
		},
	}...)

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			result, err := takeByTopologyNUMADistributed(logger, tc.topo, tc.availableCPUs, tc.numCPUs, tc.cpuGroupSize, CPUSortingStrategyPacked, tc.opts.AlignBySocket, 0)
			if err != nil {
				if tc.expErr == "" {
					t.Errorf("unexpected error [%v]", err)
				}
				if tc.expErr != "" && err.Error() != tc.expErr {
					t.Errorf("expected error to be [%v] but it was [%v]", tc.expErr, err)
				}
				return
			}
			if !result.Equals(tc.expResult) {
				t.Errorf("expected result [%s] to equal [%s]", result, tc.expResult)
			}
		})
	}
}

func TestTakeByTopologyNUMADistributedWithMinNUMAHint(t *testing.T) {
	// This test verifies the fix for https://github.com/kubernetes/kubernetes/issues/139430
	// When TopologyManager selects a 4-NUMA affinity (e.g., for GPU alignment),
	// CPUManager should not shrink the allocation to fewer NUMA nodes even if
	// fewer nodes could satisfy the CPU count.
	logger, _ := ktesting.NewTestContext(t)

	testCases := []struct {
		description      string
		topo             *topology.CPUTopology
		availableCPUs    cpuset.CPUSet
		numCPUs          int
		cpuGroupSize     int
		alignBySocket    bool
		minNUMAsFromHint int
		expNUMAs         cpuset.CPUSet // expected set of NUMA node IDs in result
	}{
		{
			// 4 NUMA nodes, 20 CPUs each (80 total). Request 30 CPUs.
			// Without hint: minNUMAs=2 (30 fits in 2 NUMAs of 20 each)
			// With hint=4: must use all 4 NUMAs
			description:      "30 CPUs with 4-NUMA hint should distribute across all 4 NUMAs",
			topo:             topoDualSocketMultiNumaPerSocketHT,
			availableCPUs:    mustParseCPUSet(t, "0-79"),
			numCPUs:          30,
			cpuGroupSize:     1,
			minNUMAsFromHint: 4,
			expNUMAs:         cpuset.New(0, 1, 2, 3),
		},
		{
			// Same topology, no hint (0) → should use minimum NUMAs (2)
			description:      "30 CPUs without hint should pack into minimum NUMAs",
			topo:             topoDualSocketMultiNumaPerSocketHT,
			availableCPUs:    mustParseCPUSet(t, "0-79"),
			numCPUs:          30,
			cpuGroupSize:     1,
			minNUMAsFromHint: 0,
			expNUMAs:         cpuset.New(0, 1),
		},
		{
			// 4 NUMA hint but only need 10 CPUs (fits in 1 NUMA)
			// hint=4 should still force 4-NUMA distribution
			description:      "10 CPUs with 4-NUMA hint should distribute across all 4 NUMAs",
			topo:             topoDualSocketMultiNumaPerSocketHT,
			availableCPUs:    mustParseCPUSet(t, "0-79"),
			numCPUs:          10,
			cpuGroupSize:     1,
			minNUMAsFromHint: 4,
			expNUMAs:         cpuset.New(0, 1, 2, 3),
		},
		{
			// Regression guard: hint exceeds available NUMA nodes (8 > 4).
			// Should gracefully fall back to the original algorithm behavior
			// (minimum NUMAs needed) without panicking or erroring.
			description:      "hint exceeds max NUMAs should fallback to minimum (regression guard)",
			topo:             topoDualSocketMultiNumaPerSocketHT,
			availableCPUs:    mustParseCPUSet(t, "0-79"),
			numCPUs:          30,
			cpuGroupSize:     1,
			minNUMAsFromHint: 8,
			expNUMAs:         cpuset.New(0, 1),
		},
		{
			// Regression guard: hint smaller than computed minNUMAs.
			// 30 CPUs on 4x20 topology → minNUMAs=2. hint=1 is below that.
			// Should NOT reduce below the computed minimum; keeps original behavior.
			description:      "hint smaller than minNUMAs should not reduce distribution (regression guard)",
			topo:             topoDualSocketMultiNumaPerSocketHT,
			availableCPUs:    mustParseCPUSet(t, "0-79"),
			numCPUs:          30,
			cpuGroupSize:     1,
			minNUMAsFromHint: 1,
			expNUMAs:         cpuset.New(0, 1),
		},
		{
			// cpuGroupSize=2 with 4-NUMA hint: verifies that CPU grouping
			// constraint is compatible with NUMA hint enforcement.
			// 20 CPUs with groups of 2 across 4 NUMAs → 5 CPUs per NUMA (valid).
			description:      "cpuGroupSize=2 with 4-NUMA hint should distribute respecting group size",
			topo:             topoDualSocketMultiNumaPerSocketHT,
			availableCPUs:    mustParseCPUSet(t, "0-79"),
			numCPUs:          20,
			cpuGroupSize:     2,
			minNUMAsFromHint: 4,
			expNUMAs:         cpuset.New(0, 1, 2, 3),
		},
		{
			// cpuGroupSize=2 without hint: should use minimum NUMAs.
			// 20 CPUs with groups of 2 on 4x20 topology → fits in 1 NUMA.
			description:      "cpuGroupSize=2 without hint should pack into minimum NUMAs",
			topo:             topoDualSocketMultiNumaPerSocketHT,
			availableCPUs:    mustParseCPUSet(t, "0-79"),
			numCPUs:          20,
			cpuGroupSize:     2,
			minNUMAsFromHint: 0,
			expNUMAs:         cpuset.New(0),
		},
		{
			// alignBySocket=true with 4-NUMA hint: verifies that socket alignment
			// is compatible with NUMA hint enforcement. Topology has 2 sockets
			// with 2 NUMAs each; hint=4 means all NUMAs across both sockets.
			description:      "alignBySocket=true with 4-NUMA hint should distribute across all NUMAs",
			topo:             topoDualSocketMultiNumaPerSocketHT,
			availableCPUs:    mustParseCPUSet(t, "0-79"),
			numCPUs:          40,
			cpuGroupSize:     1,
			alignBySocket:    true,
			minNUMAsFromHint: 4,
			expNUMAs:         cpuset.New(0, 1, 2, 3),
		},
		{
			// NUMA subset constraint: availableCPUs only contains CPUs from
			// NUMA 1,2,3 (not NUMA 0). This simulates the real scenario where
			// TopologyManager's alignedCPUs pre-filters the input pool to only
			// the NUMA nodes selected by device affinity. With hint=3, the
			// allocation should distribute across exactly NUMA 1, 2, 3 and
			// must NOT allocate any CPU from NUMA 0 (which is not in the pool).
			// This proves the NUMA-subset constraint is enforced by input
			// filtering (alignedCPUs) rather than needing the full bitmask.
			// Actual NUMA-CPU mapping for topoDualSocketMultiNumaPerSocketHT:
			// NUMA 0: CPU 0-9, 40-49 (Socket 0)
			// NUMA 1: CPU 10-19, 50-59 (Socket 0)
			// NUMA 2: CPU 20-29, 60-69 (Socket 1)
			// NUMA 3: CPU 30-39, 70-79 (Socket 1)
			description:      "NUMA subset - hint=3 with CPUs only from NUMA 1,2,3 should not use NUMA 0",
			topo:             topoDualSocketMultiNumaPerSocketHT,
			availableCPUs:    mustParseCPUSet(t, "10-39,50-79"), // NUMA 1(10-19,50-59) + NUMA 2(20-29,60-69) + NUMA 3(30-39,70-79)
			numCPUs:          15,
			cpuGroupSize:     1,
			minNUMAsFromHint: 3,
			expNUMAs:         cpuset.New(1, 2, 3),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			result, err := takeByTopologyNUMADistributed(logger, tc.topo, tc.availableCPUs, tc.numCPUs, tc.cpuGroupSize, CPUSortingStrategyPacked, tc.alignBySocket, tc.minNUMAsFromHint)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Collect the set of NUMA node IDs represented in the result.
			numaNodesBuilder := cpuset.New()
			for _, cpuID := range result.UnsortedList() {
				numaNodesBuilder = numaNodesBuilder.Union(cpuset.New(tc.topo.CPUDetails[cpuID].NUMANodeID))
			}

			if !numaNodesBuilder.Equals(tc.expNUMAs) {
				t.Errorf("expected NUMA nodes %s, got %s (result CPUs: %s)",
					tc.expNUMAs, numaNodesBuilder, result)
			}
		})
	}
}
func mustParseCPUSet(t *testing.T, s string) cpuset.CPUSet {
	t.Helper()
	cpus, err := cpuset.Parse(s)
	if err != nil {
		t.Errorf("parsing %q: %v", s, err)
	}
	return cpus
}

func checkError(t *testing.T, expErr string, err error) {
	t.Helper()
	switch {
	case expErr == "" && err != nil:
		t.Errorf("unexpected error [%v]", err)
	case expErr != "" && err == nil:
		t.Errorf("expected error [%v] but got nil", expErr)
	case err != nil && err.Error() != expErr:
		t.Errorf("expected error to be [%v] but it was [%v]", expErr, err)
	}
}

// newUncoreLayersTopology builds one socket, one NUMA node and numCPUs non-SMT CPUs.
// groupSizes are the CPUs per cache group at each layer, outermost first: the first
// is the uncore cache, the others are nested layers. Nested ids are not contiguous.
func newUncoreLayersTopology(numCPUs int, groupSizes ...int) *topology.CPUTopology {
	topo := &topology.CPUTopology{
		NumCPUs:        numCPUs,
		NumCores:       numCPUs,
		NumSockets:     1,
		NumNUMANodes:   1,
		NumUncoreCache: numCPUs / groupSizes[0],
		CPUDetails:     topology.CPUDetails{},
	}
	for cpu := range numCPUs {
		info := topology.CPUInfo{CoreID: cpu, UncoreCacheID: cpu / groupSizes[0]}
		for layer, size := range groupSizes[1:] {
			info.NestedUncoreCacheIDs = append(info.NestedUncoreCacheIDs, 100*(layer+1)+3*(cpu/size))
		}
		topo.CPUDetails[cpu] = info
	}
	return topo
}

// TestTakeByTopologyNestedUncoreCacheAlignment allocates a sequence of requests from
// what is left and checks where each lands: wholly free groups for the bulk of a
// request, the rest where it fits most cleanly, at every nested layer.
func TestTakeByTopologyNestedUncoreCacheAlignment(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	type step struct {
		cpus int
		want string
	}
	testCases := map[string]struct {
		topo               *topology.CPUTopology
		reserved           string
		preferAlignByCache bool
		steps              []step
	}{
		"one-then-two-cpus-the-pair-gets-a-cluster-of-its-own": {
			// clusters of 2 cores, L3 groups of 8 CPUs. CPUs 0-1 reserved.
			topo: newUncoreLayersTopology(16, 8, 2), reserved: "0-1", preferAlignByCache: true,
			steps: []step{{1, "2"}, {2, "4-5"}, {1, "3"}, {2, "6-7"}},
		},
		"odd-request-then-a-pair-then-a-single-cpu-filling-the-hole": {
			topo: newUncoreLayersTopology(16, 8, 2), reserved: "0-1", preferAlignByCache: true,
			steps: []step{{3, "2-4"}, {2, "6-7"}, {1, "5"}},
		},
		"even-requests-never-share-a-cluster-and-spill-to-the-next-l3-group": {
			topo: newUncoreLayersTopology(16, 8, 2), reserved: "0-1", preferAlignByCache: true,
			steps: []step{{2, "2-3"}, {4, "4-7"}, {2, "8-9"}},
		},
		"remainder-of-a-request-spanning-l3-groups-takes-a-whole-cluster": {
			topo: newUncoreLayersTopology(16, 8, 2), reserved: "0-2", preferAlignByCache: true,
			steps: []step{{10, "8-15,4-5"}},
		},
		"clusters-of-4-remainders-fill-partially-used-clusters": {
			// Xeon 6 with E-cores shape: clusters of 4 cores inside one L3 group of 16.
			topo: newUncoreLayersTopology(16, 16, 4), reserved: "", preferAlignByCache: true,
			steps: []step{{3, "0-2"}, {2, "4-5"}, {1, "3"}, {2, "6-7"}},
		},
		"several-nested-layers-fill-free-groups-then-holes": {
			// three nested layers inside one uncore cache of 16 CPUs: groups of 8, 4 and 2
			topo: newUncoreLayersTopology(16, 16, 8, 4, 2), reserved: "", preferAlignByCache: true,
			steps: []step{{1, "0"}, {2, "2-3"}, {1, "1"}, {4, "4-7"}, {8, "8-15"}},
		},
		"request-that-fits-in-one-l3-group-stays-in-it-in-whole-l2-groups": {
			// L2 per 2 cores, L3 per 8 cores, one level 4 cache for all 64 cores
			topo: newUncoreLayersTopology(64, 64, 8, 2), reserved: "", preferAlignByCache: true,
			steps: []step{{2, "0-1"}, {6, "2-7"}, {6, "8-13"}, {4, "16-19"}},
		},
		"request-that-fits-no-partly-used-l3-group-goes-to-a-free-one": {
			topo: newUncoreLayersTopology(64, 64, 8, 2), reserved: "0-2,8-10", preferAlignByCache: true,
			steps: []step{{6, "16-21"}},
		},
		"request-that-fits-no-group-uses-two-groups-and-whole-l2-groups": {
			// L3 groups of 8 CPUs with 1, 4 and 5 free CPUs, none with room for 6: two groups
			// are enough, and 8-11,16-17 are whole L2 groups where 7,16-20 would share two
			topo: newUncoreLayersTopology(24, 24, 8, 2), reserved: "0-6,12-15,21-23", preferAlignByCache: true,
			steps: []step{{6, "8-11,16-17"}},
		},
		"remainder-goes-to-the-l3-group-with-a-whole-l2-group-for-it": {
			// L3 group 0 takes 8. For the other 2, L3 group 1 is the tightest, but its free
			// CPUs 9 and 11 are holes in two L2 groups; L3 group 2 has a free L2 group
			topo: newUncoreLayersTopology(32, 32, 8, 2), reserved: "8,10,12-15", preferAlignByCache: true,
			steps: []step{{10, "0-7,16-17"}},
		},
		"uncore-cache-group-is-chosen-by-how-cleanly-the-request-fits": {
			// uncore group 0 is the tightest, but its free CPUs 5 and 6 are holes in two L2
			// groups; uncore group 1 has a free pair
			topo: newUncoreLayersTopology(16, 8, 2), reserved: "0-4,7-11", preferAlignByCache: true,
			steps: []step{{2, "12-13"}},
		},
		"free-nested-group-beats-an-uncore-cache-group-with-only-holes": {
			// uncore group 1 has exactly 4 free CPUs, all holes; uncore group 0 has a free
			// L3 group
			topo: newUncoreLayersTopology(32, 16, 4, 2), reserved: "12-16,18-20,22-27,29,31", preferAlignByCache: true,
			steps: []step{{4, "0-3"}},
		},
		"spanning-request-leaves-holes-unused-and-takes-whole-l2-groups": {
			// free CPUs in L3 groups of 4: 22-23 and 29-31. Using up the larger group would
			// take the hole 29 and break the pair 22-23
			topo: newUncoreLayersTopology(32, 16, 4, 2), reserved: "0-6,8-21,24-28", preferAlignByCache: true,
			steps: []step{{4, "22-23,30-31"}},
		},
		"spanning-request-touches-as-few-l2-groups-as-it-can": {
			// CPU 3 is a hole, 48-52 has the hole 52 next to the free pairs 48-51, 62-63 is a
			// free pair
			topo: newUncoreLayersTopology(64, 64, 8, 2), reserved: "0-2,4-47,53-61", preferAlignByCache: true,
			steps: []step{{6, "48-51,62-63"}},
		},
		"without-the-option-the-lowest-cpus-are-taken": {
			topo: newUncoreLayersTopology(16, 8, 2), reserved: "0-1", preferAlignByCache: false,
			steps: []step{{1, "2"}, {2, "3-4"}},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			available := tc.topo.CPUDetails.CPUs()
			if tc.reserved != "" {
				available = available.Difference(mustParseCPUSet(t, tc.reserved))
			}
			for i, s := range tc.steps {
				got, err := takeByTopologyNUMAPacked(logger, tc.topo, available, s.cpus, CPUSortingStrategyPacked, tc.preferAlignByCache)
				if err != nil {
					t.Fatalf("step %d (%d CPUs): unexpected error: %v", i+1, s.cpus, err)
				}
				want := cpuset.New()
				if s.want != "" {
					want = mustParseCPUSet(t, s.want)
				}
				if !got.Equals(want) {
					t.Errorf("step %d (%d CPUs): got [%s], want [%s]", i+1, s.cpus, got, want)
				}
				available = available.Difference(got)
			}
		})
	}
}

// With SMT, nested uncore cache ids must not change which CPUs are picked.
func TestTakeByTopologyNestedUncoreCachesIgnoredWithSMT(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	withNested := &topology.CPUTopology{
		NumCPUs:        topoUncoreSingleSocketSMT.NumCPUs,
		NumSockets:     topoUncoreSingleSocketSMT.NumSockets,
		NumCores:       topoUncoreSingleSocketSMT.NumCores,
		NumUncoreCache: topoUncoreSingleSocketSMT.NumUncoreCache,
		CPUDetails:     topology.CPUDetails{},
	}
	for cpu, info := range topoUncoreSingleSocketSMT.CPUDetails {
		info.NestedUncoreCacheIDs = []int{100 + info.CoreID}
		withNested.CPUDetails[cpu] = info
	}

	available := mustParseCPUSet(t, "2-3,10-11,4-7,12-15")
	want, err := takeByTopologyNUMAPacked(logger, topoUncoreSingleSocketSMT, available, 6, CPUSortingStrategyPacked, true)
	if err != nil {
		t.Fatal(err)
	}
	got, err := takeByTopologyNUMAPacked(logger, withNested, available, 6, CPUSortingStrategyPacked, true)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Equals(want) {
		t.Errorf("with nested uncore cache ids got [%s], want [%s] as without", got, want)
	}
}

// A request is satisfied even when one free CPU is all each group has.
func TestTakeByTopologyNestedUncoreCachesRemainderWithoutRoomInOneGroup(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	topo := newUncoreLayersTopology(16, 16, 4) // clusters 0-3, 4-7, 8-11, 12-15
	// one free CPU in every cluster: 3, 7, 11, 15
	available := mustParseCPUSet(t, "3,7,11,15")
	got, err := takeByTopologyNUMAPacked(logger, topo, available, 3, CPUSortingStrategyPacked, true)
	if err != nil {
		t.Fatal(err)
	}
	if want := mustParseCPUSet(t, "3,7,11"); !got.Equals(want) {
		t.Errorf("got [%s], want [%s]", got, want)
	}
}

// On a hybrid CPU (SMT cores next to cores sharing an L2) nested uncore caches are
// not used, so the core based selection stays in charge.
func TestTakeByTopologyNestedUncoreCachesIgnoredOnHybridTopology(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	// two SMT cores (threads 0-1, 2-3), four cores without SMT (threads 4-7) sharing an L2
	hybrid := &topology.CPUTopology{
		NumCPUs:        8,
		NumSockets:     1,
		NumCores:       6,
		NumNUMANodes:   1,
		NumUncoreCache: 1,
		CPUDetails: topology.CPUDetails{
			0: {CoreID: 0, UncoreCacheID: 0},
			1: {CoreID: 0, UncoreCacheID: 0},
			2: {CoreID: 2, UncoreCacheID: 0},
			3: {CoreID: 2, UncoreCacheID: 0},
			4: {CoreID: 4, UncoreCacheID: 0, NestedUncoreCacheIDs: []int{9}},
			5: {CoreID: 5, UncoreCacheID: 0, NestedUncoreCacheIDs: []int{9}},
			6: {CoreID: 6, UncoreCacheID: 0, NestedUncoreCacheIDs: []int{9}},
			7: {CoreID: 7, UncoreCacheID: 0, NestedUncoreCacheIDs: []int{9}},
		},
	}
	// Picking the lowest three CPUs would split the SMT core 2-3; the core based
	// selection takes CPU 4 and the SMT core 0-1.
	available := mustParseCPUSet(t, "0-4")
	got, err := takeByTopologyNUMAPacked(logger, hybrid, available, 3, CPUSortingStrategyPacked, true)
	if err != nil {
		t.Fatal(err)
	}
	if want := mustParseCPUSet(t, "0-1,4"); !got.Equals(want) {
		t.Errorf("got [%s], want [%s]: SMT cores stay whole", got, want)
	}
}

// nestedAllocStats sums how the picks of a simulation sit in the nested uncore
// cache layers (index 0 is the outermost).
type nestedAllocStats struct {
	allocations int
	// touched is the number of groups the picks touch
	touched []int
	// shared is the number of touched groups a pick does not cover
	shared []int
	// excess is touched minus the fewest groups that could hold the request, for
	// requests that fit in one uncore cache group
	excess []int
	// spanning counts the requests that fit in no uncore cache group, with their excess
	spanning       int
	spanningExcess []int
	// worst describes the pick with the largest excess in each layer
	worst       []string
	worstExcess []int
	// violations are broken invariants, with the seed and step
	violations []string
}

// leastGroups is the fewest groups, with the given free CPU counts, that hold n CPUs.
func leastGroups(n int, free []int) int {
	sort.Sort(sort.Reverse(sort.IntSlice(free)))
	for i, f := range free {
		n -= f
		if n <= 0 {
			return i + 1
		}
	}
	return len(free) + 1
}

// simulateNestedAllocations runs a seeded random sequence of allocations and releases
// and records how every pick sits in the nested layers of topo. With align false the
// nested layers are ignored, as a baseline.
func simulateNestedAllocations(t *testing.T, topo *topology.CPUTopology, seed int64, steps int, align bool) nestedAllocStats {
	logger, _ := ktesting.NewTestContext(t)
	layers := topo.CPUDetails.NestedUncoreCacheLayers()
	stats := nestedAllocStats{touched: make([]int, layers), shared: make([]int, layers), excess: make([]int, layers), spanningExcess: make([]int, layers),
		worst: make([]string, layers), worstExcess: make([]int, layers)}

	pickTopo := topo
	if !align {
		flat := *topo
		flat.CPUDetails = topology.CPUDetails{}
		for cpu, info := range topo.CPUDetails {
			info.NestedUncoreCacheIDs = nil
			flat.CPUDetails[cpu] = info
		}
		pickTopo = &flat
	}

	groups := make([]map[int]cpuset.CPUSet, layers)
	for l := range groups {
		groups[l] = topo.CPUDetails.NestedUncoreCacheGroups(l)
	}

	rng := rand.New(rand.NewSource(seed))
	sizes := []int{1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 6, 8, 10, 12, 16}
	free := topo.CPUDetails.CPUs()
	var live []cpuset.CPUSet
	for step := range steps {
		if len(live) > 0 && rng.Intn(100) < 45 {
			i := rng.Intn(len(live))
			free = free.Union(live[i])
			live = append(live[:i], live[i+1:]...)
			continue
		}
		n := sizes[rng.Intn(len(sizes))]
		if n > free.Size() {
			continue
		}
		pick, err := takeByTopologyNUMAPacked(logger, pickTopo, free, n, CPUSortingStrategyPacked, true)
		where := fmt.Sprintf("seed %d step %d: %d CPUs from free [%s]", seed, step, n, free)
		switch {
		case err != nil:
			stats.violations = append(stats.violations, where+": error "+err.Error())
			continue
		case pick.Size() != n:
			stats.violations = append(stats.violations, fmt.Sprintf("%s: got %d CPUs [%s]", where, pick.Size(), pick))
			continue
		case !pick.IsSubsetOf(free):
			stats.violations = append(stats.violations, fmt.Sprintf("%s: got [%s], not all free", where, pick))
			continue
		}

		stats.allocations++
		fitsOneUncore := false
		for _, id := range topo.CPUDetails.UncoreCaches().List() {
			if topo.CPUDetails.CPUsInUncoreCaches(id).Intersection(free).Size() >= n {
				fitsOneUncore = true
			}
		}
		if !fitsOneUncore {
			stats.spanning++
		}
		for l := range layers {
			// the fewest groups that could hold the request, inside one uncore cache group
			// if it fits in one
			least := -1
			if fitsOneUncore {
				for _, id := range topo.CPUDetails.UncoreCaches().List() {
					uncore := topo.CPUDetails.CPUsInUncoreCaches(id)
					if uncore.Intersection(free).Size() < n {
						continue
					}
					var freeCounts []int
					for _, g := range groups[l] {
						if g.IsSubsetOf(uncore) {
							freeCounts = append(freeCounts, g.Intersection(free).Size())
						}
					}
					if k := leastGroups(n, freeCounts); least < 0 || k < least {
						least = k
					}
				}
			} else {
				var freeCounts []int
				for _, g := range groups[l] {
					freeCounts = append(freeCounts, g.Intersection(free).Size())
				}
				least = leastGroups(n, freeCounts)
			}
			touched, shared := 0, 0
			for _, g := range groups[l] {
				if !g.Intersection(pick).IsEmpty() {
					touched++
					if !g.IsSubsetOf(pick) {
						shared++
					}
				}
			}
			excess := touched - least
			stats.touched[l] += touched
			stats.shared[l] += shared
			if !fitsOneUncore {
				stats.spanningExcess[l] += excess
				continue
			}
			stats.excess[l] += excess
			if excess > stats.worstExcess[l] {
				stats.worstExcess[l] = excess
				stats.worst[l] = fmt.Sprintf("%s: got [%s], touches %d groups, %d would do", where, pick, touched, touched-excess)
			}
		}
		free = free.Difference(pick)
		live = append(live, pick)
	}
	return stats
}

// TestNestedUncoreCacheAllocationFuzz runs containers of different sizes that come
// and go, which leaves holes, and looks for picks that touch more groups than needed
// or leave groups shared. Runs are seeded; a failure names the seed and step.
func TestNestedUncoreCacheAllocationFuzz(t *testing.T) {
	topologies := map[string]struct {
		numCPUs    int
		groupSizes []int
	}{
		"l3-of-8-and-l2-of-2-in-16-cpus":                  {16, []int{8, 2}},
		"l4-of-16-l3-of-4-and-l2-of-2-in-32-cpus":         {32, []int{16, 4, 2}},
		"l4-of-64-l3-of-8-and-l2-of-2-in-64-cpus":         {64, []int{64, 8, 2}},
		"l3-of-18-and-l2-of-2-in-72-cpus":                 {72, []int{72, 18, 2}},
		"l4-of-24-l3-of-8-l2-of-4-and-l1-of-2-in-48-cpus": {48, []int{24, 8, 4, 2}},
	}
	const runs, steps = 20, 250

	for name, tc := range topologies {
		t.Run(name, func(t *testing.T) {
			topo := newUncoreLayersTopology(tc.numCPUs, tc.groupSizes...)
			var nested, flat nestedAllocStats
			for seed := int64(1); seed <= runs; seed++ {
				n := simulateNestedAllocations(t, topo, seed, steps, true)
				f := simulateNestedAllocations(t, topo, seed, steps, false)
				if nested.touched == nil {
					layers := len(n.touched)
					nested.touched, nested.shared, nested.excess, nested.spanningExcess = make([]int, layers), make([]int, layers), make([]int, layers), make([]int, layers)
					flat.touched, flat.shared, flat.excess, flat.spanningExcess = make([]int, layers), make([]int, layers), make([]int, layers), make([]int, layers)
				}
				nested.allocations += n.allocations
				flat.allocations += f.allocations
				nested.spanning += n.spanning
				flat.spanning += f.spanning
				for l := range n.touched {
					nested.touched[l] += n.touched[l]
					nested.shared[l] += n.shared[l]
					nested.excess[l] += n.excess[l]
					flat.touched[l] += f.touched[l]
					flat.shared[l] += f.shared[l]
					flat.excess[l] += f.excess[l]
					nested.spanningExcess[l] += n.spanningExcess[l]
					flat.spanningExcess[l] += f.spanningExcess[l]
				}
				nested.violations = append(nested.violations, n.violations...)
				if nested.worst == nil {
					nested.worst, nested.worstExcess = make([]string, len(n.worst)), make([]int, len(n.worst))
				}
				for l := range n.worst {
					if n.worstExcess[l] > nested.worstExcess[l] {
						nested.worstExcess[l], nested.worst[l] = n.worstExcess[l], n.worst[l]
					}
				}
			}
			for _, v := range nested.violations {
				t.Error(v)
			}
			t.Logf("%d allocations, %d of them fit in no single uncore cache group (without nested layers: %d, %d)",
				nested.allocations, nested.spanning, flat.allocations, flat.spanning)
			for l := range nested.touched {
				t.Logf("layer %d: groups touched %d (without nesting %d), shared %d (%d); over the least possible: %d (%d), spanning requests %d (%d)",
					l, nested.touched[l], flat.touched[l], nested.shared[l], flat.shared[l],
					nested.excess[l], flat.excess[l], nested.spanningExcess[l], flat.spanningExcess[l])
			}
			for l, w := range nested.worst {
				if w != "" {
					t.Logf("worst in layer %d: %s", l, w)
				}
			}

			for l := range nested.touched {
				if nested.touched[l] > flat.touched[l] || nested.shared[l] > flat.shared[l] {
					t.Errorf("layer %d: nested layers touch %d groups and share %d, more than without them (%d, %d)",
						l, nested.touched[l], nested.shared[l], flat.touched[l], flat.shared[l])
				}
			}
			// the outermost nested layer uses the fewest groups, the layers below only rarely more
			if nested.excess[0] != 0 {
				t.Errorf("layer 0: %d groups touched over the fewest possible, want 0; worst: %s", nested.excess[0], nested.worst[0])
			}
			for l := 1; l < len(nested.excess); l++ {
				if nested.excess[l]*100 > nested.allocations {
					t.Errorf("layer %d: %d groups touched over the fewest possible in %d allocations, more than 1%%; worst: %s",
						l, nested.excess[l], nested.allocations, nested.worst[l])
				}
			}
		})
	}
}
