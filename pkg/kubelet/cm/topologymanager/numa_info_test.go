/*
Copyright 2022 The Kubernetes Authors.

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

package topologymanager

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

func TestNUMAInfo(t *testing.T) {
	tcases := []struct {
		name             string
		topology         []cadvisorapi.Node
		expectedNUMAInfo *NUMAInfo
		expectedErr      error
		opts             PolicyOptions
	}{
		{
			name: "positive test 1 node",
			topology: []cadvisorapi.Node{
				{
					Id: 0,
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{0},
				NUMADistances: NUMADistances{
					0: nil,
				},
			},
			opts: PolicyOptions{},
		},
		{
			name: "positive test 1 node, with PreferClosestNUMA",
			topology: []cadvisorapi.Node{
				{
					Id: 0,
					Distances: []uint64{
						10,
						11,
						12,
						12,
					},
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{0},
				NUMADistances: NUMADistances{
					0: {
						10,
						11,
						12,
						12,
					},
				},
			},
			opts: PolicyOptions{
				PreferClosestNUMA: true,
			},
		},
		{
			name: "positive test 2 nodes",
			topology: []cadvisorapi.Node{
				{
					Id: 0,
				},
				{
					Id: 1,
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{0, 1},
				NUMADistances: NUMADistances{
					0: nil,
					1: nil,
				},
			},
		},
		{
			name: "positive test 2 nodes, with PreferClosestNUMA",
			topology: []cadvisorapi.Node{
				{
					Id: 0,
					Distances: []uint64{
						10,
						11,
						12,
						12,
					},
				},
				{
					Id: 1,
					Distances: []uint64{
						11,
						10,
						12,
						12,
					},
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{0, 1},
				NUMADistances: NUMADistances{
					0: {
						10,
						11,
						12,
						12,
					},
					1: {
						11,
						10,
						12,
						12,
					},
				},
			},
			opts: PolicyOptions{
				PreferClosestNUMA: true,
			},
		},
		{
			name: "positive test 3 nodes",
			topology: []cadvisorapi.Node{
				{
					Id: 0,
				},
				{
					Id: 1,
				},
				{
					Id: 2,
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{0, 1, 2},
				NUMADistances: NUMADistances{
					0: nil,
					1: nil,
					2: nil,
				},
			},
		},
		{
			name: "positive test 3 nodes, with PreferClosestNUMA",
			topology: []cadvisorapi.Node{
				{
					Id: 0,
					Distances: []uint64{
						10,
						11,
						12,
						12,
					},
				},
				{
					Id: 1,
					Distances: []uint64{
						11,
						10,
						12,
						12,
					},
				},
				{
					Id: 2,
					Distances: []uint64{
						12,
						12,
						10,
						11,
					},
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{0, 1, 2},
				NUMADistances: NUMADistances{
					0: {
						10,
						11,
						12,
						12,
					},
					1: {
						11,
						10,
						12,
						12,
					},
					2: {
						12,
						12,
						10,
						11,
					},
				},
			},
			opts: PolicyOptions{
				PreferClosestNUMA: true,
			},
		},
		{
			name: "positive test 4 nodes",
			topology: []cadvisorapi.Node{
				{
					Id: 0,
				},
				{
					Id: 1,
				},
				{
					Id: 2,
				},
				{
					Id: 3,
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{0, 1, 2, 3},
				NUMADistances: NUMADistances{
					0: nil,
					1: nil,
					2: nil,
					3: nil,
				},
			},
		},
		{
			name: "positive test 4 nodes, with PreferClosestNUMA",
			topology: []cadvisorapi.Node{
				{
					Id: 0,
					Distances: []uint64{
						10,
						11,
						12,
						12,
					},
				},
				{
					Id: 1,
					Distances: []uint64{
						11,
						10,
						12,
						12,
					},
				},
				{
					Id: 2,
					Distances: []uint64{
						12,
						12,
						10,
						11,
					},
				},
				{
					Id: 3,
					Distances: []uint64{
						12,
						12,
						11,
						10,
					},
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{0, 1, 2, 3},
				NUMADistances: NUMADistances{
					0: {
						10,
						11,
						12,
						12,
					},
					1: {
						11,
						10,
						12,
						12,
					},
					2: {
						12,
						12,
						10,
						11,
					},
					3: {
						12,
						12,
						11,
						10,
					},
				},
			},
			opts: PolicyOptions{
				PreferClosestNUMA: true,
			},
		},
		{
			name: "negative test 1 node, no distance file with PreferClosestNUMA",
			topology: []cadvisorapi.Node{
				{
					Id: 9,
				},
			},
			expectedNUMAInfo: nil,
			expectedErr:      fmt.Errorf("error getting NUMA distances from cadvisor"),
			opts: PolicyOptions{
				PreferClosestNUMA: true,
			},
		},
		{
			name: "one node and its id is 1",
			topology: []cadvisorapi.Node{
				{
					Id: 1,
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{1},
				NUMADistances: NUMADistances{
					1: nil,
				},
			},
		},
		{
			name: "one node and its id is 1, with PreferClosestNUMA",
			topology: []cadvisorapi.Node{
				{
					Id: 1,
					Distances: []uint64{
						11,
						10,
						12,
						12,
					},
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{1},
				NUMADistances: NUMADistances{
					1: {
						11,
						10,
						12,
						12,
					},
				},
			},
			opts: PolicyOptions{
				PreferClosestNUMA: true,
			},
		},
		{
			name: "two nodes not sequential",
			topology: []cadvisorapi.Node{
				{
					Id: 0,
					Distances: []uint64{
						10,
						11,
						12,
						12,
					},
				},
				{
					Id: 2,
					Distances: []uint64{
						12,
						12,
						10,
						11,
					},
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{0, 2},
				NUMADistances: NUMADistances{
					0: nil,
					2: nil,
				},
			},
		},
		{
			name: "two nodes not sequential, with PreferClosestNUMA",
			topology: []cadvisorapi.Node{
				{
					Id: 0,
					Distances: []uint64{
						10,
						11,
						12,
						12,
					},
				},
				{
					Id: 2,
					Distances: []uint64{
						12,
						12,
						10,
						11,
					},
				},
			},
			expectedNUMAInfo: &NUMAInfo{
				Nodes: []int{0, 2},
				NUMADistances: NUMADistances{
					0: {
						10,
						11,
						12,
						12,
					},
					2: {
						12,
						12,
						10,
						11,
					},
				},
			},
			opts: PolicyOptions{
				PreferClosestNUMA: true,
			},
		},
	}

	for _, tcase := range tcases {
		topology, err := NewNUMAInfo(tcase.topology, tcase.opts)
		if tcase.expectedErr == nil && err != nil {
			t.Fatalf("Expected err to equal nil, not %v", err)
		} else if tcase.expectedErr != nil && err == nil {
			t.Fatalf("Expected err to equal %v, not nil", tcase.expectedErr)
		} else if tcase.expectedErr != nil {
			if !strings.Contains(err.Error(), tcase.expectedErr.Error()) {
				t.Errorf("Unexpected error message. Have: %s wants %s", err.Error(), tcase.expectedErr.Error())
			}
		}

		if !reflect.DeepEqual(topology, tcase.expectedNUMAInfo) {
			t.Fatalf("Expected topology to equal %v, not %v", tcase.expectedNUMAInfo, topology)
		}

	}
}

func TestCalculateAvgDistanceFor(t *testing.T) {
	tcases := []struct {
		name        string
		bm          []int
		distance    NUMADistances
		expectedAvg float64
	}{
		{
			name: "1 NUMA node",
			bm: []int{
				0,
			},
			distance: NUMADistances{
				0: {
					10,
				},
			},
			expectedAvg: 10,
		},
		{
			name: "2 NUMA node, 1 set in bitmask",
			bm: []int{
				0,
			},
			distance: NUMADistances{
				0: {
					10,
					11,
				},
				1: {
					11,
					10,
				},
			},
			expectedAvg: 10,
		},
		{
			name: "2 NUMA node, 2 set in bitmask",
			bm: []int{
				0,
				1,
			},
			distance: NUMADistances{
				0: {
					10,
					11,
				},
				1: {
					11,
					10,
				},
			},
			expectedAvg: 10.5,
		},
		{
			name: "4 NUMA node, 2 set in bitmask",
			bm: []int{
				0,
				2,
			},
			distance: NUMADistances{
				0: {
					10,
					11,
					12,
					12,
				},
				1: {
					11,
					10,
					12,
					12,
				},
				2: {
					12,
					12,
					10,
					11,
				},
				3: {
					12,
					12,
					11,
					10,
				},
			},
			expectedAvg: 11,
		},
		{
			name: "4 NUMA node, 3 set in bitmask",
			bm: []int{
				0,
				2,
				3,
			},
			distance: NUMADistances{
				0: {
					10,
					11,
					12,
					12,
				},
				1: {
					11,
					10,
					12,
					12,
				},
				2: {
					12,
					12,
					10,
					11,
				},
				3: {
					12,
					12,
					11,
					10,
				},
			},
			expectedAvg: 11.11111111111111,
		},
		{
			name:        "0 NUMA node, 0 set in bitmask",
			bm:          []int{},
			distance:    NUMADistances{},
			expectedAvg: 0,
		},
	}

	for _, tcase := range tcases {
		bm, err := bitmask.NewBitMask(tcase.bm...)
		if err != nil {
			t.Errorf("no error expected got %v", err)
		}

		numaInfo := NUMAInfo{
			Nodes:         tcase.bm,
			NUMADistances: tcase.distance,
		}

		result := numaInfo.NUMADistances.CalculateAverageFor(bm)
		if result != tcase.expectedAvg {
			t.Errorf("Expected result to equal %g, not %g", tcase.expectedAvg, result)
		}
	}

}

func TestClosest(t *testing.T) {
	tcases := []struct {
		description string
		current     bitmask.BitMask
		candidate   bitmask.BitMask
		expected    string
		numaInfo    *NUMAInfo
	}{
		{
			description: "current and candidate length is not the same, current narrower",
			current:     NewTestBitMask(0),
			candidate:   NewTestBitMask(0, 2),
			expected:    "current",
			numaInfo:    &NUMAInfo{},
		},
		{
			description: "current and candidate length is the same, distance is the same, current more lower bits set",
			current:     NewTestBitMask(0, 1),
			candidate:   NewTestBitMask(0, 2),
			expected:    "current",
			numaInfo: &NUMAInfo{
				NUMADistances: NUMADistances{
					0: {10, 10, 10},
					1: {10, 10, 10},
					2: {10, 10, 10},
				},
			},
		},
		{
			description: "current and candidate length is the same, distance is the same, candidate more lower bits set",
			current:     NewTestBitMask(0, 3),
			candidate:   NewTestBitMask(0, 2),
			expected:    "candidate",
			numaInfo: &NUMAInfo{
				NUMADistances: NUMADistances{
					0: {10, 10, 10, 10},
					1: {10, 10, 10, 10},
					2: {10, 10, 10, 10},
					3: {10, 10, 10, 10},
				},
			},
		},
		{
			description: "current and candidate length is the same, candidate average distance is smaller",
			current:     NewTestBitMask(0, 3),
			candidate:   NewTestBitMask(0, 1),
			expected:    "candidate",
			numaInfo: &NUMAInfo{
				NUMADistances: NUMADistances{
					0: {10, 11, 12, 12},
					1: {11, 10, 12, 12},
					2: {12, 12, 10, 11},
					3: {12, 12, 11, 10},
				},
			},
		},
		{
			description: "current and candidate length is the same, current average distance is smaller",
			current:     NewTestBitMask(2, 3),
			candidate:   NewTestBitMask(0, 3),
			expected:    "current",
			numaInfo: &NUMAInfo{
				NUMADistances: NUMADistances{
					0: {10, 11, 12, 12},
					1: {11, 10, 12, 12},
					2: {12, 12, 10, 11},
					3: {12, 12, 11, 10},
				},
			},
		},
	}

	for _, tc := range tcases {
		t.Run(tc.description, func(t *testing.T) {

			result := tc.numaInfo.Closest(tc.candidate, tc.current)
			if result != tc.current && result != tc.candidate {
				t.Errorf("Expected result to be either 'current' or 'candidate' hint")
			}
			if tc.expected == "current" && result != tc.current {
				t.Errorf("Expected result to be %v, got %v", tc.current, result)
			}
			if tc.expected == "candidate" && result != tc.candidate {
				t.Errorf("Expected result to be %v, got %v", tc.candidate, result)
			}
		})
	}
}
