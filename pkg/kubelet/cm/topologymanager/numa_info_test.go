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
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
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
				},
				{
					Id: 1,
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
			expectedErr:      fmt.Errorf("no such file or directory"),
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
				},
				{
					Id: 2,
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
				},
				{
					Id: 2,
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

	nodeDir, err := os.MkdirTemp("", "TestNUMAInfo")
	if err != nil {
		t.Fatalf("Unable to create temporary directory: %v", err)
	}
	defer os.RemoveAll(nodeDir)

	numaDistances := map[int]string{
		0: "10 11 12 12",
		1: "11 10 12 12",
		2: "12 12 10 11",
		3: "12 12 11 10",
	}

	for i, distances := range numaDistances {
		numaDir := filepath.Join(nodeDir, fmt.Sprintf("node%d", i))
		if err := os.Mkdir(numaDir, 0700); err != nil {
			t.Fatalf("Unable to create numaDir %s: %v", numaDir, err)
		}

		distanceFile := filepath.Join(numaDir, "distance")

		if err = os.WriteFile(distanceFile, []byte(distances), 0644); err != nil {
			t.Fatalf("Unable to create test distanceFile: %v", err)
		}
	}

	// stub sysFs to read from temp dir
	sysFs := &NUMASysFs{nodeDir: nodeDir}

	for _, tcase := range tcases {
		topology, err := newNUMAInfo(tcase.topology, sysFs, tcase.opts)
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

func TestGetDistances(t *testing.T) {
	testCases := []struct {
		name              string
		expectedErr       bool
		expectedDistances []uint64
		nodeId            int
		nodeExists        bool
	}{
		{
			name:              "reading proper distance file",
			expectedErr:       false,
			expectedDistances: []uint64{10, 11, 12, 13},
			nodeId:            0,
			nodeExists:        true,
		},
		{
			name:              "no distance file",
			expectedErr:       true,
			expectedDistances: nil,
			nodeId:            99,
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.name, func(t *testing.T) {
			var err error

			nodeDir, err := os.MkdirTemp("", "TestGetDistances")
			if err != nil {
				t.Fatalf("Unable to create temporary directory: %v", err)
			}

			defer os.RemoveAll(nodeDir)

			if tcase.nodeExists {
				numaDir := filepath.Join(nodeDir, fmt.Sprintf("node%d", tcase.nodeId))
				if err := os.Mkdir(numaDir, 0700); err != nil {
					t.Fatalf("Unable to create numaDir %s: %v", numaDir, err)
				}

				distanceFile := filepath.Join(numaDir, "distance")

				var buffer bytes.Buffer
				for i, distance := range tcase.expectedDistances {
					buffer.WriteString(strconv.Itoa(int(distance)))
					if i != len(tcase.expectedDistances)-1 {
						buffer.WriteString(" ")
					}
				}

				if err = os.WriteFile(distanceFile, buffer.Bytes(), 0644); err != nil {
					t.Fatalf("Unable to create test distanceFile: %v", err)
				}
			}

			sysFs := &NUMASysFs{nodeDir: nodeDir}

			distances, err := sysFs.GetDistances(tcase.nodeId)
			if !tcase.expectedErr && err != nil {
				t.Fatalf("Expected err to equal nil, not %v", err)
			} else if tcase.expectedErr && err == nil {
				t.Fatalf("Expected err to equal %v, not nil", tcase.expectedErr)
			}

			if !tcase.expectedErr && !reflect.DeepEqual(distances, tcase.expectedDistances) {
				t.Fatalf("Expected distances to equal %v, not %v", tcase.expectedDistances, distances)
			}
		})
	}
}

func TestSplitDistances(t *testing.T) {
	tcases := []struct {
		description  string
		rawDistances string
		expected     []uint64
		expectedErr  error
	}{
		{
			description:  "read one distance",
			rawDistances: "10",
			expected:     []uint64{10},
			expectedErr:  nil,
		},
		{
			description:  "read two distances",
			rawDistances: "10 20",
			expected:     []uint64{10, 20},
			expectedErr:  nil,
		},
		{
			description:  "can't convert negative number to uint64",
			rawDistances: "10 -20",
			expected:     nil,
			expectedErr:  fmt.Errorf("cannot conver"),
		},
	}

	for _, tc := range tcases {
		result, err := splitDistances(tc.rawDistances)

		if tc.expectedErr == nil && err != nil {
			t.Fatalf("Expected err to equal nil, not %v", err)
		} else if tc.expectedErr != nil && err == nil {
			t.Fatalf("Expected err to equal %v, not nil", tc.expectedErr)
		} else if tc.expectedErr != nil {
			if !strings.Contains(err.Error(), tc.expectedErr.Error()) {
				t.Errorf("Unexpected error message. Have: %s wants %s", err.Error(), tc.expectedErr.Error())
			}
		}

		if !reflect.DeepEqual(tc.expected, result) {
			t.Fatalf("Expected distances to equal: %v, got: %v", tc.expected, result)
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
