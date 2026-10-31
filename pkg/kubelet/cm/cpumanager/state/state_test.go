/*
Copyright 2021 The Kubernetes Authors.

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

package state

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/utils/cpuset"
)

func TestClone(t *testing.T) {
	expect := ContainerCPUAssignments{
		"pod": map[string]cpuset.CPUSet{
			"container1": cpuset.New(4, 5, 6),
			"container2": cpuset.New(1, 2, 3),
		},
	}
	actual := expect.Clone()
	if &expect == &actual || !reflect.DeepEqual(expect, actual) {
		t.Fail()
	}
}

func TestString(t *testing.T) {
	testCases := []struct {
		name           string
		assignments    ContainerCPUAssignments
		expectedOutput string
	}{
		{
			name: "two pods with multiple containers",
			assignments: ContainerCPUAssignments{
				"pod1": {
					"container1": cpuset.New(1, 2),
					"container2": cpuset.New(3, 4),
				},
				"pod2": {
					"container3": cpuset.New(5),
				},
			},
			expectedOutput: `{"pod1":{"container1":"1-2","container2":"3-4"},"pod2":{"container3":"5"}}`,
		},
		{
			name: "three pods with multiple containers",
			assignments: ContainerCPUAssignments{
				"pod1": {
					"container1": cpuset.New(1, 5),
				},
				"pod3": {
					"container5": cpuset.New(9),
					"container2": cpuset.New(3, 4, 5, 6, 7, 8),
				},
				"pod2": {
					"container3": cpuset.New(5),
					"container2": cpuset.New(3, 8),
				},
			},
			expectedOutput: `{"pod1":{"container1":"1,5"},"pod2":{"container2":"3,8","container3":"5"},"pod3":{"container2":"3-8","container5":"9"}}`,
		},
		{
			name:           "empty assignments",
			assignments:    ContainerCPUAssignments{},
			expectedOutput: `{}`,
		},
		{
			name: "single pod single container",
			assignments: ContainerCPUAssignments{
				"pod1": {
					"container1": cpuset.New(0, 1, 2),
				},
			},
			expectedOutput: `{"pod1":{"container1":"0-2"}}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.assignments.String(), tc.expectedOutput); diff != "" {
				t.Errorf("String() output mismatch.\n diff=%s got(+) want(-)", diff)
			}
		})
	}
}

func TestCloneWithInPlacePodVerticalScalingExclusiveCPUs(t *testing.T) {
	expect := ContainerCPUBaselines{
		"pod": map[string]ContainerCPUBaseline{
			"container1": {Baseline: cpuset.New(4, 5, 6)},
			"container2": {Baseline: cpuset.New(1, 2, 3)},
		},
	}
	actual := expect.Clone()
	if &expect == &actual || !reflect.DeepEqual(expect, actual) {
		t.Fail()
	}
}

// Test contains artificial data that cannot be produced in real scenarios for easier verification of PendingResizeDelta function.
func TestPendingResizeDelta(t *testing.T) {
	// c1 has {1,2}, c2 has {4,6} and the pod-level assignment {1,2,3,4} holds
	// both, except CPU 6.
	assignments := ContainerCPUAssignments{
		"pod": {
			"c1": cpuset.New(1, 2),
			"c2": cpuset.New(4, 6),
		},
	}
	podAssignments := PodCPUAssignments{
		"pod": {CPUSet: cpuset.New(1, 2, 3, 4)},
	}
	pod := cpuset.New(1, 2, 3, 4, 5)

	tests := []struct {
		name           string
		pending        PendingResize
		expectTaken    cpuset.CPUSet
		expectReleased cpuset.CPUSet
	}{
		{
			name: "containers and pod-level assignment",
			pending: PendingResize{
				PodUID:           "pod",
				ContainerCPUSets: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 3), "c2": cpuset.New(4)},
				PodCPUSet:        &pod,
			},
			expectTaken:    cpuset.New(5),
			expectReleased: cpuset.New(6),
		},
		{
			name:           "a CPU moving between containers of the pod",
			pending:        PendingResize{PodUID: "pod", ContainerCPUSets: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 6), "c2": cpuset.New(4)}},
			expectTaken:    cpuset.New(),
			expectReleased: cpuset.New(),
		},
		{
			name:           "a CPU coming from the pod-level assignment",
			pending:        PendingResize{PodUID: "pod", ContainerCPUSets: map[string]cpuset.CPUSet{"c1": cpuset.New(1, 2, 3)}},
			expectTaken:    cpuset.New(),
			expectReleased: cpuset.New(),
		},
		{
			name:           "the pod-level assignment only",
			pending:        PendingResize{PodUID: "pod", PodCPUSet: &pod},
			expectTaken:    cpuset.New(5),
			expectReleased: cpuset.New(),
		},
		{
			name:           "a pod without assignments",
			pending:        PendingResize{PodUID: "new", ContainerCPUSets: map[string]cpuset.CPUSet{"c": cpuset.New(7)}},
			expectTaken:    cpuset.New(7),
			expectReleased: cpuset.New(),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			taken, released := PendingResizeDelta(assignments, podAssignments, tc.pending)
			if !taken.Equals(tc.expectTaken) || !released.Equals(tc.expectReleased) {
				t.Errorf("Expected taken %s and released %s, got %s and %s", tc.expectTaken, tc.expectReleased, taken, released)
			}
		})
	}
}
