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
