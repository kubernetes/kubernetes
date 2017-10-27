/*
Copyright 2015 The Kubernetes Authors.

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

package node

import (
	"reflect"

	"k8s.io/api/core/v1"
	"testing"
)

func TestGetNodeCondition(t *testing.T) {
	nodeStatus := v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{
				Type:   v1.NodeReady,
				Status: v1.ConditionTrue,
			},
			{
				Type:   v1.NodeOutOfDisk,
				Status: v1.ConditionFalse,
			},
			{
				Type:   v1.NodeMemoryPressure,
				Status: v1.ConditionFalse,
			},
		},
	}
	tests := []struct {
		name          string
		condition     v1.NodeCondition
		expectedIndex int
	}{
		{
			name: "index 0",
			condition: v1.NodeCondition{
				Type:   v1.NodeReady,
				Status: v1.ConditionTrue},
			expectedIndex: 0,
		},
		{
			name: "index 1",
			condition: v1.NodeCondition{
				Type:   v1.NodeOutOfDisk,
				Status: v1.ConditionFalse},
			expectedIndex: 1,
		},
		{
			name: "index 2",
			condition: v1.NodeCondition{
				Type:   v1.NodeMemoryPressure,
				Status: v1.ConditionFalse},
			expectedIndex: 2,
		},
		{
			name: "index -1",
			condition: v1.NodeCondition{
				Type:   v1.NodeDiskPressure,
				Status: v1.ConditionFalse},
			expectedIndex: -1,
		},
	}

	for _, test := range tests {
		index, condition := GetNodeCondition(&nodeStatus, test.condition.Type)

		if test.expectedIndex != index {
			t.Errorf("unexpected result of test: %v, expected index: %v, actual index: %v", test.name, test.expectedIndex, index)
		}
		if test.expectedIndex != -1 {
			if !reflect.DeepEqual(test.condition, *condition) {
				t.Errorf("unexpected result of test: %v, expected condition: %v, actual condition: %v", test.name, test.condition, *condition)
			}
		} else {
			if condition != nil {
				t.Errorf("unexpected result of test: %v, expect nil, but got: %v", test.name, condition)
			}
		}
	}
}

func TestIsNodeReady(t *testing.T) {
	tests := []struct {
		node     v1.Node
		expected bool
	}{

		{
			node: v1.Node{
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionTrue,
						},
					},
				}},
			expected: true,
		},
		{
			node: v1.Node{
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:   v1.NodeReady,
							Status: v1.ConditionFalse,
						},
						{
							Type:   v1.NodeOutOfDisk,
							Status: v1.ConditionTrue,
						},
						{
							Type:   v1.NodeDiskPressure,
							Status: v1.ConditionFalse,
						},
					},
				}},
			expected: false,
		},
	}

	for _, test := range tests {
		result := IsNodeReady(&test.node)
		if test.expected != result {
			t.Errorf("unexpected result of test: %v, expected: %v, actual: %v", test, test.expected, result)
		}
	}
}
