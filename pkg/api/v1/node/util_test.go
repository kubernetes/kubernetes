/*
Copyright 2018 The Kubernetes Authors.

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
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/diff"
)

func TestGetNodeCondition(t *testing.T) {
	statusIns := &v1.NodeStatus{
		Conditions: []v1.NodeCondition{
			{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue},
			{Type: v1.NodeReady, Status: v1.ConditionTrue},
		},
	}

	const nctReadyIns = v1.NodeConditionType("Ready")
	const nctNotReadyIns = v1.NodeConditionType("NotReady")

	tests := map[string]struct {
		status          *v1.NodeStatus
		nct             v1.NodeConditionType
		expectValue     int
		expectCondition *v1.NodeCondition
	}{
		"successfully get node condition": {
			status:          statusIns,
			nct:             nctReadyIns,
			expectValue:     1,
			expectCondition: &v1.NodeCondition{Type: v1.NodeReady, Status: v1.ConditionTrue},
		},
		"unable to get node condition": {
			status:          statusIns,
			nct:             nctNotReadyIns,
			expectValue:     -1,
			expectCondition: nil,
		},
	}

	for k, v := range tests {
		actualValue, actualCondition := GetNodeCondition(v.status, v.nct)
		if actualValue != v.expectValue {
			t.Errorf("%s failed, expected %d but received %d", k, v.expectValue, actualValue)
		}
		if !reflect.DeepEqual(v.expectCondition, actualCondition) {
			t.Errorf("Got different run options than expected.\nDifference detected on:\n%s", diff.ObjectReflectDiff(v.expectCondition, actualCondition))
		}
	}
}

func TestIsNodeReady(t *testing.T) {

	nodeReadyIns := &v1.Node{
		Spec: v1.NodeSpec{},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   v1.NodeOutOfDisk,
					Status: v1.ConditionFalse,
				},
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
		},
	}

	nodeNotReadyIns := &v1.Node{
		Spec: v1.NodeSpec{},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   v1.NodeMemoryPressure,
					Status: v1.ConditionFalse,
				},
				{
					Type:   v1.NodeDiskPressure,
					Status: v1.ConditionFalse,
				},
			},
		},
	}

	tests := map[string]struct {
		node   *v1.Node
		expect bool
	}{
		"node is in ready status": {
			node:   nodeReadyIns,
			expect: true,
		},
		"node is not in ready status": {
			node:   nodeNotReadyIns,
			expect: false,
		},
	}

	for k, v := range tests {
		actual := IsNodeReady(v.node)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}

}
