/*
Copyright 2016 The Kubernetes Authors.

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

package helpers

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
)

func TestTaintExists(t *testing.T) {
	testingTaints := []v1.Taint{
		{
			Key:    "foo_1",
			Value:  "bar_1",
			Effect: v1.TaintEffectNoExecute,
		},
		{
			Key:    "foo_2",
			Value:  "bar_2",
			Effect: v1.TaintEffectNoSchedule,
		},
	}

	cases := []struct {
		name           string
		taintToFind    *v1.Taint
		expectedResult bool
	}{
		{
			name:           "taint exists",
			taintToFind:    &v1.Taint{Key: "foo_1", Value: "bar_1", Effect: v1.TaintEffectNoExecute},
			expectedResult: true,
		},
		{
			name:           "different key",
			taintToFind:    &v1.Taint{Key: "no_such_key", Value: "bar_1", Effect: v1.TaintEffectNoExecute},
			expectedResult: false,
		},
		{
			name:           "different effect",
			taintToFind:    &v1.Taint{Key: "foo_1", Value: "bar_1", Effect: v1.TaintEffectNoSchedule},
			expectedResult: false,
		},
	}

	for _, c := range cases {
		result := taintExists(testingTaints, c.taintToFind)

		if result != c.expectedResult {
			t.Errorf("[%s] unexpected results: %v", c.name, result)
			continue
		}
	}
}

func TestRemoveTaint(t *testing.T) {
	cases := []struct {
		name           string
		node           *v1.Node
		taintToRemove  *v1.Taint
		expectedTaints []v1.Taint
		expectedResult bool
	}{
		{
			name: "remove taint unsuccessfully",
			node: &v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "foo",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			taintToRemove: &v1.Taint{
				Key:    "foo_1",
				Effect: v1.TaintEffectNoSchedule,
			},
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: false,
		},
		{
			name: "remove taint successfully",
			node: &v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "foo",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			taintToRemove: &v1.Taint{
				Key:    "foo",
				Effect: v1.TaintEffectNoSchedule,
			},
			expectedTaints: []v1.Taint{},
			expectedResult: true,
		},
		{
			name: "remove taint from node with no taint",
			node: &v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{},
				},
			},
			taintToRemove: &v1.Taint{
				Key:    "foo",
				Effect: v1.TaintEffectNoSchedule,
			},
			expectedTaints: []v1.Taint{},
			expectedResult: false,
		},
	}

	for _, c := range cases {
		newNode, result, err := removeTaint(c.node, c.taintToRemove)
		if err != nil {
			t.Errorf("[%s] should not raise error but got: %v", c.name, err)
		}
		if result != c.expectedResult {
			t.Errorf("[%s] should return %t, but got: %t", c.name, c.expectedResult, result)
		}
		if !reflect.DeepEqual(newNode.Spec.Taints, c.expectedTaints) {
			t.Errorf("[%s] the new node object should have taints %v, but got: %v", c.name, c.expectedTaints, newNode.Spec.Taints)
		}
	}
}

func TestDeleteTaint(t *testing.T) {
	cases := []struct {
		name           string
		taints         []v1.Taint
		taintToDelete  *v1.Taint
		expectedTaints []v1.Taint
		expectedResult bool
	}{
		{
			name: "delete taint with different name",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintToDelete: &v1.Taint{Key: "foo_1", Effect: v1.TaintEffectNoSchedule},
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: false,
		},
		{
			name: "delete taint with different effect",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintToDelete: &v1.Taint{Key: "foo", Effect: v1.TaintEffectNoExecute},
			expectedTaints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			expectedResult: false,
		},
		{
			name: "delete taint successfully",
			taints: []v1.Taint{
				{
					Key:    "foo",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			taintToDelete:  &v1.Taint{Key: "foo", Effect: v1.TaintEffectNoSchedule},
			expectedTaints: []v1.Taint{},
			expectedResult: true,
		},
		{
			name:           "delete taint from empty taint array",
			taints:         []v1.Taint{},
			taintToDelete:  &v1.Taint{Key: "foo", Effect: v1.TaintEffectNoSchedule},
			expectedTaints: []v1.Taint{},
			expectedResult: false,
		},
	}

	for _, c := range cases {
		taints, result := deleteTaint(c.taints, c.taintToDelete)
		if result != c.expectedResult {
			t.Errorf("[%s] should return %t, but got: %t", c.name, c.expectedResult, result)
		}
		if !reflect.DeepEqual(taints, c.expectedTaints) {
			t.Errorf("[%s] the result taints should be %v, but got: %v", c.name, c.expectedTaints, taints)
		}
	}
}
