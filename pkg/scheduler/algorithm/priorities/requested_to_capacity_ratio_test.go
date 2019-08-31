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

package priorities

import (
	"reflect"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

func TestCreatingFunctionShapeErrorsIfEmptyPoints(t *testing.T) {
	var err error
	_, err = NewFunctionShape([]FunctionShapePoint{})
	assert.Equal(t, "at least one point must be specified", err.Error())
}

func TestCreatingResourceNegativeWeight(t *testing.T) {
	err := validateResourceWeightMap(ResourceToWeightMap{v1.ResourceCPU: -1})
	assert.Equal(t, "resource cpu weight -1 must not be less than 1", err.Error())
}

func TestCreatingResourceDefaultWeight(t *testing.T) {
	err := validateResourceWeightMap(ResourceToWeightMap{})
	assert.Equal(t, "resourceToWeightMap cannot be nil", err.Error())

}

func TestCreatingFunctionShapeErrorsIfXIsNotSorted(t *testing.T) {
	var err error
	_, err = NewFunctionShape([]FunctionShapePoint{{10, 1}, {15, 2}, {20, 3}, {19, 4}, {25, 5}})
	assert.Equal(t, "utilization values must be sorted. Utilization[2]==20 >= Utilization[3]==19", err.Error())

	_, err = NewFunctionShape([]FunctionShapePoint{{10, 1}, {20, 2}, {20, 3}, {22, 4}, {25, 5}})
	assert.Equal(t, "utilization values must be sorted. Utilization[1]==20 >= Utilization[2]==20", err.Error())
}

func TestCreatingFunctionPointNotInAllowedRange(t *testing.T) {
	var err error
	_, err = NewFunctionShape([]FunctionShapePoint{{-1, 0}, {100, 10}})
	assert.Equal(t, "utilization values must not be less than 0. Utilization[0]==-1", err.Error())

	_, err = NewFunctionShape([]FunctionShapePoint{{0, 0}, {101, 10}})
	assert.Equal(t, "utilization values must not be greater than 100. Utilization[1]==101", err.Error())

	_, err = NewFunctionShape([]FunctionShapePoint{{0, -1}, {100, 10}})
	assert.Equal(t, "score values must not be less than 0. Score[0]==-1", err.Error())

	_, err = NewFunctionShape([]FunctionShapePoint{{0, 0}, {100, 11}})
	assert.Equal(t, "score valuses not be greater than 10. Score[1]==11", err.Error())
}

func TestBrokenLinearFunction(t *testing.T) {
	type Assertion struct {
		p        int64
		expected int64
	}
	type Test struct {
		points     []FunctionShapePoint
		assertions []Assertion
	}

	tests := []Test{
		{
			points: []FunctionShapePoint{{10, 1}, {90, 9}},
			assertions: []Assertion{
				{p: -10, expected: 1},
				{p: 0, expected: 1},
				{p: 9, expected: 1},
				{p: 10, expected: 1},
				{p: 15, expected: 1},
				{p: 19, expected: 1},
				{p: 20, expected: 2},
				{p: 89, expected: 8},
				{p: 90, expected: 9},
				{p: 99, expected: 9},
				{p: 100, expected: 9},
				{p: 110, expected: 9},
			},
		},
		{
			points: []FunctionShapePoint{{0, 2}, {40, 10}, {100, 0}},
			assertions: []Assertion{
				{p: -10, expected: 2},
				{p: 0, expected: 2},
				{p: 20, expected: 6},
				{p: 30, expected: 8},
				{p: 40, expected: 10},
				{p: 70, expected: 5},
				{p: 100, expected: 0},
				{p: 110, expected: 0},
			},
		},
		{
			points: []FunctionShapePoint{{0, 2}, {40, 2}, {100, 2}},
			assertions: []Assertion{
				{p: -10, expected: 2},
				{p: 0, expected: 2},
				{p: 20, expected: 2},
				{p: 30, expected: 2},
				{p: 40, expected: 2},
				{p: 70, expected: 2},
				{p: 100, expected: 2},
				{p: 110, expected: 2},
			},
		},
	}

	for _, test := range tests {
		functionShape, err := NewFunctionShape(test.points)
		assert.Nil(t, err)
		function := buildBrokenLinearFunction(functionShape)
		for _, assertion := range test.assertions {
			assert.InDelta(t, assertion.expected, function(assertion.p), 0.1, "points=%v, p=%f", test.points, assertion.p)
		}
	}
}

func TestRequestedToCapacityRatio(t *testing.T) {
	type resources struct {
		cpu int64
		mem int64
	}

	type nodeResources struct {
		capacity resources
		used     resources
	}

	type test struct {
		test               string
		requested          resources
		nodes              map[string]nodeResources
		expectedPriorities schedulerapi.HostPriorityList
	}

	tests := []test{
		{
			test:      "nothing scheduled, nothing requested (default - least requested nodes have priority)",
			requested: resources{0, 0},
			nodes: map[string]nodeResources{
				"node1": {
					capacity: resources{4000, 10000},
					used:     resources{0, 0},
				},
				"node2": {
					capacity: resources{4000, 10000},
					used:     resources{0, 0},
				},
			},
			expectedPriorities: []schedulerapi.HostPriority{{Host: "node1", Score: 10}, {Host: "node2", Score: 10}},
		},
		{
			test:      "nothing scheduled, resources requested, differently sized machines (default - least requested nodes have priority)",
			requested: resources{3000, 5000},
			nodes: map[string]nodeResources{
				"node1": {
					capacity: resources{4000, 10000},
					used:     resources{0, 0},
				},
				"node2": {
					capacity: resources{6000, 10000},
					used:     resources{0, 0},
				},
			},
			expectedPriorities: []schedulerapi.HostPriority{{Host: "node1", Score: 4}, {Host: "node2", Score: 5}},
		},
		{
			test:      "no resources requested, pods scheduled with resources (default - least requested nodes have priority)",
			requested: resources{0, 0},
			nodes: map[string]nodeResources{
				"node1": {
					capacity: resources{4000, 10000},
					used:     resources{3000, 5000},
				},
				"node2": {
					capacity: resources{6000, 10000},
					used:     resources{3000, 5000},
				},
			},
			expectedPriorities: []schedulerapi.HostPriority{{Host: "node1", Score: 4}, {Host: "node2", Score: 5}},
		},
	}

	buildResourcesPod := func(node string, requestedResources resources) *v1.Pod {
		return &v1.Pod{Spec: v1.PodSpec{
			NodeName: node,
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(requestedResources.cpu, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(requestedResources.mem, resource.DecimalSI),
						},
					},
				},
			},
		},
		}
	}

	for _, test := range tests {

		var nodeNames []string
		for nodeName := range test.nodes {
			nodeNames = append(nodeNames, nodeName)
		}
		sort.Strings(nodeNames)

		var nodes []*v1.Node
		for _, nodeName := range nodeNames {
			node := test.nodes[nodeName]
			nodes = append(nodes, makeNode(nodeName, node.capacity.cpu, node.capacity.mem))
		}

		var scheduledPods []*v1.Pod
		for name, node := range test.nodes {
			scheduledPods = append(scheduledPods,
				buildResourcesPod(name, node.used))
		}

		newPod := buildResourcesPod("", test.requested)

		nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(scheduledPods, nodes)
		list, err := priorityFunction(RequestedToCapacityRatioResourceAllocationPriorityDefault().PriorityMap, nil, nil)(newPod, nodeNameToInfo, nodes)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedPriorities, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedPriorities, list)
		}
	}
}
func TestResourceBinPackingSingleExtended(t *testing.T) {
	extendedResource := "intel.com/foo"
	extendedResource1 := map[string]int64{
		"intel.com/foo": 4,
	}

	extendedResource2 := map[string]int64{
		"intel.com/foo": 8,
	}

	noResources := v1.PodSpec{
		Containers: []v1.Container{},
	}
	extendedResourcePod1 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(extendedResource): resource.MustParse("2"),
					},
				},
			},
		},
	}
	extendedResourcePod2 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(extendedResource): resource.MustParse("4"),
					},
				},
			},
		},
	}
	machine2Pod := extendedResourcePod1
	machine2Pod.NodeName = "machine2"
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList schedulerapi.HostPriorityList
		name         string
	}{
		{

			//	Node1 scores (used resources) on 0-10 scale
			//	Node1 Score:
			//	rawScoringFunction(used + requested / available)
			//	resourceScoringFunction((0+0),8)
			//		= 100 - (8-0)*(100/8) = 0 = rawScoringFunction(0)
			//	Node1 Score: 0
			//	Node2 scores (used resources) on 0-10 scale
			//	rawScoringFunction(used + requested / available)
			//	resourceScoringFunction((0+0),4)
			//		= 100 - (4-0)*(100/4) = 0 = rawScoringFunction(0)
			//	Node2 Score: 0

			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResource2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResource1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			name:         "nothing scheduled, nothing requested",
		},

		{

			// Node1 scores (used resources) on 0-10 scale
			// Node1 Score:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),8)
			// 	= 100 - (8-2)*(100/8) = 25 = rawScoringFunction(25)
			// Node1 Score: 2
			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),4)
			// 	= 100 - (4-2)*(100/4) = 50 = rawScoringFunction(50)
			// Node2 Score: 5

			pod:          &v1.Pod{Spec: extendedResourcePod1},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResource2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResource1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 2}, {Host: "machine2", Score: 5}},
			name:         "resources requested, pods scheduled with less resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},

		{

			// Node1 scores (used resources) on 0-10 scale
			// Node1 Score:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),8)
			// 	= 100 - (8-2)*(100/8) = 25 =rawScoringFunction(25)
			// Node1 Score: 2
			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((2+2),4)
			// 	= 100 - (4-4)*(100/4) = 100 = rawScoringFunction(100)
			// Node2 Score: 10

			pod:          &v1.Pod{Spec: extendedResourcePod1},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResource2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResource1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 2}, {Host: "machine2", Score: 10}},
			name:         "resources requested, pods scheduled with resources, on node with existing pod running ",
			pods: []*v1.Pod{
				{Spec: machine2Pod},
			},
		},

		{

			// Node1 scores (used resources) on 0-10 scale
			// Node1 Score:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+4),8)
			// 	= 100 - (8-4)*(100/8) = 50 = rawScoringFunction(50)
			// Node1 Score: 5
			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+4),4)
			// 	= 100 - (4-4)*(100/4) = 100 = rawScoringFunction(100)
			// Node2 Score: 10

			pod:          &v1.Pod{Spec: extendedResourcePod2},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResource2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResource1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 5}, {Host: "machine2", Score: 10}},
			name:         "resources requested, pods scheduled with more resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(test.pods, test.nodes)
			functionShape, _ := NewFunctionShape([]FunctionShapePoint{{0, 0}, {100, 10}})
			resourceToWeightMap := ResourceToWeightMap{v1.ResourceName("intel.com/foo"): 1}
			prior := RequestedToCapacityRatioResourceAllocationPriority(functionShape, resourceToWeightMap)
			list, err := priorityFunction(prior.PriorityMap, nil, nil)(test.pod, nodeNameToInfo, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedList, list) {
				t.Errorf("expected %#v, got %#v", test.expectedList, list)
			}
		})
	}
}

func TestResourceBinPackingMultipleExtended(t *testing.T) {
	extendedResource1 := "intel.com/foo"
	extendedResource2 := "intel.com/bar"
	extendedResources1 := map[string]int64{
		"intel.com/foo": 4,
		"intel.com/bar": 8,
	}

	extendedResources2 := map[string]int64{
		"intel.com/foo": 8,
		"intel.com/bar": 4,
	}

	noResources := v1.PodSpec{
		Containers: []v1.Container{},
	}
	extnededResourcePod1 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(extendedResource1): resource.MustParse("2"),
						v1.ResourceName(extendedResource2): resource.MustParse("2"),
					},
				},
			},
		},
	}
	extnededResourcePod2 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(extendedResource1): resource.MustParse("4"),
						v1.ResourceName(extendedResource2): resource.MustParse("2"),
					},
				},
			},
		},
	}
	machine2Pod := extnededResourcePod1
	machine2Pod.NodeName = "machine2"
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList schedulerapi.HostPriorityList
		name         string
	}{
		{

			// resources["intel.com/foo"] = 3
			// resources["intel.com/bar"] = 5
			// Node1 scores (used resources) on 0-10 scale
			// Node1 Score:
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+0),8)
			// 	= 100 - (8-0)*(100/8) = 0 = rawScoringFunction(0)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+0),4)
			// 	= 100 - (4-0)*(100/4) = 0 = rawScoringFunction(0)
			// Node1 Score: (0 * 3) + (0 * 5) / 8 = 0

			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+0),4)
			// 	= 100 - (4-0)*(100/4) = 0 = rawScoringFunction(0)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+0),8)
			// 	= 100 - (8-0)*(100/8) = 0 = rawScoringFunction(0)
			// Node2 Score: (0 * 3) + (0 * 5) / 8 = 0

			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResources2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			name:         "nothing scheduled, nothing requested",
		},

		{

			// resources["intel.com/foo"] = 3
			// resources["intel.com/bar"] = 5
			// Node1 scores (used resources) on 0-10 scale
			// Node1 Score:
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),8)
			// 	= 100 - (8-2)*(100/8) = 25 = rawScoringFunction(25)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),4)
			// 	= 100 - (4-2)*(100/4) = 50 = rawScoringFunction(50)
			// Node1 Score: (2 * 3) + (5 * 5) / 8 = 4

			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),4)
			// 	= 100 - (4-2)*(100/4) = 50 = rawScoringFunction(50)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),8)
			// 	= 100 - (8-2)*(100/8) = 25 = rawScoringFunction(25)
			// Node2 Score: (5 * 3) + (2 * 5) / 8 = 3

			pod:          &v1.Pod{Spec: extnededResourcePod1},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResources2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 4}, {Host: "machine2", Score: 3}},
			name:         "resources requested, pods scheduled with less resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},

		{

			// resources["intel.com/foo"] = 3
			// resources["intel.com/bar"] = 5
			// Node1 scores (used resources) on 0-10 scale
			// Node1 Score:
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),8)
			// 	= 100 - (8-2)*(100/8) = 25 = rawScoringFunction(25)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),4)
			// 	= 100 - (4-2)*(100/4) = 50 = rawScoringFunction(50)
			// Node1 Score: (2 * 3) + (5 * 5) / 8 = 4
			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((2+2),4)
			// 	= 100 - (4-4)*(100/4) = 100 = rawScoringFunction(100)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((2+2),8)
			// 	= 100 - (8-4)*(100/8) = 50 = rawScoringFunction(50)
			// Node2 Score: (10 * 3) + (5 * 5) / 8 = 7

			pod:          &v1.Pod{Spec: extnededResourcePod1},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResources2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 4}, {Host: "machine2", Score: 7}},
			name:         "resources requested, pods scheduled with resources, on node with existing pod running ",
			pods: []*v1.Pod{
				{Spec: machine2Pod},
			},
		},

		{

			// resources["intel.com/foo"] = 3
			// resources["intel.com/bar"] = 5
			// Node1 scores (used resources) on 0-10 scale
			// used + requested / available
			// intel.com/foo Score: { (0 + 4) / 8 } * 10 = 0
			// intel.com/bar Score: { (0 + 2) / 4 } * 10 = 0
			// Node1 Score: (0.25 * 3) + (0.5 * 5) / 8 = 5
			// resources["intel.com/foo"] = 3
			// resources["intel.com/bar"] = 5
			// Node2 scores (used resources) on 0-10 scale
			// used + requested / available
			// intel.com/foo Score: { (0 + 4) / 4 } * 10 = 0
			// intel.com/bar Score: { (0 + 2) / 8 } * 10 = 0
			// Node2 Score: (1 * 3) + (0.25 * 5) / 8 = 5

			// resources["intel.com/foo"] = 3
			// resources["intel.com/bar"] = 5
			// Node1 scores (used resources) on 0-10 scale
			// Node1 Score:
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+4),8)
			// 	= 100 - (8-4)*(100/8) = 50 = rawScoringFunction(50)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),4)
			// 	= 100 - (4-2)*(100/4) = 50 = rawScoringFunction(50)
			// Node1 Score: (5 * 3) + (5 * 5) / 8 = 5
			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+4),4)
			// 	= 100 - (4-4)*(100/4) = 100 = rawScoringFunction(100)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),8)
			// 	= 100 - (8-2)*(100/8) = 25 = rawScoringFunction(25)
			// Node2 Score: (10 * 3) + (2 * 5) / 8 = 5

			pod:          &v1.Pod{Spec: extnededResourcePod2},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResources2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 5}, {Host: "machine2", Score: 5}},
			name:         "resources requested, pods scheduled with more resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(test.pods, test.nodes)
			functionShape, _ := NewFunctionShape([]FunctionShapePoint{{0, 0}, {100, 10}})
			resourceToWeightMap := ResourceToWeightMap{v1.ResourceName("intel.com/foo"): 3, v1.ResourceName("intel.com/bar"): 5}
			prior := RequestedToCapacityRatioResourceAllocationPriority(functionShape, resourceToWeightMap)
			list, err := priorityFunction(prior.PriorityMap, nil, nil)(test.pod, nodeNameToInfo, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedList, list) {
				t.Errorf("expected %#v, got %#v", test.expectedList, list)
			}
		})
	}
}
