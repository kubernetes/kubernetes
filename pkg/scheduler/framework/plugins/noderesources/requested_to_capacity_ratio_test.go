/*
Copyright 2019 The Kubernetes Authors.

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

package noderesources

import (
	"context"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

func TestRequestedToCapacityRatio(t *testing.T) {
	type test struct {
		name               string
		requestedPod       *v1.Pod
		nodes              []*v1.Node
		scheduledPods      []*v1.Pod
		expectedPriorities framework.NodeScoreList
	}

	tests := []test{
		{
			name:               "nothing scheduled, nothing requested (default - least requested nodes have priority)",
			requestedPod:       makePod("", 0, 0),
			nodes:              []*v1.Node{makeNode("node1", 4000, 10000), makeNode("node2", 4000, 10000)},
			scheduledPods:      []*v1.Pod{makePod("node1", 0, 0), makePod("node2", 0, 0)},
			expectedPriorities: []framework.NodeScore{{Name: "node1", Score: 100}, {Name: "node2", Score: 100}},
		},
		{
			name:               "nothing scheduled, resources requested, differently sized machines (default - least requested nodes have priority)",
			requestedPod:       makePod("", 3000, 5000),
			nodes:              []*v1.Node{makeNode("node1", 4000, 10000), makeNode("node2", 6000, 10000)},
			scheduledPods:      []*v1.Pod{makePod("node1", 0, 0), makePod("node2", 0, 0)},
			expectedPriorities: []framework.NodeScore{{Name: "node1", Score: 38}, {Name: "node2", Score: 50}},
		},
		{
			name:               "no resources requested, pods scheduled with resources (default - least requested nodes have priority)",
			requestedPod:       makePod("", 0, 0),
			nodes:              []*v1.Node{makeNode("node1", 4000, 10000), makeNode("node2", 6000, 10000)},
			scheduledPods:      []*v1.Pod{makePod("node1", 3000, 5000), makePod("node2", 3000, 5000)},
			expectedPriorities: []framework.NodeScore{{Name: "node1", Score: 38}, {Name: "node2", Score: 50}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := framework.NewCycleState()
			snapshot := cache.NewSnapshot(test.scheduledPods, test.nodes)
			fh, _ := runtime.NewFramework(nil, nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			args := config.RequestedToCapacityRatioArgs{
				Shape: []config.UtilizationShapePoint{
					{Utilization: 0, Score: 10},
					{Utilization: 100, Score: 0},
				},
				Resources: []config.ResourceSpec{
					{Name: "memory", Weight: 1},
					{Name: "cpu", Weight: 1},
				},
			}
			p, err := NewRequestedToCapacityRatio(&args, fh)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			var gotPriorities framework.NodeScoreList
			for _, n := range test.nodes {
				score, status := p.(framework.ScorePlugin).Score(context.Background(), state, test.requestedPod, n.Name)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotPriorities = append(gotPriorities, framework.NodeScore{Name: n.Name, Score: score})
			}

			if !reflect.DeepEqual(test.expectedPriorities, gotPriorities) {
				t.Errorf("expected:\n\t%+v,\ngot:\n\t%+v", test.expectedPriorities, gotPriorities)
			}
		})
	}
}

func makePod(node string, milliCPU, memory int64) *v1.Pod {
	return &v1.Pod{
		Spec: v1.PodSpec{
			NodeName: node,
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(memory, resource.DecimalSI),
						},
					},
				},
			},
		},
	}
}

func TestBrokenLinearFunction(t *testing.T) {
	type Assertion struct {
		p        int64
		expected int64
	}
	type Test struct {
		points     []functionShapePoint
		assertions []Assertion
	}

	tests := []Test{
		{
			points: []functionShapePoint{{10, 1}, {90, 9}},
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
			points: []functionShapePoint{{0, 2}, {40, 10}, {100, 0}},
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
			points: []functionShapePoint{{0, 2}, {40, 2}, {100, 2}},
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
		function := buildBrokenLinearFunction(test.points)
		for _, assertion := range test.assertions {
			assert.InDelta(t, assertion.expected, function(assertion.p), 0.1, "points=%v, p=%f", test.points, assertion.p)
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
		expectedList framework.NodeScoreList
		name         string
	}{
		{

			//  Node1 scores (used resources) on 0-MaxNodeScore scale
			//  Node1 Score:
			//  rawScoringFunction(used + requested / available)
			//  resourceScoringFunction((0+0),8)
			//      = maxUtilization - (8-0)*(maxUtilization/8) = 0 = rawScoringFunction(0)
			//  Node1 Score: 0
			//  Node2 scores (used resources) on 0-MaxNodeScore scale
			//  rawScoringFunction(used + requested / available)
			//  resourceScoringFunction((0+0),4)
			//      = maxUtilization - (4-0)*(maxUtilization/4) = 0 = rawScoringFunction(0)
			//  Node2 Score: 0

			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResource2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResource1)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "nothing scheduled, nothing requested",
		},

		{

			// Node1 scores (used resources) on 0-MaxNodeScore scale
			// Node1 Score:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),8)
			//  = maxUtilization - (8-2)*(maxUtilization/8) = 25 = rawScoringFunction(25)
			// Node1 Score: 2
			// Node2 scores (used resources) on 0-MaxNodeScore scale
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),4)
			//  = maxUtilization - (4-2)*(maxUtilization/4) = 50 = rawScoringFunction(50)
			// Node2 Score: 5

			pod:          &v1.Pod{Spec: extendedResourcePod1},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResource2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResource1)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 2}, {Name: "machine2", Score: 5}},
			name:         "resources requested, pods scheduled with less resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},

		{

			// Node1 scores (used resources) on 0-MaxNodeScore scale
			// Node1 Score:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),8)
			//  = maxUtilization - (8-2)*(maxUtilization/8) = 25 = rawScoringFunction(25)
			// Node1 Score: 2
			// Node2 scores (used resources) on 0-MaxNodeScore scale
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((2+2),4)
			//  = maxUtilization - (4-4)*(maxUtilization/4) = maxUtilization = rawScoringFunction(maxUtilization)
			// Node2 Score: 10

			pod:          &v1.Pod{Spec: extendedResourcePod1},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResource2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResource1)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 2}, {Name: "machine2", Score: 10}},
			name:         "resources requested, pods scheduled with resources, on node with existing pod running ",
			pods: []*v1.Pod{
				{Spec: machine2Pod},
			},
		},

		{

			// Node1 scores (used resources) on 0-MaxNodeScore scale
			// Node1 Score:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+4),8)
			//  = maxUtilization - (8-4)*(maxUtilization/8) = 50 = rawScoringFunction(50)
			// Node1 Score: 5
			// Node2 scores (used resources) on 0-MaxNodeScore scale
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+4),4)
			//  = maxUtilization - (4-4)*(maxUtilization/4) = maxUtilization = rawScoringFunction(maxUtilization)
			// Node2 Score: 10

			pod:          &v1.Pod{Spec: extendedResourcePod2},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResource2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResource1)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 5}, {Name: "machine2", Score: 10}},
			name:         "resources requested, pods scheduled with more resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := framework.NewCycleState()
			snapshot := cache.NewSnapshot(test.pods, test.nodes)
			fh, _ := runtime.NewFramework(nil, nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			args := config.RequestedToCapacityRatioArgs{
				Shape: []config.UtilizationShapePoint{
					{Utilization: 0, Score: 0},
					{Utilization: 100, Score: 1},
				},
				Resources: []config.ResourceSpec{
					{Name: "intel.com/foo", Weight: 1},
				},
			}
			p, err := NewRequestedToCapacityRatio(&args, fh)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			var gotList framework.NodeScoreList
			for _, n := range test.nodes {
				score, status := p.(framework.ScorePlugin).Score(context.Background(), state, test.pod, n.Name)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: n.Name, Score: score})
			}

			if !reflect.DeepEqual(test.expectedList, gotList) {
				t.Errorf("expected %#v, got %#v", test.expectedList, gotList)
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
		expectedList framework.NodeScoreList
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
			//  = 100 - (8-0)*(100/8) = 0 = rawScoringFunction(0)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+0),4)
			//  = 100 - (4-0)*(100/4) = 0 = rawScoringFunction(0)
			// Node1 Score: (0 * 3) + (0 * 5) / 8 = 0

			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+0),4)
			//  = 100 - (4-0)*(100/4) = 0 = rawScoringFunction(0)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+0),8)
			//  = 100 - (8-0)*(100/8) = 0 = rawScoringFunction(0)
			// Node2 Score: (0 * 3) + (0 * 5) / 8 = 0

			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResources2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResources1)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
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
			//  = 100 - (8-2)*(100/8) = 25 = rawScoringFunction(25)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),4)
			//  = 100 - (4-2)*(100/4) = 50 = rawScoringFunction(50)
			// Node1 Score: (2 * 3) + (5 * 5) / 8 = 4

			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),4)
			//  = 100 - (4-2)*(100/4) = 50 = rawScoringFunction(50)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),8)
			//  = 100 - (8-2)*(100/8) = 25 = rawScoringFunction(25)
			// Node2 Score: (5 * 3) + (2 * 5) / 8 = 3

			pod:          &v1.Pod{Spec: extnededResourcePod1},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResources2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResources1)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 4}, {Name: "machine2", Score: 3}},
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
			//  = 100 - (8-2)*(100/8) = 25 = rawScoringFunction(25)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),4)
			//  = 100 - (4-2)*(100/4) = 50 = rawScoringFunction(50)
			// Node1 Score: (2 * 3) + (5 * 5) / 8 = 4
			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((2+2),4)
			//  = 100 - (4-4)*(100/4) = 100 = rawScoringFunction(100)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((2+2),8)
			//  = 100 - (8-4)*(100/8) = 50 = rawScoringFunction(50)
			// Node2 Score: (10 * 3) + (5 * 5) / 8 = 7

			pod:          &v1.Pod{Spec: extnededResourcePod1},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResources2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResources1)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 4}, {Name: "machine2", Score: 7}},
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
			//  = 100 - (8-4)*(100/8) = 50 = rawScoringFunction(50)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),4)
			//  = 100 - (4-2)*(100/4) = 50 = rawScoringFunction(50)
			// Node1 Score: (5 * 3) + (5 * 5) / 8 = 5
			// Node2 scores (used resources) on 0-10 scale
			// rawScoringFunction(used + requested / available)
			// intel.com/foo:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+4),4)
			//  = 100 - (4-4)*(100/4) = 100 = rawScoringFunction(100)
			// intel.com/bar:
			// rawScoringFunction(used + requested / available)
			// resourceScoringFunction((0+2),8)
			//  = 100 - (8-2)*(100/8) = 25 = rawScoringFunction(25)
			// Node2 Score: (10 * 3) + (2 * 5) / 8 = 5

			pod:          &v1.Pod{Spec: extnededResourcePod2},
			nodes:        []*v1.Node{makeNodeWithExtendedResource("machine1", 4000, 10000*1024*1024, extendedResources2), makeNodeWithExtendedResource("machine2", 4000, 10000*1024*1024, extendedResources1)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 5}, {Name: "machine2", Score: 5}},
			name:         "resources requested, pods scheduled with more resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := framework.NewCycleState()
			snapshot := cache.NewSnapshot(test.pods, test.nodes)
			fh, _ := runtime.NewFramework(nil, nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			args := config.RequestedToCapacityRatioArgs{
				Shape: []config.UtilizationShapePoint{
					{Utilization: 0, Score: 0},
					{Utilization: 100, Score: 1},
				},
				Resources: []config.ResourceSpec{
					{Name: "intel.com/foo", Weight: 3},
					{Name: "intel.com/bar", Weight: 5},
				},
			}
			p, err := NewRequestedToCapacityRatio(&args, fh)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			var gotList framework.NodeScoreList
			for _, n := range test.nodes {
				score, status := p.(framework.ScorePlugin).Score(context.Background(), state, test.pod, n.Name)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: n.Name, Score: score})
			}

			if !reflect.DeepEqual(test.expectedList, gotList) {
				t.Errorf("expected %#v, got %#v", test.expectedList, gotList)
			}
		})
	}
}
