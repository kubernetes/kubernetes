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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

func TestNodeResourcesBalancedAllocation(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
	}
	machine1Spec := v1.PodSpec{
		NodeName: "machine1",
	}
	machine2Spec := v1.PodSpec{
		NodeName: "machine2",
	}
	noResources := v1.PodSpec{
		Containers: []v1.Container{},
	}
	cpuOnly := v1.PodSpec{
		NodeName: "machine1",
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1000m"),
						v1.ResourceMemory: resource.MustParse("0"),
					},
				},
			},
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("2000m"),
						v1.ResourceMemory: resource.MustParse("0"),
					},
				},
			},
		},
	}
	cpuOnly2 := cpuOnly
	cpuOnly2.NodeName = "machine2"
	cpuAndMemory := v1.PodSpec{
		NodeName: "machine2",
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1000m"),
						v1.ResourceMemory: resource.MustParse("2000"),
					},
				},
			},
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("2000m"),
						v1.ResourceMemory: resource.MustParse("3000"),
					},
				},
			},
		},
	}
	nonZeroContainer := v1.PodSpec{
		Containers: []v1.Container{{}},
	}
	nonZeroContainer1 := v1.PodSpec{
		NodeName:   "machine1",
		Containers: []v1.Container{{}},
	}
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			// Node1 scores (remaining resources) on 0-MaxNodeScore scale
			// CPU Fraction: 0 / 4000 = 0%
			// Memory Fraction: 0 / 10000 = 0%
			// Node1 Score: MaxNodeScore - (0-0)*MaxNodeScore = MaxNodeScore
			// Node2 scores (remaining resources) on 0-MaxNodeScore scale
			// CPU Fraction: 0 / 4000 = 0 %
			// Memory Fraction: 0 / 10000 = 0%
			// Node2 Score: MaxNodeScore - (0-0)*MaxNodeScore = MaxNodeScore
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "nothing scheduled, nothing requested",
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 3000 / 4000= 75%
			// Memory Fraction: 5000 / 10000 = 50%
			// Node1 Score: MaxNodeScore - (0.75-0.5)*MaxNodeScore = 75
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 3000 / 6000= 50%
			// Memory Fraction: 5000/10000 = 50%
			// Node2 Score: MaxNodeScore - (0.5-0.5)*MaxNodeScore = MaxNodeScore
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 75}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 0 / 4000= 0%
			// Memory Fraction: 0 / 10000 = 0%
			// Node1 Score: MaxNodeScore - (0-0)*MaxNodeScore = MaxNodeScore
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 0 / 4000= 0%
			// Memory Fraction: 0 / 10000 = 0%
			// Node2 Score: MaxNodeScore - (0-0)*MaxNodeScore= MaxNodeScore
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "no resources requested, pods scheduled",
			pods: []*v1.Pod{
				{Spec: machine1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: machine1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: machine2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: machine2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 300 / 250 = 100%
			// Memory Fraction: 600 / 10000 = 60%
			// Node1 Score: MaxNodeScore - (100-60)*MaxNodeScore = 60
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 100 / 250 = 40%
			// Memory Fraction: 200 / 10000 = 20%
			// Node2 Score: MaxNodeScore - (40-20)*MaxNodeScore= 80
			pod:          &v1.Pod{Spec: nonZeroContainer},
			nodes:        []*v1.Node{makeNode("machine1", 250, 1000*1024*1024), makeNode("machine2", 250, 1000*1024*1024)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 60}, {Name: "machine2", Score: 80}},
			name:         "no resources requested, pods scheduled",
			pods: []*v1.Pod{
				{Spec: nonZeroContainer1},
				{Spec: nonZeroContainer1},
			},
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 0 / 20000 = 0%
			// Node1 Score: MaxNodeScore - (0.6-0)*MaxNodeScore = 40
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 5000 / 20000 = 25%
			// Node2 Score: MaxNodeScore - (0.6-0.25)*MaxNodeScore = 65
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 40}, {Name: "machine2", Score: 65}},
			name:         "no resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: cpuOnly, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: cpuOnly2, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: cpuAndMemory, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 5000 / 20000 = 25%
			// Node1 Score: MaxNodeScore - (0.6-0.25)*MaxNodeScore = 65
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 10000 / 20000 = 50%
			// Node2 Score: MaxNodeScore - (0.6-0.5)*MaxNodeScore = 90
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 65}, {Name: "machine2", Score: 90}},
			name:         "resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 5000 / 20000 = 25%
			// Node1 Score: MaxNodeScore - (0.6-0.25)*MaxNodeScore = 65
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 10000 / 50000 = 20%
			// Node2 Score: MaxNodeScore - (0.6-0.2)*MaxNodeScore = 60
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 50000)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 65}, {Name: "machine2", Score: 60}},
			name:         "resources requested, pods scheduled with resources, differently sized machines",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 6000 = 1
			// Memory Fraction: 0 / 10000 = 0
			// Node1 Score: MaxNodeScore - (1 - 0) * MaxNodeScore = 0
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 6000 = 1
			// Memory Fraction 5000 / 10000 = 50%
			// Node2 Score: MaxNodeScore - (1 - 0.5) * MaxNodeScore = 50
			pod:          &v1.Pod{Spec: cpuOnly},
			nodes:        []*v1.Node{makeNode("machine1", 6000, 10000), makeNode("machine2", 6000, 10000)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 50}},
			name:         "requested resources at node capacity",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			snapshot := cache.NewSnapshot(test.pods, test.nodes)
			fh, _ := runtime.NewFramework(nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			p, _ := NewBalancedAllocation(nil, fh, feature.Features{EnablePodOverhead: true})

			for i := range test.nodes {
				hostResult, err := p.(framework.ScorePlugin).Score(context.Background(), nil, test.pod, test.nodes[i].Name)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if !reflect.DeepEqual(test.expectedList[i].Score, hostResult) {
					t.Errorf("got score %v for host %v, expected %v", hostResult, test.nodes[i].Name, test.expectedList[i].Score)
				}
			}
		})
	}
}
