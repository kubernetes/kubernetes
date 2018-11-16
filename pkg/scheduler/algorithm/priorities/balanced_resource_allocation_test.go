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

package priorities

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/kubernetes/pkg/features"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
)

// getExistingVolumeCountForNode gets the current number of volumes on node.
func getExistingVolumeCountForNode(pods []*v1.Pod, maxVolumes int) int {
	volumeCount := 0
	for _, pod := range pods {
		volumeCount += len(pod.Spec.Volumes)
	}
	if maxVolumes-volumeCount > 0 {
		return maxVolumes - volumeCount
	}
	return 0
}

func TestBalancedResourceAllocation(t *testing.T) {
	// Enable volumesOnNodeForBalancing to do balanced resource allocation
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BalanceAttachedNodeVolumes, true)()
	podwithVol1 := v1.PodSpec{
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
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "ovp"},
				},
			},
		},
		NodeName: "machine4",
	}
	podwithVol2 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("0m"),
						v1.ResourceMemory: resource.MustParse("0"),
					},
				},
			},
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("0m"),
						v1.ResourceMemory: resource.MustParse("0"),
					},
				},
			},
		},
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "ovp1"},
				},
			},
		},
		NodeName: "machine4",
	}
	podwithVol3 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("0m"),
						v1.ResourceMemory: resource.MustParse("0"),
					},
				},
			},
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("0m"),
						v1.ResourceMemory: resource.MustParse("0"),
					},
				},
			},
		},
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "ovp1"},
				},
			},
		},
		NodeName: "machine4",
	}
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
	cpuAndMemory3 := v1.PodSpec{
		NodeName: "machine3",
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
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList schedulerapi.HostPriorityList
		name         string
	}{
		{
			/*
				Node1 scores (remaining resources) on 0-10 scale
				CPU Fraction: 0 / 4000 = 0%
				Memory Fraction: 0 / 10000 = 0%
				Node1 Score: 10 - (0-0)*10 = 10

				Node2 scores (remaining resources) on 0-10 scale
				CPU Fraction: 0 / 4000 = 0 %
				Memory Fraction: 0 / 10000 = 0%
				Node2 Score: 10 - (0-0)*10 = 10
			*/
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: schedulerapi.MaxPriority}, {Host: "machine2", Score: schedulerapi.MaxPriority}},
			name:         "nothing scheduled, nothing requested",
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Fraction: 3000 / 4000= 75%
				Memory Fraction: 5000 / 10000 = 50%
				Node1 Score: 10 - (0.75-0.5)*10 = 7

				Node2 scores on 0-10 scale
				CPU Fraction: 3000 / 6000= 50%
				Memory Fraction: 5000/10000 = 50%
				Node2 Score: 10 - (0.5-0.5)*10 = 10
			*/
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 7}, {Host: "machine2", Score: schedulerapi.MaxPriority}},
			name:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Fraction: 0 / 4000= 0%
				Memory Fraction: 0 / 10000 = 0%
				Node1 Score: 10 - (0-0)*10 = 10

				Node2 scores on 0-10 scale
				CPU Fraction: 0 / 4000= 0%
				Memory Fraction: 0 / 10000 = 0%
				Node2 Score: 10 - (0-0)*10 = 10
			*/
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: schedulerapi.MaxPriority}, {Host: "machine2", Score: schedulerapi.MaxPriority}},
			name:         "no resources requested, pods scheduled",
			pods: []*v1.Pod{
				{Spec: machine1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: machine1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: machine2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: machine2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 0 / 20000 = 0%
				Node1 Score: 10 - (0.6-0)*10 = 4

				Node2 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 5000 / 20000 = 25%
				Node2 Score: 10 - (0.6-0.25)*10 = 6
			*/
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 4}, {Host: "machine2", Score: 6}},
			name:         "no resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: cpuOnly, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: cpuOnly2, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: cpuAndMemory, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 5000 / 20000 = 25%
				Node1 Score: 10 - (0.6-0.25)*10 = 6

				Node2 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 10000 / 20000 = 50%
				Node2 Score: 10 - (0.6-0.5)*10 = 9
			*/
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 6}, {Host: "machine2", Score: 9}},
			name:         "resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 5000 / 20000 = 25%
				Node1 Score: 10 - (0.6-0.25)*10 = 6

				Node2 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 10000 / 50000 = 20%
				Node2 Score: 10 - (0.6-0.2)*10 = 6
			*/
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 50000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 6}, {Host: "machine2", Score: 6}},
			name:         "resources requested, pods scheduled with resources, differently sized machines",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Fraction: 6000 / 4000 > 100% ==> Score := 0
				Memory Fraction: 0 / 10000 = 0
				Node1 Score: 0

				Node2 scores on 0-10 scale
				CPU Fraction: 6000 / 4000 > 100% ==> Score := 0
				Memory Fraction 5000 / 10000 = 50%
				Node2 Score: 0
			*/
			pod:          &v1.Pod{Spec: cpuOnly},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			name:         "requested resources exceed node capacity",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 0, 0), makeNode("machine2", 0, 0)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			name:         "zero node resources, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Machine4 will be chosen here because it already has a existing volume making the variance
				of volume count, CPU usage, memory usage closer.
			*/
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "ovp2"},
							},
						},
					},
				},
			},
			nodes:        []*v1.Node{makeNode("machine3", 3500, 40000), makeNode("machine4", 4000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine3", Score: 8}, {Host: "machine4", Score: 9}},
			name:         "Include volume count on a node for balanced resource allocation",
			pods: []*v1.Pod{
				{Spec: cpuAndMemory3},
				{Spec: podwithVol1},
				{Spec: podwithVol2},
				{Spec: podwithVol3},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)
			if len(test.pod.Spec.Volumes) > 0 {
				maxVolumes := 5
				for _, info := range nodeNameToInfo {
					info.TransientInfo.TransNodeInfo.AllocatableVolumesCount = getExistingVolumeCountForNode(info.Pods(), maxVolumes)
					info.TransientInfo.TransNodeInfo.RequestedVolumes = len(test.pod.Spec.Volumes)
				}
			}
			list, err := priorityFunction(BalancedResourceAllocationMap, nil, nil)(test.pod, nodeNameToInfo, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedList, list) {
				t.Errorf("expected %#v, got %#v", test.expectedList, list)
			}
		})
	}
}
