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

package framework

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/ktesting/initoption"
)

func TestNewResource(t *testing.T) {
	tests := []struct {
		name         string
		resourceList v1.ResourceList
		expected     *Resource
	}{
		{
			name:         "empty resource",
			resourceList: map[v1.ResourceName]resource.Quantity{},
			expected:     &Resource{},
		},
		{
			name: "complex resource",
			resourceList: map[v1.ResourceName]resource.Quantity{
				v1.ResourceCPU:                      *resource.NewScaledQuantity(4, -3),
				v1.ResourceMemory:                   *resource.NewQuantity(2000, resource.BinarySI),
				v1.ResourcePods:                     *resource.NewQuantity(80, resource.BinarySI),
				v1.ResourceEphemeralStorage:         *resource.NewQuantity(5000, resource.BinarySI),
				"scalar.test/" + "scalar1":          *resource.NewQuantity(1, resource.DecimalSI),
				v1.ResourceHugePagesPrefix + "test": *resource.NewQuantity(2, resource.BinarySI),
			},
			expected: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
				AllowedPodNumber: 80,
				ScalarResources:  map[v1.ResourceName]int64{"scalar.test/scalar1": 1, "hugepages-test": 2},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			r := NewResource(test.resourceList)
			if !reflect.DeepEqual(test.expected, r) {
				t.Errorf("expected: %#v, got: %#v", test.expected, r)
			}
		})
	}
}

func TestResourceClone(t *testing.T) {
	tests := []struct {
		resource *Resource
		expected *Resource
	}{
		{
			resource: &Resource{},
			expected: &Resource{},
		},
		{
			resource: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
				AllowedPodNumber: 80,
				ScalarResources:  map[v1.ResourceName]int64{"scalar.test/scalar1": 1, "hugepages-test": 2},
			},
			expected: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
				AllowedPodNumber: 80,
				ScalarResources:  map[v1.ResourceName]int64{"scalar.test/scalar1": 1, "hugepages-test": 2},
			},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			r := test.resource.Clone()
			// Modify the field to check if the result is a clone of the origin one.
			test.resource.MilliCPU += 1000
			if !reflect.DeepEqual(test.expected, r) {
				t.Errorf("expected: %#v, got: %#v", test.expected, r)
			}
		})
	}
}

func TestResourceAddScalar(t *testing.T) {
	tests := []struct {
		resource       *Resource
		scalarName     v1.ResourceName
		scalarQuantity int64
		expected       *Resource
	}{
		{
			resource:       &Resource{},
			scalarName:     "scalar1",
			scalarQuantity: 100,
			expected: &Resource{
				ScalarResources: map[v1.ResourceName]int64{"scalar1": 100},
			},
		},
		{
			resource: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
				AllowedPodNumber: 80,
				ScalarResources:  map[v1.ResourceName]int64{"hugepages-test": 2},
			},
			scalarName:     "scalar2",
			scalarQuantity: 200,
			expected: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
				AllowedPodNumber: 80,
				ScalarResources:  map[v1.ResourceName]int64{"hugepages-test": 2, "scalar2": 200},
			},
		},
	}

	for _, test := range tests {
		t.Run(string(test.scalarName), func(t *testing.T) {
			test.resource.AddScalar(test.scalarName, test.scalarQuantity)
			if !reflect.DeepEqual(test.expected, test.resource) {
				t.Errorf("expected: %#v, got: %#v", test.expected, test.resource)
			}
		})
	}
}

func TestSetMaxResource(t *testing.T) {
	tests := []struct {
		resource     *Resource
		resourceList v1.ResourceList
		expected     *Resource
	}{
		{
			resource: &Resource{},
			resourceList: map[v1.ResourceName]resource.Quantity{
				v1.ResourceCPU:              *resource.NewScaledQuantity(4, -3),
				v1.ResourceMemory:           *resource.NewQuantity(2000, resource.BinarySI),
				v1.ResourceEphemeralStorage: *resource.NewQuantity(5000, resource.BinarySI),
			},
			expected: &Resource{
				MilliCPU:         4,
				Memory:           2000,
				EphemeralStorage: 5000,
			},
		},
		{
			resource: &Resource{
				MilliCPU:         4,
				Memory:           4000,
				EphemeralStorage: 5000,
				ScalarResources:  map[v1.ResourceName]int64{"scalar.test/scalar1": 1, "hugepages-test": 2},
			},
			resourceList: map[v1.ResourceName]resource.Quantity{
				v1.ResourceCPU:                      *resource.NewScaledQuantity(4, -3),
				v1.ResourceMemory:                   *resource.NewQuantity(2000, resource.BinarySI),
				v1.ResourceEphemeralStorage:         *resource.NewQuantity(7000, resource.BinarySI),
				"scalar.test/scalar1":               *resource.NewQuantity(4, resource.DecimalSI),
				v1.ResourceHugePagesPrefix + "test": *resource.NewQuantity(5, resource.BinarySI),
			},
			expected: &Resource{
				MilliCPU:         4,
				Memory:           4000,
				EphemeralStorage: 7000,
				ScalarResources:  map[v1.ResourceName]int64{"scalar.test/scalar1": 4, "hugepages-test": 5},
			},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			test.resource.SetMaxResource(test.resourceList)
			if !reflect.DeepEqual(test.expected, test.resource) {
				t.Errorf("expected: %#v, got: %#v", test.expected, test.resource)
			}
		})
	}
}

func TestNewNodeInfo(t *testing.T) {
	nodeName := "test-node"
	pods := []*v1.Pod{
		st.MakePod().UID("test-1").Namespace("node_info_cache_test").Name("test-1").Node(nodeName).
			Containers([]v1.Container{st.MakeContainer().ResourceRequests(map[v1.ResourceName]string{
				v1.ResourceCPU:    "100m",
				v1.ResourceMemory: "500",
			}).ContainerPort([]v1.ContainerPort{{
				HostIP:   "127.0.0.1",
				HostPort: 80,
				Protocol: "TCP",
			}}).Obj()}).
			Obj(),

		st.MakePod().UID("test-2").Namespace("node_info_cache_test").Name("test-2").Node(nodeName).
			Containers([]v1.Container{st.MakeContainer().ResourceRequests(map[v1.ResourceName]string{
				v1.ResourceCPU:    "200m",
				v1.ResourceMemory: "1Ki",
			}).ContainerPort([]v1.ContainerPort{{
				HostIP:   "127.0.0.1",
				HostPort: 8080,
				Protocol: "TCP",
			}}).Obj()}).
			Obj(),
	}

	expected := &NodeInfo{
		Requested: &Resource{
			MilliCPU:         300,
			Memory:           1524,
			EphemeralStorage: 0,
			AllowedPodNumber: 0,
			ScalarResources:  map[v1.ResourceName]int64(nil),
		},
		NonZeroRequested: &Resource{
			MilliCPU:         300,
			Memory:           1524,
			EphemeralStorage: 0,
			AllowedPodNumber: 0,
			ScalarResources:  map[v1.ResourceName]int64(nil),
		},
		Allocatable: &Resource{},
		Generation:  2,
		UsedPorts: HostPortInfo{
			"127.0.0.1": map[ProtocolPort]struct{}{
				{Protocol: "TCP", Port: 80}:   {},
				{Protocol: "TCP", Port: 8080}: {},
			},
		},
		ImageStates:  map[string]*ImageStateSummary{},
		PVCRefCounts: map[string]int{},
		Pods: []*PodInfo{
			{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "node_info_cache_test",
						Name:      "test-1",
						UID:       types.UID("test-1"),
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("100m"),
										v1.ResourceMemory: resource.MustParse("500"),
									},
								},
								Ports: []v1.ContainerPort{
									{
										HostIP:   "127.0.0.1",
										HostPort: 80,
										Protocol: "TCP",
									},
								},
							},
						},
						NodeName: nodeName,
					},
				},
			},
			{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "node_info_cache_test",
						Name:      "test-2",
						UID:       types.UID("test-2"),
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("200m"),
										v1.ResourceMemory: resource.MustParse("1Ki"),
									},
								},
								Ports: []v1.ContainerPort{
									{
										HostIP:   "127.0.0.1",
										HostPort: 8080,
										Protocol: "TCP",
									},
								},
							},
						},
						NodeName: nodeName,
					},
				},
			},
		},
	}

	gen := generation
	ni := NewNodeInfo(pods...)
	if ni.Generation <= gen {
		t.Errorf("Generation is not incremented. previous: %v, current: %v", gen, ni.Generation)
	}
	expected.Generation = ni.Generation
	if !reflect.DeepEqual(expected, ni) {
		t.Errorf("expected: %#v, got: %#v", expected, ni)
	}
}

func TestNodeInfoClone(t *testing.T) {
	nodeName := "test-node"
	tests := []struct {
		nodeInfo *NodeInfo
		expected *NodeInfo
	}{
		{
			nodeInfo: &NodeInfo{
				Requested:        &Resource{},
				NonZeroRequested: &Resource{},
				Allocatable:      &Resource{},
				Generation:       2,
				UsedPorts: HostPortInfo{
					"127.0.0.1": map[ProtocolPort]struct{}{
						{Protocol: "TCP", Port: 80}:   {},
						{Protocol: "TCP", Port: 8080}: {},
					},
				},
				ImageStates:  map[string]*ImageStateSummary{},
				PVCRefCounts: map[string]int{},
				Pods: []*PodInfo{
					{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-1",
								UID:       types.UID("test-1"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("100m"),
												v1.ResourceMemory: resource.MustParse("500"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 80,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
							},
						},
					},
					{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-2",
								UID:       types.UID("test-2"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("200m"),
												v1.ResourceMemory: resource.MustParse("1Ki"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 8080,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
							},
						},
					},
				},
			},
			expected: &NodeInfo{
				Requested:        &Resource{},
				NonZeroRequested: &Resource{},
				Allocatable:      &Resource{},
				Generation:       2,
				UsedPorts: HostPortInfo{
					"127.0.0.1": map[ProtocolPort]struct{}{
						{Protocol: "TCP", Port: 80}:   {},
						{Protocol: "TCP", Port: 8080}: {},
					},
				},
				ImageStates:  map[string]*ImageStateSummary{},
				PVCRefCounts: map[string]int{},
				Pods: []*PodInfo{
					{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-1",
								UID:       types.UID("test-1"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("100m"),
												v1.ResourceMemory: resource.MustParse("500"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 80,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
							},
						},
					},
					{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-2",
								UID:       types.UID("test-2"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("200m"),
												v1.ResourceMemory: resource.MustParse("1Ki"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 8080,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
							},
						},
					},
				},
			},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			ni := test.nodeInfo.Snapshot()
			// Modify the field to check if the result is a clone of the origin one.
			test.nodeInfo.Generation += 10
			test.nodeInfo.UsedPorts.Remove("127.0.0.1", "TCP", 80)
			if !reflect.DeepEqual(test.expected, ni) {
				t.Errorf("expected: %#v, got: %#v", test.expected, ni)
			}
		})
	}
}

func TestNodeInfoAddPod(t *testing.T) {
	nodeName := "test-node"
	pods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "node_info_cache_test",
				Name:      "test-1",
				UID:       types.UID("test-1"),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("500"),
							},
						},
						Ports: []v1.ContainerPort{
							{
								HostIP:   "127.0.0.1",
								HostPort: 80,
								Protocol: "TCP",
							},
						},
					},
				},
				NodeName: nodeName,
				Overhead: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("500m"),
				},
				Volumes: []v1.Volume{
					{
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: "pvc-1",
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "node_info_cache_test",
				Name:      "test-2",
				UID:       types.UID("test-2"),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU: resource.MustParse("200m"),
							},
						},
						Ports: []v1.ContainerPort{
							{
								HostIP:   "127.0.0.1",
								HostPort: 8080,
								Protocol: "TCP",
							},
						},
					},
				},
				NodeName: nodeName,
				Overhead: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("500m"),
					v1.ResourceMemory: resource.MustParse("500"),
				},
				Volumes: []v1.Volume{
					{
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: "pvc-1",
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "node_info_cache_test",
				Name:      "test-3",
				UID:       types.UID("test-3"),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU: resource.MustParse("200m"),
							},
						},
						Ports: []v1.ContainerPort{
							{
								HostIP:   "127.0.0.1",
								HostPort: 8080,
								Protocol: "TCP",
							},
						},
					},
				},
				InitContainers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("500m"),
								v1.ResourceMemory: resource.MustParse("200Mi"),
							},
						},
					},
				},
				NodeName: nodeName,
				Overhead: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("500m"),
					v1.ResourceMemory: resource.MustParse("500"),
				},
				Volumes: []v1.Volume{
					{
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: "pvc-2",
							},
						},
					},
				},
			},
		},
	}
	expected := &NodeInfo{
		node: &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node",
			},
		},
		Requested: &Resource{
			MilliCPU:         2300,
			Memory:           209716700, //1500 + 200MB in initContainers
			EphemeralStorage: 0,
			AllowedPodNumber: 0,
			ScalarResources:  map[v1.ResourceName]int64(nil),
		},
		NonZeroRequested: &Resource{
			MilliCPU:         2300,
			Memory:           419431900, //200MB(initContainers) + 200MB(default memory value) + 1500 specified in requests/overhead
			EphemeralStorage: 0,
			AllowedPodNumber: 0,
			ScalarResources:  map[v1.ResourceName]int64(nil),
		},
		Allocatable: &Resource{},
		Generation:  2,
		UsedPorts: HostPortInfo{
			"127.0.0.1": map[ProtocolPort]struct{}{
				{Protocol: "TCP", Port: 80}:   {},
				{Protocol: "TCP", Port: 8080}: {},
			},
		},
		ImageStates:  map[string]*ImageStateSummary{},
		PVCRefCounts: map[string]int{"node_info_cache_test/pvc-1": 2, "node_info_cache_test/pvc-2": 1},
		Pods: []*PodInfo{
			{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "node_info_cache_test",
						Name:      "test-1",
						UID:       types.UID("test-1"),
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("100m"),
										v1.ResourceMemory: resource.MustParse("500"),
									},
								},
								Ports: []v1.ContainerPort{
									{
										HostIP:   "127.0.0.1",
										HostPort: 80,
										Protocol: "TCP",
									},
								},
							},
						},
						NodeName: nodeName,
						Overhead: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("500m"),
						},
						Volumes: []v1.Volume{
							{
								VolumeSource: v1.VolumeSource{
									PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
										ClaimName: "pvc-1",
									},
								},
							},
						},
					},
				},
			},
			{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "node_info_cache_test",
						Name:      "test-2",
						UID:       types.UID("test-2"),
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("200m"),
									},
								},
								Ports: []v1.ContainerPort{
									{
										HostIP:   "127.0.0.1",
										HostPort: 8080,
										Protocol: "TCP",
									},
								},
							},
						},
						NodeName: nodeName,
						Overhead: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("500m"),
							v1.ResourceMemory: resource.MustParse("500"),
						},
						Volumes: []v1.Volume{
							{
								VolumeSource: v1.VolumeSource{
									PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
										ClaimName: "pvc-1",
									},
								},
							},
						},
					},
				},
			},
			{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "node_info_cache_test",
						Name:      "test-3",
						UID:       types.UID("test-3"),
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("200m"),
									},
								},
								Ports: []v1.ContainerPort{
									{
										HostIP:   "127.0.0.1",
										HostPort: 8080,
										Protocol: "TCP",
									},
								},
							},
						},
						InitContainers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("500m"),
										v1.ResourceMemory: resource.MustParse("200Mi"),
									},
								},
							},
						},
						NodeName: nodeName,
						Overhead: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("500m"),
							v1.ResourceMemory: resource.MustParse("500"),
						},
						Volumes: []v1.Volume{
							{
								VolumeSource: v1.VolumeSource{
									PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
										ClaimName: "pvc-2",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	ni := fakeNodeInfo()
	gen := ni.Generation
	for _, pod := range pods {
		ni.AddPod(pod)
		if ni.Generation <= gen {
			t.Errorf("Generation is not incremented. Prev: %v, current: %v", gen, ni.Generation)
		}
		gen = ni.Generation
	}

	expected.Generation = ni.Generation
	if !reflect.DeepEqual(expected, ni) {
		t.Errorf("expected: %#v, got: %#v", expected, ni)
	}
}

func TestNodeInfoRemovePod(t *testing.T) {
	nodeName := "test-node"
	pods := []*v1.Pod{
		st.MakePod().UID("test-1").Namespace("node_info_cache_test").Name("test-1").Node(nodeName).
			Containers([]v1.Container{st.MakeContainer().ResourceRequests(map[v1.ResourceName]string{
				v1.ResourceCPU:    "100m",
				v1.ResourceMemory: "500",
			}).ContainerPort([]v1.ContainerPort{{
				HostIP:   "127.0.0.1",
				HostPort: 80,
				Protocol: "TCP",
			}}).Obj()}).
			Volumes([]v1.Volume{{VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: "pvc-1"}}}}).
			Obj(),

		st.MakePod().UID("test-2").Namespace("node_info_cache_test").Name("test-2").Node(nodeName).
			Containers([]v1.Container{st.MakeContainer().ResourceRequests(map[v1.ResourceName]string{
				v1.ResourceCPU:    "200m",
				v1.ResourceMemory: "1Ki",
			}).ContainerPort([]v1.ContainerPort{{
				HostIP:   "127.0.0.1",
				HostPort: 8080,
				Protocol: "TCP",
			}}).Obj()}).
			Obj(),
	}

	// add pod Overhead
	for _, pod := range pods {
		pod.Spec.Overhead = v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("500m"),
			v1.ResourceMemory: resource.MustParse("500"),
		}
	}

	tests := []struct {
		pod              *v1.Pod
		errExpected      bool
		expectedNodeInfo *NodeInfo
	}{
		{
			pod:         st.MakePod().UID("non-exist").Namespace("node_info_cache_test").Node(nodeName).Obj(),
			errExpected: true,
			expectedNodeInfo: &NodeInfo{
				node: &v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test-node",
					},
				},
				Requested: &Resource{
					MilliCPU:         1300,
					Memory:           2524,
					EphemeralStorage: 0,
					AllowedPodNumber: 0,
					ScalarResources:  map[v1.ResourceName]int64(nil),
				},
				NonZeroRequested: &Resource{
					MilliCPU:         1300,
					Memory:           2524,
					EphemeralStorage: 0,
					AllowedPodNumber: 0,
					ScalarResources:  map[v1.ResourceName]int64(nil),
				},
				Allocatable: &Resource{},
				Generation:  2,
				UsedPorts: HostPortInfo{
					"127.0.0.1": map[ProtocolPort]struct{}{
						{Protocol: "TCP", Port: 80}:   {},
						{Protocol: "TCP", Port: 8080}: {},
					},
				},
				ImageStates:  map[string]*ImageStateSummary{},
				PVCRefCounts: map[string]int{"node_info_cache_test/pvc-1": 1},
				Pods: []*PodInfo{
					{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-1",
								UID:       types.UID("test-1"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("100m"),
												v1.ResourceMemory: resource.MustParse("500"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 80,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
								Overhead: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("500m"),
									v1.ResourceMemory: resource.MustParse("500"),
								},
								Volumes: []v1.Volume{
									{
										VolumeSource: v1.VolumeSource{
											PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
												ClaimName: "pvc-1",
											},
										},
									},
								},
							},
						},
					},
					{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-2",
								UID:       types.UID("test-2"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("200m"),
												v1.ResourceMemory: resource.MustParse("1Ki"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 8080,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
								Overhead: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("500m"),
									v1.ResourceMemory: resource.MustParse("500"),
								},
							},
						},
					},
				},
			},
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "node_info_cache_test",
					Name:      "test-1",
					UID:       types.UID("test-1"),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("500"),
								},
							},
							Ports: []v1.ContainerPort{
								{
									HostIP:   "127.0.0.1",
									HostPort: 80,
									Protocol: "TCP",
								},
							},
						},
					},
					NodeName: nodeName,
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("500m"),
						v1.ResourceMemory: resource.MustParse("500"),
					},
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: "pvc-1",
								},
							},
						},
					},
				},
			},
			errExpected: false,
			expectedNodeInfo: &NodeInfo{
				node: &v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: "test-node",
					},
				},
				Requested: &Resource{
					MilliCPU:         700,
					Memory:           1524,
					EphemeralStorage: 0,
					AllowedPodNumber: 0,
					ScalarResources:  map[v1.ResourceName]int64(nil),
				},
				NonZeroRequested: &Resource{
					MilliCPU:         700,
					Memory:           1524,
					EphemeralStorage: 0,
					AllowedPodNumber: 0,
					ScalarResources:  map[v1.ResourceName]int64(nil),
				},
				Allocatable: &Resource{},
				Generation:  3,
				UsedPorts: HostPortInfo{
					"127.0.0.1": map[ProtocolPort]struct{}{
						{Protocol: "TCP", Port: 8080}: {},
					},
				},
				ImageStates:  map[string]*ImageStateSummary{},
				PVCRefCounts: map[string]int{},
				Pods: []*PodInfo{
					{
						Pod: &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Namespace: "node_info_cache_test",
								Name:      "test-2",
								UID:       types.UID("test-2"),
							},
							Spec: v1.PodSpec{
								Containers: []v1.Container{
									{
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("200m"),
												v1.ResourceMemory: resource.MustParse("1Ki"),
											},
										},
										Ports: []v1.ContainerPort{
											{
												HostIP:   "127.0.0.1",
												HostPort: 8080,
												Protocol: "TCP",
											},
										},
									},
								},
								NodeName: nodeName,
								Overhead: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("500m"),
									v1.ResourceMemory: resource.MustParse("500"),
								},
							},
						},
					},
				},
			},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			ni := fakeNodeInfo(pods...)

			gen := ni.Generation
			err := ni.RemovePod(logger, test.pod)
			if err != nil {
				if test.errExpected {
					expectedErrorMsg := fmt.Errorf("no corresponding pod %s in pods of node %s", test.pod.Name, ni.Node().Name)
					if expectedErrorMsg == err {
						t.Errorf("expected error: %v, got: %v", expectedErrorMsg, err)
					}
				} else {
					t.Errorf("expected no error, got: %v", err)
				}
			} else {
				if ni.Generation <= gen {
					t.Errorf("Generation is not incremented. Prev: %v, current: %v", gen, ni.Generation)
				}
			}

			test.expectedNodeInfo.Generation = ni.Generation
			if !reflect.DeepEqual(test.expectedNodeInfo, ni) {
				t.Errorf("expected: %#v, got: %#v", test.expectedNodeInfo, ni)
			}
		})
	}
}

func fakeNodeInfo(pods ...*v1.Pod) *NodeInfo {
	ni := NewNodeInfo(pods...)
	ni.SetNode(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
	})
	return ni
}

type hostPortInfoParam struct {
	protocol, ip string
	port         int32
}

func TestHostPortInfo_AddRemove(t *testing.T) {
	tests := []struct {
		desc    string
		added   []hostPortInfoParam
		removed []hostPortInfoParam
		length  int
	}{
		{
			desc: "normal add case",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
				// this might not make sense in real case, but the struct doesn't forbid it.
				{"TCP", "0.0.0.0", 79},
				{"UDP", "0.0.0.0", 80},
				{"TCP", "0.0.0.0", 81},
				{"TCP", "0.0.0.0", 82},
				{"TCP", "0.0.0.0", 0},
				{"TCP", "0.0.0.0", -1},
			},
			length: 8,
		},
		{
			desc: "empty ip and protocol add should work",
			added: []hostPortInfoParam{
				{"", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"", "127.0.0.1", 81},
				{"", "127.0.0.1", 82},
				{"", "", 79},
				{"UDP", "", 80},
				{"", "", 81},
				{"", "", 82},
				{"", "", 0},
				{"", "", -1},
			},
			length: 8,
		},
		{
			desc: "normal remove case",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
				{"TCP", "0.0.0.0", 79},
				{"UDP", "0.0.0.0", 80},
				{"TCP", "0.0.0.0", 81},
				{"TCP", "0.0.0.0", 82},
			},
			removed: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
				{"TCP", "0.0.0.0", 79},
				{"UDP", "0.0.0.0", 80},
				{"TCP", "0.0.0.0", 81},
				{"TCP", "0.0.0.0", 82},
			},
			length: 0,
		},
		{
			desc: "empty ip and protocol remove should work",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
				{"TCP", "0.0.0.0", 79},
				{"UDP", "0.0.0.0", 80},
				{"TCP", "0.0.0.0", 81},
				{"TCP", "0.0.0.0", 82},
			},
			removed: []hostPortInfoParam{
				{"", "127.0.0.1", 79},
				{"", "127.0.0.1", 81},
				{"", "127.0.0.1", 82},
				{"UDP", "127.0.0.1", 80},
				{"", "", 79},
				{"", "", 81},
				{"", "", 82},
				{"UDP", "", 80},
			},
			length: 0,
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			hp := make(HostPortInfo)
			for _, param := range test.added {
				hp.Add(param.ip, param.protocol, param.port)
			}
			for _, param := range test.removed {
				hp.Remove(param.ip, param.protocol, param.port)
			}
			if hp.Len() != test.length {
				t.Errorf("%v failed: expect length %d; got %d", test.desc, test.length, hp.Len())
				t.Error(hp)
			}
		})
	}
}

func TestHostPortInfo_Check(t *testing.T) {
	tests := []struct {
		desc   string
		added  []hostPortInfoParam
		check  hostPortInfoParam
		expect bool
	}{
		{
			desc: "empty check should check 0.0.0.0 and TCP",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 80},
			},
			check:  hostPortInfoParam{"", "", 81},
			expect: false,
		},
		{
			desc: "empty check should check 0.0.0.0 and TCP (conflicted)",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 80},
			},
			check:  hostPortInfoParam{"", "", 80},
			expect: true,
		},
		{
			desc: "empty port check should pass",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 80},
			},
			check:  hostPortInfoParam{"", "", 0},
			expect: false,
		},
		{
			desc: "0.0.0.0 should check all registered IPs",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 80},
			},
			check:  hostPortInfoParam{"TCP", "0.0.0.0", 80},
			expect: true,
		},
		{
			desc: "0.0.0.0 with different protocol should be allowed",
			added: []hostPortInfoParam{
				{"UDP", "127.0.0.1", 80},
			},
			check:  hostPortInfoParam{"TCP", "0.0.0.0", 80},
			expect: false,
		},
		{
			desc: "0.0.0.0 with different port should be allowed",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
			},
			check:  hostPortInfoParam{"TCP", "0.0.0.0", 80},
			expect: false,
		},
		{
			desc: "normal ip should check all registered 0.0.0.0",
			added: []hostPortInfoParam{
				{"TCP", "0.0.0.0", 80},
			},
			check:  hostPortInfoParam{"TCP", "127.0.0.1", 80},
			expect: true,
		},
		{
			desc: "normal ip with different port/protocol should be allowed (0.0.0.0)",
			added: []hostPortInfoParam{
				{"TCP", "0.0.0.0", 79},
				{"UDP", "0.0.0.0", 80},
				{"TCP", "0.0.0.0", 81},
				{"TCP", "0.0.0.0", 82},
			},
			check:  hostPortInfoParam{"TCP", "127.0.0.1", 80},
			expect: false,
		},
		{
			desc: "normal ip with different port/protocol should be allowed",
			added: []hostPortInfoParam{
				{"TCP", "127.0.0.1", 79},
				{"UDP", "127.0.0.1", 80},
				{"TCP", "127.0.0.1", 81},
				{"TCP", "127.0.0.1", 82},
			},
			check:  hostPortInfoParam{"TCP", "127.0.0.1", 80},
			expect: false,
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			hp := make(HostPortInfo)
			for _, param := range test.added {
				hp.Add(param.ip, param.protocol, param.port)
			}
			if hp.CheckConflict(test.check.ip, test.check.protocol, test.check.port) != test.expect {
				t.Errorf("expected %t; got %t", test.expect, !test.expect)
			}
		})
	}
}

func TestGetNamespacesFromPodAffinityTerm(t *testing.T) {
	tests := []struct {
		name string
		term *v1.PodAffinityTerm
		want sets.Set[string]
	}{
		{
			name: "podAffinityTerm_namespace_empty",
			term: &v1.PodAffinityTerm{},
			want: sets.Set[string]{metav1.NamespaceDefault: sets.Empty{}},
		},
		{
			name: "podAffinityTerm_namespace_not_empty",
			term: &v1.PodAffinityTerm{
				Namespaces: []string{metav1.NamespacePublic, metav1.NamespaceSystem},
			},
			want: sets.New(metav1.NamespacePublic, metav1.NamespaceSystem),
		},
		{
			name: "podAffinityTerm_namespace_selector_not_nil",
			term: &v1.PodAffinityTerm{
				NamespaceSelector: &metav1.LabelSelector{},
			},
			want: sets.Set[string]{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := getNamespacesFromPodAffinityTerm(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "topologies_pod",
					Namespace: metav1.NamespaceDefault,
				},
			}, test.term)
			if diff := cmp.Diff(test.want, got); diff != "" {
				t.Errorf("unexpected diff (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestFitError_Error(t *testing.T) {
	tests := []struct {
		name          string
		pod           *v1.Pod
		numAllNodes   int
		diagnosis     Diagnosis
		wantReasonMsg string
	}{
		{
			name:        "nodes failed Prefilter plugin",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "Node(s) failed PreFilter plugin FalsePreFilter",
				NodeToStatus: NewNodeToStatus(map[string]*Status{
					// They're inserted by the framework.
					// We don't include them in the reason message because they'd be just duplicates.
					"node1": NewStatus(Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
					"node2": NewStatus(Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
					"node3": NewStatus(Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
				}, NewStatus(UnschedulableAndUnresolvable)),
			},
			wantReasonMsg: "0/3 nodes are available: Node(s) failed PreFilter plugin FalsePreFilter.",
		},
		{
			name:        "nodes failed Prefilter plugin and the preemption also failed",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "Node(s) failed PreFilter plugin FalsePreFilter",
				NodeToStatus: NewNodeToStatus(map[string]*Status{
					// They're inserted by the framework.
					// We don't include them in the reason message because they'd be just duplicates.
					"node1": NewStatus(Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
					"node2": NewStatus(Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
					"node3": NewStatus(Unschedulable, "Node(s) failed PreFilter plugin FalsePreFilter"),
				}, NewStatus(UnschedulableAndUnresolvable)),
				// PostFilterMsg will be included.
				PostFilterMsg: "Error running PostFilter plugin FailedPostFilter",
			},
			wantReasonMsg: "0/3 nodes are available: Node(s) failed PreFilter plugin FalsePreFilter. Error running PostFilter plugin FailedPostFilter",
		},
		{
			name:        "nodes failed one Filter plugin with an empty PostFilterMsg",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "",
				NodeToStatus: NewNodeToStatus(map[string]*Status{
					"node1": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node2": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node3": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
				}, NewStatus(UnschedulableAndUnresolvable)),
			},
			wantReasonMsg: "0/3 nodes are available: 3 Node(s) failed Filter plugin FalseFilter-1.",
		},
		{
			name:        "nodes failed one Filter plugin with a non-empty PostFilterMsg",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "",
				NodeToStatus: NewNodeToStatus(map[string]*Status{
					"node1": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node2": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node3": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
				}, NewStatus(UnschedulableAndUnresolvable)),
				PostFilterMsg: "Error running PostFilter plugin FailedPostFilter",
			},
			wantReasonMsg: "0/3 nodes are available: 3 Node(s) failed Filter plugin FalseFilter-1. Error running PostFilter plugin FailedPostFilter",
		},
		{
			name:        "nodes failed two Filter plugins with an empty PostFilterMsg",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "",
				NodeToStatus: NewNodeToStatus(map[string]*Status{
					"node1": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node2": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node3": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-2"),
				}, NewStatus(UnschedulableAndUnresolvable)),
			},
			wantReasonMsg: "0/3 nodes are available: 1 Node(s) failed Filter plugin FalseFilter-2, 2 Node(s) failed Filter plugin FalseFilter-1.",
		},
		{
			name:        "nodes failed two Filter plugins with a non-empty PostFilterMsg",
			numAllNodes: 3,
			diagnosis: Diagnosis{
				PreFilterMsg: "",
				NodeToStatus: NewNodeToStatus(map[string]*Status{
					"node1": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node2": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-1"),
					"node3": NewStatus(Unschedulable, "Node(s) failed Filter plugin FalseFilter-2"),
				}, NewStatus(UnschedulableAndUnresolvable)),
				PostFilterMsg: "Error running PostFilter plugin FailedPostFilter",
			},
			wantReasonMsg: "0/3 nodes are available: 1 Node(s) failed Filter plugin FalseFilter-2, 2 Node(s) failed Filter plugin FalseFilter-1. Error running PostFilter plugin FailedPostFilter",
		},
		{
			name:        "failed to Permit on node",
			numAllNodes: 1,
			diagnosis: Diagnosis{
				NodeToStatus: NewNodeToStatus(map[string]*Status{
					// There should be only one node here.
					"node1": NewStatus(Unschedulable, "Node failed Permit plugin Permit-1"),
				}, NewStatus(UnschedulableAndUnresolvable)),
			},
			wantReasonMsg: "0/1 nodes are available: 1 Node failed Permit plugin Permit-1.",
		},
		{
			name:        "failed to Reserve on node",
			numAllNodes: 1,
			diagnosis: Diagnosis{
				NodeToStatus: NewNodeToStatus(map[string]*Status{
					// There should be only one node here.
					"node1": NewStatus(Unschedulable, "Node failed Reserve plugin Reserve-1"),
				}, NewStatus(UnschedulableAndUnresolvable)),
			},
			wantReasonMsg: "0/1 nodes are available: 1 Node failed Reserve plugin Reserve-1.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &FitError{
				Pod:         tt.pod,
				NumAllNodes: tt.numAllNodes,
				Diagnosis:   tt.diagnosis,
			}
			if gotReasonMsg := f.Error(); gotReasonMsg != tt.wantReasonMsg {
				t.Errorf("Error() = Got: %v Want: %v", gotReasonMsg, tt.wantReasonMsg)
			}
		})
	}
}

func TestCalculatePodResourcesWithResize(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	cpu500m := resource.MustParse("500m")
	mem500M := resource.MustParse("500Mi")
	cpu700m := resource.MustParse("700m")
	mem800M := resource.MustParse("800Mi")
	testpod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "pod_resize_test",
			Name:      "testpod",
			UID:       types.UID("testpod"),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      "c1",
					Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M}},
				},
			},
		},
		Status: v1.PodStatus{
			Phase:  v1.PodRunning,
			Resize: "",
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name:               "c1",
					AllocatedResources: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
				},
			},
		},
	}

	tests := []struct {
		name               string
		requests           v1.ResourceList
		allocatedResources v1.ResourceList
		resizeStatus       v1.PodResizeStatus
		expectedResource   Resource
		expectedNon0CPU    int64
		expectedNon0Mem    int64
	}{
		{
			name:               "Pod with no pending resize",
			requests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			allocatedResources: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			resizeStatus:       "",
			expectedResource:   Resource{MilliCPU: cpu500m.MilliValue(), Memory: mem500M.Value()},
			expectedNon0CPU:    cpu500m.MilliValue(),
			expectedNon0Mem:    mem500M.Value(),
		},
		{
			name:               "Pod with resize in progress",
			requests:           v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			allocatedResources: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			resizeStatus:       v1.PodResizeStatusInProgress,
			expectedResource:   Resource{MilliCPU: cpu500m.MilliValue(), Memory: mem500M.Value()},
			expectedNon0CPU:    cpu500m.MilliValue(),
			expectedNon0Mem:    mem500M.Value(),
		},
		{
			name:               "Pod with deferred resize",
			requests:           v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			allocatedResources: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			resizeStatus:       v1.PodResizeStatusDeferred,
			expectedResource:   Resource{MilliCPU: cpu700m.MilliValue(), Memory: mem800M.Value()},
			expectedNon0CPU:    cpu700m.MilliValue(),
			expectedNon0Mem:    mem800M.Value(),
		},
		{
			name:               "Pod with infeasible resize",
			requests:           v1.ResourceList{v1.ResourceCPU: cpu700m, v1.ResourceMemory: mem800M},
			allocatedResources: v1.ResourceList{v1.ResourceCPU: cpu500m, v1.ResourceMemory: mem500M},
			resizeStatus:       v1.PodResizeStatusInfeasible,
			expectedResource:   Resource{MilliCPU: cpu500m.MilliValue(), Memory: mem500M.Value()},
			expectedNon0CPU:    cpu500m.MilliValue(),
			expectedNon0Mem:    mem500M.Value(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := testpod.DeepCopy()
			pod.Spec.Containers[0].Resources.Requests = tt.requests
			pod.Status.ContainerStatuses[0].AllocatedResources = tt.allocatedResources
			pod.Status.Resize = tt.resizeStatus

			res, non0CPU, non0Mem := calculateResource(pod)
			if !reflect.DeepEqual(tt.expectedResource, res) {
				t.Errorf("Test: %s expected resource: %+v, got: %+v", tt.name, tt.expectedResource, res)
			}
			if non0CPU != tt.expectedNon0CPU {
				t.Errorf("Test: %s expected non0CPU: %d, got: %d", tt.name, tt.expectedNon0CPU, non0CPU)
			}
			if non0Mem != tt.expectedNon0Mem {
				t.Errorf("Test: %s expected non0Mem: %d, got: %d", tt.name, tt.expectedNon0Mem, non0Mem)
			}
		})
	}
}

func TestCloudEvent_Match(t *testing.T) {
	testCases := []struct {
		name        string
		event       ClusterEvent
		comingEvent ClusterEvent
		wantResult  bool
	}{
		{
			name:        "wildcard event matches with all kinds of coming events",
			event:       ClusterEvent{Resource: WildCard, ActionType: All},
			comingEvent: ClusterEvent{Resource: Pod, ActionType: UpdateNodeLabel},
			wantResult:  true,
		},
		{
			name:        "event with resource = 'Pod' matching with coming events carries same actionType",
			event:       ClusterEvent{Resource: Pod, ActionType: UpdateNodeLabel | UpdateNodeTaint},
			comingEvent: ClusterEvent{Resource: Pod, ActionType: UpdateNodeLabel},
			wantResult:  true,
		},
		{
			name:        "event with resource = '*' matching with coming events carries same actionType",
			event:       ClusterEvent{Resource: WildCard, ActionType: UpdateNodeLabel},
			comingEvent: ClusterEvent{Resource: Pod, ActionType: UpdateNodeLabel},
			wantResult:  true,
		},
		{
			name:        "event with resource = '*' matching with coming events carries different actionType",
			event:       ClusterEvent{Resource: WildCard, ActionType: UpdateNodeLabel},
			comingEvent: ClusterEvent{Resource: Pod, ActionType: UpdateNodeAllocatable},
			wantResult:  false,
		},
		{
			name:        "event matching with coming events carries '*' resources",
			event:       ClusterEvent{Resource: Pod, ActionType: UpdateNodeLabel},
			comingEvent: ClusterEvent{Resource: WildCard, ActionType: UpdateNodeLabel},
			wantResult:  false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.event.Match(tc.comingEvent)
			if got != tc.wantResult {
				t.Fatalf("unexpected result")
			}
		})
	}
}

func TestNodeInfoKMetadata(t *testing.T) {
	tCtx := ktesting.Init(t, initoption.BufferLogs(true))
	logger := tCtx.Logger()
	logger.Info("Some NodeInfo slice", "nodes", klog.KObjSlice([]*NodeInfo{nil, {}, {node: &v1.Node{}}, {node: &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "worker"}}}}))

	output := logger.GetSink().(ktesting.Underlier).GetBuffer().String()

	// The initial nil entry gets turned into empty ObjectRef by klog,
	// which becomes an empty string during output formatting.
	if !strings.Contains(output, `Some NodeInfo slice nodes=["","<no node>","","worker"]`) {
		tCtx.Fatalf("unexpected output:\n%s", output)
	}
}
