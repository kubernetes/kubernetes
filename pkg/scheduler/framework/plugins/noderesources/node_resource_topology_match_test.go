/*
Copyright 2020 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"testing"


	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	intstr "k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	topologyv1alpha1 "k8s.io/noderesourcetopology-api/pkg/apis/topology/v1alpha1"
)

const nicResourceName = "vendor/nic1"
const notExistingNICResourceName = "vendor/notexistingnic"
const containerName = "container1"

func makePodByResourceList(resources *v1.ResourceList) (*v1.Pod) {
	return &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{{
				Resources: v1.ResourceRequirements{
					Requests: *resources,
				},
			}},
		}}
}

func TestTopologyRequests(t *testing.T) {
	nodes := nodeTopologyMap{}
	nodes["node1"] = topologyv1alpha1.NodeResourceTopology{
		ObjectMeta: metav1.ObjectMeta{Name: "node1"},
		TopologyPolicies: []string{string(v1.SingleNUMANodeTopologyManagerPolicy)},
		Zones: topologyv1alpha1.ZoneMap {
			"node-0": topologyv1alpha1.Zone{
				Type: "Node", Resources: topologyv1alpha1.ResourceInfoMap{v1.ResourceCPU: topologyv1alpha1.ResourceInfo{Capacity: intstr.Parse("20"), Allocatable: intstr.Parse("4")}, v1.ResourceMemory: topologyv1alpha1.ResourceInfo{Capacity: intstr.Parse("8Gi"), Allocatable: intstr.Parse("8Gi")}, nicResourceName: topologyv1alpha1.ResourceInfo{Capacity: intstr.Parse("30"), Allocatable: intstr.Parse("10") }},
			}, "node-1": topologyv1alpha1.Zone{
				Type: "Node", Resources: topologyv1alpha1.ResourceInfoMap{v1.ResourceCPU: topologyv1alpha1.ResourceInfo{Capacity: intstr.Parse("30"), Allocatable: intstr.Parse("8")}, v1.ResourceMemory: topologyv1alpha1.ResourceInfo{Capacity: intstr.Parse("8Gi"), Allocatable: intstr.Parse("8Gi") }, nicResourceName: topologyv1alpha1.ResourceInfo{Capacity : intstr.Parse("30"), Allocatable: intstr.Parse("10") }}}},
	}

	nodes["node2"] = topologyv1alpha1.NodeResourceTopology{
		ObjectMeta: metav1.ObjectMeta{Name: "node1"},
		TopologyPolicies: []string{string(v1.SingleNUMANodeTopologyManagerPolicy)},
		Zones: topologyv1alpha1.ZoneMap {
			"node-0": topologyv1alpha1.Zone{
				Type: "Node", Resources: topologyv1alpha1.ResourceInfoMap{v1.ResourceCPU: topologyv1alpha1.ResourceInfo{Capacity: intstr.Parse("20"), Allocatable: intstr.Parse("2")}, v1.ResourceMemory: topologyv1alpha1.ResourceInfo{Capacity: intstr.Parse("8Gi"), Allocatable: intstr.Parse("4Gi")}, nicResourceName: topologyv1alpha1.ResourceInfo{Capacity: intstr.Parse("30"), Allocatable: intstr.Parse("5") }},
			}, "node-1": topologyv1alpha1.Zone{
				Type: "Node", Resources: topologyv1alpha1.ResourceInfoMap{v1.ResourceCPU: topologyv1alpha1.ResourceInfo{Capacity: intstr.Parse("30"), Allocatable: intstr.Parse("4")}, v1.ResourceMemory: topologyv1alpha1.ResourceInfo{Capacity: intstr.Parse("8Gi"), Allocatable: intstr.Parse("4Gi") }, nicResourceName: topologyv1alpha1.ResourceInfo{Capacity : intstr.Parse("30"), Allocatable: intstr.Parse("2") }}}},
	}
	node1Resources := v1.ResourceList{
		v1.ResourceCPU:    *resource.NewQuantity(12, resource.DecimalSI),
		v1.ResourceMemory: resource.MustParse("16Gi"),
		v1.ResourcePods:   *resource.NewQuantity(20, resource.DecimalSI),
		nicResourceName:   *resource.NewQuantity(14, resource.DecimalSI),
	}
	node1 := v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node1"},
		Status: v1.NodeStatus{
			Capacity: node1Resources,
			Allocatable: node1Resources,
		},
	}

	node2Resources := v1.ResourceList{
		v1.ResourceCPU:    *resource.NewQuantity(6, resource.DecimalSI),
		v1.ResourceMemory: resource.MustParse("8Gi"),
		v1.ResourcePods:   *resource.NewQuantity(20, resource.DecimalSI),
		nicResourceName:   *resource.NewQuantity(7, resource.DecimalSI),
	}
	node2 := v1.Node{ ObjectMeta: metav1.ObjectMeta{Name: "node1"}, Status: v1.NodeStatus{
			Capacity: node2Resources,
			Allocatable: node2Resources,
		},
	}

  // Test different QoS Guaranteed/Burstable/BestEffort
	topologyTests := []struct {
		pod            *v1.Pod
		nodeTopologies *nodeTopologyMap
		name           string
		node           v1.Node
		wantStatus     *framework.Status
	}{
		{
			pod: &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{{}},}},
			nodeTopologies: &nodes,
			name: "Best effort QoS, pod fit",
			node: node1,
			wantStatus: nil,
		},
		{
			pod: makePodByResourceList(&v1.ResourceList{
						v1.ResourceCPU: *resource.NewQuantity(2, resource.DecimalSI),
						v1.ResourceMemory: resource.MustParse("2Gi"),
						nicResourceName: *resource.NewQuantity(3, resource.DecimalSI),}),
			nodeTopologies: &nodes,
			name: "Guaranteed QoS, pod fit",
			node: node2,
			wantStatus: nil,
		},
		{
			pod: makePodByResourceList(&v1.ResourceList{
						v1.ResourceCPU: *resource.NewQuantity(4, resource.DecimalSI),
						nicResourceName: *resource.NewQuantity(3, resource.DecimalSI),}),
			nodeTopologies: &nodes,
			name: "Burstable QoS, pod fit",
			node: node2,
			wantStatus: nil,
		},
		{
			pod: makePodByResourceList(&v1.ResourceList{
						v1.ResourceCPU: *resource.NewQuantity(14, resource.DecimalSI),
						nicResourceName: *resource.NewQuantity(3, resource.DecimalSI),}),
			nodeTopologies: &nodes,
			name: "Burstable QoS, pod doesn't fit",
			node: node2,
			wantStatus: nil, // number of cpu is exceeded, but in case of burstable QoS for cpu resources we rely on fit.go
		},
		{
			pod: makePodByResourceList(&v1.ResourceList{
						v1.ResourceCPU: *resource.NewQuantity(4, resource.DecimalSI),
						nicResourceName: *resource.NewQuantity(11, resource.DecimalSI),}),
			nodeTopologies: &nodes,
			name: "Burstable QoS, pod doesn't fit",
			node: node2,
			wantStatus: framework.NewStatus(framework.Unschedulable, fmt.Sprintf("Can't align container: %s", containerName)),
		},
		{
			pod: makePodByResourceList(&v1.ResourceList{
						v1.ResourceCPU: *resource.NewQuantity(9, resource.DecimalSI),
						v1.ResourceMemory: resource.MustParse("1Gi"),
						nicResourceName: *resource.NewQuantity(3, resource.DecimalSI),}),
			nodeTopologies: &nodes,
			name: "Guaranteed QoS, pod doesn't fit",
			node: node1,
			wantStatus: framework.NewStatus(framework.Unschedulable, fmt.Sprintf("Can't align container: %s", containerName)),
		},
		{
			pod: makePodByResourceList(&v1.ResourceList{
						v1.ResourceCPU: *resource.NewQuantity(2, resource.DecimalSI),
						v1.ResourceMemory: resource.MustParse("1Gi"),
						notExistingNICResourceName: *resource.NewQuantity(0, resource.DecimalSI),}),
			nodeTopologies: &nodes,
			name: "Guaranteed QoS, pod fit",
			node: node1,
			wantStatus: nil, //topology_match has to skip request of 0 resources
		},
	}

	nodeInfo := framework.NewNodeInfo()
	for _, test := range topologyTests {
		t.Run(test.name, func(t *testing.T) {
			tm := NodeResourceTopologyMatch{}
			tm.nodeTopologies = nodes
			tm.topologyPolicyHandlers = make(PolicyHandlerMap)
	                tm.topologyPolicyHandlers[v1.SingleNUMANodeTopologyManagerPolicy] = tm
			nodeInfo.SetNode(&test.node)
			test.pod.Spec.Containers[0].Name = containerName
			test.pod.Spec.Containers[0].Resources.Limits = test.pod.Spec.Containers[0].Resources.Requests
			gotStatus := tm.Filter(context.Background(), framework.NewCycleState(), test.pod, nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}

}
