/*
Copyright 2017 The Kubernetes Authors.

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

package schedulercache

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
)

func makeNodeInfo(node *v1.Node, nodeTaints []v1.Taint) *NodeInfo {
	return &NodeInfo{
		node:                node,
		allowedPodNumber:    0,
		taints:              nodeTaints,
		requestedResource:   &Resource{},
		nonzeroRequest:      &Resource{},
		allocatableResource: &Resource{},
	}
}

func makePod(podName, nodeName string, podAffinity *v1.Affinity) *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: "default",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: v1.PodSpec{
			NodeName: nodeName,
			Affinity: podAffinity,
			Containers: []v1.Container{
				v1.Container{
					Name:  "testname",
					Image: "testimage",
				},
			},
		},
	}
}

func makeNode(nodeName string, nodeLables map[string]string) *v1.Node {
	return &v1.Node{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Node",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      nodeName,
			Namespace: "default",
			Labels:    nodeLables,
		},
		Spec: v1.NodeSpec{
			ExternalID: "test",
		},
		Status: v1.NodeStatus{},
	}
}
func TestNewNodeInfo(t *testing.T) {
	affinity := v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
				{
					Weight: 5,
					PodAffinityTerm: v1.PodAffinityTerm{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "security",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"S1"},
								},
							},
						},
						TopologyKey: "region",
					},
				},
			},
		},
	}
	podWithAffinity := makePod("foo", "", &affinity)
	podWithoutAffinity := makePod("bar", "", &v1.Affinity{})
	newNodeInfo := NewNodeInfo([]*v1.Pod{podWithAffinity, podWithoutAffinity}...)

	nodeInfo := NodeInfo{
		allowedPodNumber:    0,
		requestedResource:   &Resource{},
		allocatableResource: &Resource{},

		pods:             []*v1.Pod{podWithAffinity, podWithoutAffinity},
		podsWithAffinity: []*v1.Pod{podWithAffinity},
		generation:       2,
		nonzeroRequest:   &Resource{MilliCPU: 2 * priorityutil.DefaultMilliCpuRequest, Memory: 2 * priorityutil.DefaultMemoryRequest},
	}
	if !reflect.DeepEqual(*newNodeInfo, nodeInfo) {
		t.Fatalf("newnodeinfo error,expected: %v,got: %v", nodeInfo, *newNodeInfo)
	}
}
func TestNode(t *testing.T) {
	node := makeNode("testnode", map[string]string{})
	nodeInfo := makeNodeInfo(node, []v1.Taint{})
	if !reflect.DeepEqual(nodeInfo.Node(), node) {
		t.Fatalf("get nodeinfo node error,expected: %v,got: %v", node, nodeInfo.Node())
	}
}

func TestPodsWithAffinity(t *testing.T) {
	var nodeInfo *NodeInfo
	affinity := v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
				{
					Weight: 5,
					PodAffinityTerm: v1.PodAffinityTerm{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "security",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"S1"},
								},
							},
						},
						TopologyKey: "region",
					},
				},
			},
		},
	}
	if !reflect.DeepEqual(nodeInfo.PodsWithAffinity(), []*v1.Pod(nil)) {
		t.Fatalf("get nodeinfo node error,expected: %#v,got: %#v", []*v1.Pod(nil), nodeInfo.PodsWithAffinity())
	}

	node := makeNode("testnode", map[string]string{})
	nodeInfo = makeNodeInfo(node, []v1.Taint{})
	podWithoutAffinity := makePod("foo", "", &v1.Affinity{})
	podWithAffinity := makePod("foo", "", &affinity)
	nodeInfo.addPod(podWithAffinity)
	nodeInfo.addPod(podWithoutAffinity)
	if !reflect.DeepEqual(nodeInfo.PodsWithAffinity(), []*v1.Pod{podWithAffinity}) {
		t.Fatalf("get nodeinfo node error,expected: %v,got: %v", []*v1.Pod{podWithAffinity}, nodeInfo.PodsWithAffinity())
	}
}
func TestAllowedPodNumber(t *testing.T) {
	var nodeInfo *NodeInfo
	if !reflect.DeepEqual(nodeInfo.AllowedPodNumber(), 0) {
		t.Fatalf("get nodeinfo allowedpodnumber error,expected: %v,got: %v", 0, nodeInfo.AllowedPodNumber())
	}

	node := makeNode("testnode", map[string]string{})
	nodeInfo = makeNodeInfo(node, []v1.Taint{})
	if !reflect.DeepEqual(nodeInfo.AllowedPodNumber(), nodeInfo.allowedPodNumber) {
		t.Fatalf("get nodeinfo allowedpodnumber error,expected: %v,got: %v", nodeInfo.allowedPodNumber, nodeInfo.AllowedPodNumber())
	}
}

func TestTaints(t *testing.T) {
	var nodeInfo *NodeInfo
	taints := []v1.Taint{
		v1.Taint{
			Key:   "foo",
			Value: "bar",
		},
	}
	if gotTaints, err := nodeInfo.Taints(); err != nil {
		t.Fatalf("get nodeinfo taints error: %s", err)
	} else if !reflect.DeepEqual(gotTaints, []v1.Taint(nil)) {
		t.Fatalf("get nodeinfo taints error,expected: %#v,got: %#v", []v1.Taint(nil), gotTaints)
	}

	node := makeNode("test", map[string]string{})
	nodeInfo = makeNodeInfo(node, taints)
	if gotTaints, err := nodeInfo.Taints(); err != nil {
		t.Fatalf("get nodeinfo taints error: %s", err)
	} else if !reflect.DeepEqual(gotTaints, taints) {
		t.Fatalf("get nodeinfo taints error,expected: %#v,got: %#v", taints, gotTaints)
	}
}

func TestPressureCondition(t *testing.T) {
	var nodeInfo1, nodeInfo2 *NodeInfo
	node := makeNode("testnode", map[string]string{})
	nodeInfo1 = makeNodeInfo(node, []v1.Taint{})
	nodeInfo1.memoryPressureCondition = v1.ConditionTrue
	nodeInfo1.diskPressureCondition = v1.ConditionTrue

	for nodeInfo, memorypressurecondition := range map[*NodeInfo]v1.ConditionStatus{nodeInfo1: v1.ConditionTrue, nodeInfo2: v1.ConditionUnknown} {
		if !reflect.DeepEqual(nodeInfo.MemoryPressureCondition(), memorypressurecondition) {
			t.Fatalf("get nodeinfo memoryPressureCondition error,expected: %v,got: %v", memorypressurecondition, nodeInfo.MemoryPressureCondition())
		}
	}

	for nodeInfo, diskpressurecondition := range map[*NodeInfo]v1.ConditionStatus{nodeInfo1: v1.ConditionTrue, nodeInfo2: v1.ConditionUnknown} {
		if !reflect.DeepEqual(nodeInfo.DiskPressureCondition(), diskpressurecondition) {
			t.Fatalf("get nodeinfo diskPressureCondition error,expected: %v,got: %v", diskpressurecondition, nodeInfo.DiskPressureCondition())
		}
	}
}

func TestResources(t *testing.T) {
	var nodeInfo *NodeInfo
	testResource := Resource{1, 1, 1, map[v1.ResourceName]int64{"testresource": 1}}
	node := makeNode("testnode", map[string]string{})

	if !reflect.DeepEqual(nodeInfo.RequestedResource(), Resource{}) {
		t.Fatalf("get nodeinfo requestedResource error,expected: %v,got: %v", Resource{}, nodeInfo.RequestedResource())
	}
	if !reflect.DeepEqual(nodeInfo.NonZeroRequest(), Resource{}) {
		t.Fatalf("get nodeinfo nonzerorequest error,expected: %v,got: %v", Resource{}, nodeInfo.NonZeroRequest())
	}
	if !reflect.DeepEqual(nodeInfo.AllocatableResource(), Resource{}) {
		t.Fatalf("get nodeinfo allocatableResource error,expected: %v,got: %v", Resource{}, nodeInfo.AllocatableResource())
	}

	nodeInfo = makeNodeInfo(node, []v1.Taint{})
	nodeInfo.requestedResource = &testResource
	nodeInfo.nonzeroRequest = &testResource
	nodeInfo.allocatableResource = &testResource
	if !reflect.DeepEqual(nodeInfo.RequestedResource(), *nodeInfo.requestedResource) {
		t.Fatalf("get nodeinfo requestedResource error,expected: %v,got: %v", nodeInfo.requestedResource, nodeInfo.RequestedResource())
	}
	if !reflect.DeepEqual(nodeInfo.NonZeroRequest(), *nodeInfo.nonzeroRequest) {
		t.Fatalf("get nodeinfo nonzerorequest error,expected: %v,got: %v", *nodeInfo.nonzeroRequest, nodeInfo.NonZeroRequest())
	}
	if !reflect.DeepEqual(nodeInfo.AllocatableResource(), *nodeInfo.allocatableResource) {
		t.Fatalf("get nodeinfo allocatableResource error,expected: %v,got: %v", *nodeInfo.allocatableResource, nodeInfo.AllocatableResource())
	}

	resourceList := testResource.ResourceList()
	rsl := v1.ResourceList{
		v1.ResourceCPU:       *resource.NewMilliQuantity(testResource.MilliCPU, resource.DecimalSI),
		v1.ResourceMemory:    *resource.NewQuantity(testResource.Memory, resource.BinarySI),
		v1.ResourceNvidiaGPU: *resource.NewQuantity(testResource.NvidiaGPU, resource.DecimalSI),
		"testresource":       *resource.NewQuantity(1, resource.DecimalSI),
	}
	if !reflect.DeepEqual(resourceList, rsl) {
		t.Fatalf("get resourcelist error,expected: %v,got: %v", rsl)
	}
	testResource.AddOpaque("testresource", 1)
	if trs := testResource.OpaqueIntResources["testresource"]; trs != 2 {
		t.Fatalf("add opaque error,expected: %d,got: %d", 2, trs)
	}
}

func TestClone(t *testing.T) {
	nodeLabels := map[string]string{
		"foo": "bar",
	}
	taints := []v1.Taint{
		v1.Taint{
			Key:   "foo",
			Value: "bar",
		},
	}
	affinity := v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
				{
					Weight: 5,
					PodAffinityTerm: v1.PodAffinityTerm{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "security",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"S1"},
								},
							},
						},
						TopologyKey: "region",
					},
				},
			},
		},
	}
	node := makeNode("testnode", nodeLabels)
	nodeInfo := makeNodeInfo(node, taints)
	podWithAffinity := makePod("foopod", "", &affinity)
	nodeInfo.addPod(podWithAffinity)
	nodeInfoClone := nodeInfo.Clone()
	if !reflect.DeepEqual(nodeInfoClone, nodeInfo) {
		t.Fatalf("nodeinfo clone error,expected: %v,got: %v", nodeInfo, nodeInfoClone)
	}
}

func TestNodeinfoOfPod(t *testing.T) {
	nodeLabels := map[string]string{
		"foo": "bar",
	}
	taints := []v1.Taint{
		v1.Taint{
			Key:   "foo",
			Value: "bar",
		},
	}
	affinity := v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
				{
					Weight: 5,
					PodAffinityTerm: v1.PodAffinityTerm{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "security",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"S1"},
								},
							},
						},
						TopologyKey: "region",
					},
				},
			},
		},
	}
	node := makeNode("testnode", nodeLabels)
	nodeInfo := makeNodeInfo(node, taints)
	podWithAffinity := makePod("podwithaffinity", "", &affinity)
	nodeInfo.addPod(podWithAffinity)
	err := nodeInfo.removePod(podWithAffinity)
	if err != nil {
		t.Fatalf("nodeinfo remove pod with affinity  error:%s", err)
	}
	if !reflect.DeepEqual(nodeInfo.podsWithAffinity, []*v1.Pod{}) {
		t.Fatalf("nodeinfo remove pod with affinity error,expected: %v,got: %v", []*v1.Pod{}, nodeInfo.podsWithAffinity)
	}
}

func TestRemoveNode(t *testing.T) {
	taints := []v1.Taint{
		v1.Taint{
			Key:   "foo",
			Value: "bar",
		},
	}
	nodeInfo := makeNodeInfo(&v1.Node{}, taints)
	node := makeNode("testnode", map[string]string{})
	nodeInfo.SetNode(node)
	pod := makePod("testpod", "", &v1.Affinity{})
	nodeInfo.addPod(pod)
	nodeInfo.RemoveNode(node)
	if reflect.DeepEqual(nodeInfo, NodeInfo{generation: 1}) {
		t.Errorf("nodeinfo remove node err: %v,got: %v", NodeInfo{generation: 1}, nodeInfo)
	}
}
