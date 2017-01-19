package schedulercache

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"reflect"
	"testing"
)

var (
	taints = []v1.Taint{
		v1.Taint{
			Key:   "foo",
			Value: "bar",
		},
	}
	testresource = Resource{1, 1, 1, map[v1.ResourceName]int64{"testresource": 1}}
	affnity      = v1.Affinity{
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
	nodelabels = map[string]string{
		"foo": "bar",
	}
)

func makenodeinfo(node *v1.Node, nodetaints []v1.Taint) NodeInfo {
	return NodeInfo{
		node:                    node,
		allowedPodNumber:        0,
		taints:                  nodetaints,
		memoryPressureCondition: v1.ConditionTrue,
		diskPressureCondition:   v1.ConditionTrue,
		requestedResource:       &Resource{},
		nonzeroRequest:          &Resource{},
		allocatableResource:     &Resource{},
	}
}
func makepod(podname, nodename string, podaffnity *v1.Affinity) *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      podname,
			Namespace: "default",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: v1.PodSpec{
			NodeName: nodename,
			Affinity: podaffnity,
			Containers: []v1.Container{
				v1.Container{
					Name:  "testname",
					Image: "testimage",
				},
			},
		},
	}
}

func makenode(nodename string, nodelables map[string]string) *v1.Node {
	return &v1.Node{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Node",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      nodename,
			Namespace: "",
			Labels:    nodelables,
		},
		Spec: v1.NodeSpec{
			ExternalID: "test",
		},
		Status: v1.NodeStatus{},
	}
}
func TestNode(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	if !reflect.DeepEqual(nodeinfo.Node(), node) {
		t.Fatalf("get nodeinfo node error,expected: %v,got: %v", node, nodeinfo.Node())
	}
}

func TestPod(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	pod := makepod("foo", "", &v1.Affinity{})
	pods := []*v1.Pod{pod}
	nodeinfo.pods = pods
	if !reflect.DeepEqual(nodeinfo.Pods(), pods) {
		t.Fatalf("get nodeinfo pods error,expected: %v,got: %v", pods, nodeinfo.Pods())
	}
}

func TestPodsWithAffinity(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	podwithoutaffinity := makepod("foo", "", &v1.Affinity{})
	podwithaffinity := makepod("foo", "", &affnity)
	nodeinfo.addPod(podwithaffinity)
	nodeinfo.addPod(podwithoutaffinity)
	if !reflect.DeepEqual(nodeinfo.PodsWithAffinity(), []*v1.Pod{podwithaffinity}) {
		t.Fatalf("get nodeinfo node error,expected: %v,got: %v", []*v1.Pod{podwithaffinity}, nodeinfo.PodsWithAffinity())
	}
}

func TestAllowedPodNumber(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	if !reflect.DeepEqual(nodeinfo.AllowedPodNumber(), nodeinfo.allowedPodNumber) {
		t.Fatalf("get nodeinfo allowedpodnumber error,expected: %v,got: %v", nodeinfo.allowedPodNumber, nodeinfo.AllowedPodNumber())
	}
}

func TestTaints(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	gottaints, err := nodeinfo.Taints()
	if err != nil {
		t.Fatalf("get nodeinfo taints error: %s", err)
	}
	if !reflect.DeepEqual(gottaints, taints) {
		t.Fatalf("get nodeinfo taints error,expected: %v,got: %v", taints, gottaints)
	}
}

func TestMemoryPressureCondition(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	nodeinfo.memoryPressureCondition = v1.ConditionTrue
	if !reflect.DeepEqual(nodeinfo.MemoryPressureCondition(), nodeinfo.memoryPressureCondition) {
		t.Fatalf("get nodeinfo memoryPressureCondition error,expected: %v,got: %v", nodeinfo.memoryPressureCondition, nodeinfo.MemoryPressureCondition())
	}
}

func TestDiskPressureCondition(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	nodeinfo.diskPressureCondition = v1.ConditionTrue
	if !reflect.DeepEqual(nodeinfo.DiskPressureCondition(), nodeinfo.diskPressureCondition) {
		t.Fatalf("get nodeinfo diskPressureCondition error,expected: %v,got: %v", nodeinfo.diskPressureCondition, nodeinfo.DiskPressureCondition())
	}
}

func TestNonZeroRequest(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	nodeinfo.nonzeroRequest = &testresource
	if !reflect.DeepEqual(nodeinfo.NonZeroRequest(), *nodeinfo.nonzeroRequest) {
		t.Fatalf("get nodeinfo diskPressureCondition error,expected: %v,got: %v", *nodeinfo.nonzeroRequest, nodeinfo.NonZeroRequest())
	}
}
func TestAllocatableResource(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	nodeinfo.allocatableResource = &testresource
	if !reflect.DeepEqual(nodeinfo.AllocatableResource(), *nodeinfo.allocatableResource) {
		t.Fatalf("get nodeinfo allocatableResource error,expected: %v,got: %v", *nodeinfo.allocatableResource, nodeinfo.AllocatableResource())
	}
}

func TestClone(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	niclone := nodeinfo.Clone()
	if !reflect.DeepEqual(*niclone, nodeinfo) {
		t.Fatalf("nodeinfo clone error,expected: %v,got: %v", nodeinfo, *niclone)
	}
}

func TesthasPodAffinityConstraints(t *testing.T) {
	podwithaffinity := makepod("foo", "", &affnity)
	podwithoutaffinity := makepod("foo", "", &v1.Affinity{})
	if !reflect.DeepEqual(hasPodAffinityConstraints(podwithaffinity), true) {
		t.Fatalf("nodeinfo check pod has podaffinityconstraints error,expected: %v,got: %v", true, hasPodAffinityConstraints(podwithaffinity))
	}
	if !reflect.DeepEqual(hasPodAffinityConstraints(podwithoutaffinity), false) {
		t.Fatalf("nodeinfo check pod has no podaffinityconstraints error,expected: %v,got: %v", false, hasPodAffinityConstraints(podwithoutaffinity))
	}
}

func TestaddPod(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	podwithaffinity := makepod("foo", "", &affnity)
	podwithoutaffinity := makepod("foo", "", &v1.Affinity{})
	nodeinfo.addPod(podwithaffinity)
	nodeinfo.addPod(podwithoutaffinity)
	if !reflect.DeepEqual(nodeinfo.podsWithAffinity, []*v1.Pod{podwithaffinity}) {
		t.Fatalf("nodeinfo add pod with affinity error,expected: %v,got: %v", []*v1.Pod{podwithaffinity}, nodeinfo.podsWithAffinity)
	}
	if !reflect.DeepEqual(nodeinfo.pods, []*v1.Pod{podwithoutaffinity}) {
		t.Fatalf("nodeinfo add pod without affinity error,expected: %v,got: %v", []*v1.Pod{podwithoutaffinity}, nodeinfo.pods)
	}

}

func TestremovePod(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	podwithaffinity := makepod("foo", "", &affnity)
	podwithoutaffinity := makepod("foo", "", &v1.Affinity{})
	nodeinfo.addPod(podwithaffinity)
	nodeinfo.addPod(podwithoutaffinity)
	err := nodeinfo.removePod(podwithaffinity)
	if err != nil {
		t.Fatalf("nodeinfo remove pod with affinity  error:%s", err)
	}
	if !reflect.DeepEqual(nodeinfo.podsWithAffinity, []&v1.Pod{}) {
		t.Fatalf("nodeinfo remove pod with affinity error,expected: %v,got: %v", []&v1.Pod{}, nodeinfo.podsWithAffinity)
	}
	err = nodeinfo.removePod(podwithoutaffinity)
	if err != nil {
		t.Fatalf("nodeinfo remove pod without affinity error:%s", err)
	}
	if !reflect.DeepEqual(nodeinfo.pods, []&v1.Pod{}) {
		t.Fatalf("nodeinfo remove pod without affinity error,expected: %v,got: %v", []&v1.Pod{}, nodeinfo.pods)
	}
}

func TestSetNode(t *testing.T) {
	node := makenode("test", nodelabels)
	nodeinfo := makenodeinfo(node, taints)
	overnode := makenode("test1", map[string]string{})
	err := nodeinfo.SetNode(overnode)
	if err != nil {
		t.Fatalf("set nodeinfo node error: %s", err)
	}
	if !reflect.DeepEqual(overnode, nodeinfo.Node()) {
		t.Fatalf("set nodeinfo node error,expected: %v,got: %v", overnode, nodeinfo.Node())
	}
}
