/*
Copyright The Kubernetes Authors.

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

package interpodaffinity

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

func init() {
	metrics.Register()
}

// mockNamespaceLister provides an empty list of namespaces
type mockNamespaceLister struct {
	getFn func(string) (*v1.Namespace, error)
}

func (m *mockNamespaceLister) List(selector labels.Selector) ([]*v1.Namespace, error) {
	return nil, nil
}
func (m *mockNamespaceLister) Get(name string) (*v1.Namespace, error) {
	if m.getFn != nil {
		return m.getFn(name)
	}
	return &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}, nil
}

// mockNodeInfoLister provides a simple static mock for the non-host-scoped queries
type mockNodeInfoLister struct {
	nodes []fwk.NodeInfo
}

func (m *mockNodeInfoLister) List() ([]fwk.NodeInfo, error) {
	return m.nodes, nil
}
func (m *mockNodeInfoLister) HavePodsWithAffinityList() ([]fwk.NodeInfo, error) {
	return m.nodes, nil
}
func (m *mockNodeInfoLister) HavePodsWithRequiredAntiAffinityList() ([]fwk.NodeInfo, error) {
	return m.nodes, nil
}
func (m *mockNodeInfoLister) HavePodsWithRequiredNonHostScopedAntiAffinityList() ([]fwk.NodeInfo, error) {
	return m.nodes, nil
}
func (m *mockNodeInfoLister) Get(nodeName string) (fwk.NodeInfo, error) {
	for _, n := range m.nodes {
		if n.Node().Name == nodeName {
			return n, nil
		}
	}
	return nil, nil
}

type mockSharedLister struct {
	nodeLister *mockNodeInfoLister
}

func (m *mockSharedLister) NodeInfos() fwk.NodeInfoLister           { return m.nodeLister }
func (m *mockSharedLister) StorageInfos() fwk.StorageInfoLister     { return nil }
func (m *mockSharedLister) PodGroupStates() fwk.PodGroupStateLister { return nil }

func TestFastTrackInterPodAffinity(t *testing.T) {
	plugin := &InterPodAffinity{
		nsLister:     &mockNamespaceLister{},
		parallelizer: parallelize.NewParallelizer(1),
		sharedLister: &mockSharedLister{
			nodeLister: &mockNodeInfoLister{
				nodes: []fwk.NodeInfo{},
			},
		},
	}

	podWithZoneAntiAffinity := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "default", Labels: map[string]string{"app": "foo"}},
		Spec: v1.PodSpec{
			Affinity: &v1.Affinity{
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
							TopologyKey:   "topology.kubernetes.io/zone",
						},
					},
				},
			},
		},
	}

	podWithHostAntiAffinity := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-2", Namespace: "default", Labels: map[string]string{"app": "bar"}},
		Spec: v1.PodSpec{
			Affinity: &v1.Affinity{
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "bar"}},
							TopologyKey:   v1.LabelHostname,
						},
					},
				},
			},
		},
	}

	podWithNamespaceAntiAffinity := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-3", Namespace: "other", Labels: map[string]string{"app": "baz"}},
		Spec: v1.PodSpec{
			Affinity: &v1.Affinity{
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "blocked"}},
							TopologyKey:   "topology.kubernetes.io/zone",
							NamespaceSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{"team": "red"},
							},
						},
					},
				},
			},
		},
	}

	node1 := framework.NewNodeInfo(podWithZoneAntiAffinity, podWithHostAntiAffinity, podWithNamespaceAntiAffinity)
	node1.SetNode(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
			Labels: map[string]string{
				"topology.kubernetes.io/zone": "us-west-1a",
				v1.LabelHostname:              "node-1",
			},
		},
	})

	node2 := framework.NewNodeInfo()
	node2.SetNode(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-2",
			Labels: map[string]string{
				"topology.kubernetes.io/zone": "us-west-1a",
				v1.LabelHostname:              "node-2",
			},
		},
	})

	node3 := framework.NewNodeInfo()
	node3.SetNode(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-3",
			Labels: map[string]string{
				"topology.kubernetes.io/zone": "us-west-1b",
				v1.LabelHostname:              "node-3",
			},
		},
	})

	plugin.sharedLister.(*mockSharedLister).nodeLister.nodes = []fwk.NodeInfo{node1, node2, node3}

	podWithNoRulesFoo := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-no-rules-foo", Namespace: "default", Labels: map[string]string{"app": "foo"}},
		Spec:       v1.PodSpec{},
	}

	podWithNoRulesBar := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-no-rules-bar", Namespace: "default", Labels: map[string]string{"app": "bar"}},
		Spec:       v1.PodSpec{},
	}

	podWithBlockedAppRedTeam := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-red", Namespace: "red-ns", Labels: map[string]string{"app": "blocked"}},
		Spec:       v1.PodSpec{},
	}

	podWithBlockedAppBlueTeam := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-blue", Namespace: "blue-ns", Labels: map[string]string{"app": "blocked"}},
		Spec:       v1.PodSpec{},
	}

	// Mock Namespace Labels
	plugin.nsLister.(*mockNamespaceLister).getFn = func(name string) (*v1.Namespace, error) {
		labels := map[string]string{}
		switch name {
		case "red-ns":
			labels["team"] = "red"
		case "blue-ns":
			labels["team"] = "blue"
		}
		return &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: labels}}, nil
	}

	for _, tc := range []struct {
		name         string
		incomingPod  *v1.Pod
		targetNode   fwk.NodeInfo
		wantSchedule bool
	}{
		{
			name:         "Zone-Scoped Anti-Affinity blocks scheduling in same zone (node-2)",
			incomingPod:  podWithZoneAntiAffinity, // Wants no "app: foo" in same zone
			targetNode:   node2,                   // node-2 is in us-west-1a (same as node-1 where pod-1 runs)
			wantSchedule: false,
		},
		{
			name:         "Zone-Scoped Anti-Affinity allows scheduling in different zone (node-3)",
			incomingPod:  podWithZoneAntiAffinity,
			targetNode:   node3, // node-3 is in us-west-1b
			wantSchedule: true,
		},
		{
			name:         "Host-Scoped Anti-Affinity blocks scheduling on same node (node-1)",
			incomingPod:  podWithHostAntiAffinity, // Wants no "app: bar" on same node
			targetNode:   node1,                   // node-1 has pod-2 with "app: bar"
			wantSchedule: false,
		},
		{
			name:         "Host-Scoped Anti-Affinity allows scheduling on different node in same zone (node-2)",
			incomingPod:  podWithHostAntiAffinity,
			targetNode:   node2,
			wantSchedule: true,
		},
		{
			name:         "Incoming pod with NO rules is blocked by existing pod's Zone-Scoped Anti-Affinity (node-2)",
			incomingPod:  podWithNoRulesFoo, // Has "app: foo", matches pod-1's anti-affinity on node-1
			targetNode:   node2,             // node-2 is in us-west-1a (same as node-1)
			wantSchedule: false,
		},
		{
			name:         "Incoming pod with NO rules is blocked by existing pod's Host-Scoped Anti-Affinity (node-1)",
			incomingPod:  podWithNoRulesBar, // Has "app: bar", matches pod-2's anti-affinity on node-1
			targetNode:   node1,
			wantSchedule: false,
		},
		{
			name:         "Incoming pod with NO rules is allowed on node with no conflicts (node-3)",
			incomingPod:  podWithNoRulesBar,
			targetNode:   node3,
			wantSchedule: true,
		},
		{
			name:         "Incoming pod is blocked by existing pod's namespaceSelector anti-affinity (matching namespace)",
			incomingPod:  podWithBlockedAppRedTeam,
			targetNode:   node2, // us-west-1a, same as node-1 where pod-3 is
			wantSchedule: false,
		},
		{
			name:         "Incoming pod is NOT blocked by existing pod's namespaceSelector anti-affinity (non-matching namespace)",
			incomingPod:  podWithBlockedAppBlueTeam,
			targetNode:   node2, // us-west-1a, same as node-1 where pod-3 is
			wantSchedule: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			state := framework.NewCycleState()

			// PreFilter
			_, status := plugin.PreFilter(ctx, state, tc.incomingPod, plugin.sharedLister.(*mockSharedLister).nodeLister.nodes)
			if !status.IsSuccess() && status.Code() != fwk.Skip {
				t.Fatalf("PreFilter failed: %v", status.Message())
			}

			// Filter
			status = plugin.Filter(ctx, state, tc.incomingPod, tc.targetNode)
			canSchedule := status.IsSuccess()
			if canSchedule != tc.wantSchedule {
				t.Errorf("Expected scheduling to be %v, got %v (status: %v)", tc.wantSchedule, canSchedule, status)
			}
		})
	}
}
