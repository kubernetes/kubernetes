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
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
)

func TestAffinityDiscrepancy(t *testing.T) {
	plugin := &InterPodAffinity{
		nsLister:     &mockNamespaceLister{},
		parallelizer: parallelize.NewParallelizer(1),
		sharedLister: &mockSharedLister{
			nodeLister: &mockNodeInfoLister{
				nodes: []fwk.NodeInfo{},
			},
		},
	}

	// Incoming pod has TWO host-scoped affinity terms
	podWithTwoAffinityTerms := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "default"},
		Spec: v1.PodSpec{
			Affinity: &v1.Affinity{
				PodAffinity: &v1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
							TopologyKey:   v1.LabelHostname,
						},
						{
							LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"baz": "qux"}},
							TopologyKey:   v1.LabelHostname,
						},
					},
				},
			},
		},
	}

	// Node 1 has two DIFFERENT pods, each matching ONE term.
	// Kubernetes requires ONE pod to match ALL terms. So node-1 should NOT satisfy affinity.
	podA := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "podA", Namespace: "default", Labels: map[string]string{"foo": "bar"}}}
	podB := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "podB", Namespace: "default", Labels: map[string]string{"baz": "qux"}}}

	node1 := framework.NewNodeInfo(podA, podB)
	node1.SetNode(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{v1.LabelHostname: "node-1"},
		},
	})

	// Node 3 has a pod matching BOTH terms.
	podC := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "podC", Namespace: "default", Labels: map[string]string{"foo": "bar", "baz": "qux"}}}
	node3 := framework.NewNodeInfo(podC)
	node3.SetNode(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-3",
			Labels: map[string]string{v1.LabelHostname: "node-3"},
		},
	})

	plugin.sharedLister.(*mockSharedLister).nodeLister.nodes = []fwk.NodeInfo{node1, node3}

	ctx := context.Background()
	state := framework.NewCycleState()

	// PreFilter
	_, status := plugin.PreFilter(ctx, state, podWithTwoAffinityTerms, plugin.sharedLister.(*mockSharedLister).nodeLister.nodes)
	if !status.IsSuccess() && status.Code() != fwk.Skip {
		t.Fatalf("PreFilter failed: %v", status.Message())
	}

	// Internal check: hasMatchingHostScopedAffinityPodGlobally should be TRUE because of podC
	s, _ := getPreFilterState(state)
	if !s.hasMatchingHostScopedAffinityPodGlobally {
		t.Errorf("hasMatchingHostScopedAffinityPodGlobally is FALSE, expected TRUE (since podC matches BOTH terms)")
	}

	// Filter on node-1: should FAIL because no single pod matches BOTH terms
	status = plugin.Filter(ctx, state, podWithTwoAffinityTerms, node1)
	if status.IsSuccess() {
		t.Errorf("Expected scheduling on node-1 to FAIL (no single pod matches both terms), but it SUCCEEDED")
	}

	// Filter on node-3: should SUCCEED because podC matches BOTH terms
	status = plugin.Filter(ctx, state, podWithTwoAffinityTerms, node3)
	if !status.IsSuccess() {
		t.Errorf("Expected scheduling on node-3 to SUCCEED (podC matches both terms), but got %v", status.Message())
	}

	// Now try an EMPTY node Node 2.
	// hasMatchingHostScopedAffinityPodGlobally is TRUE, so it should NOT trigger the fallback.
	node2 := framework.NewNodeInfo()
	node2.SetNode(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-2",
			Labels: map[string]string{v1.LabelHostname: "node-2"},
		},
	})

	status = plugin.Filter(ctx, state, podWithTwoAffinityTerms, node2)
	if status.IsSuccess() {
		t.Errorf("Expected scheduling on node-2 to FAIL (matching pods exist elsewhere, node-2 doesn't have them), but it SUCCEEDED")
	}

	// Now test fallback: remove node-3 (the only matching pod).
	plugin.sharedLister.(*mockSharedLister).nodeLister.nodes = []fwk.NodeInfo{node1, node2}
	podWithTwoAffinityTerms.Labels = map[string]string{"foo": "bar", "baz": "qux"} // Pod matches its own terms

	state = framework.NewCycleState()
	plugin.PreFilter(ctx, state, podWithTwoAffinityTerms, plugin.sharedLister.(*mockSharedLister).nodeLister.nodes)
	s, _ = getPreFilterState(state)
	if s.hasMatchingHostScopedAffinityPodGlobally {
		t.Errorf("Expected hasMatchingHostScopedAffinityPodGlobally to be FALSE after removing node-3")
	}

	// Now node-2 should succeed due to fallback
	status = plugin.Filter(ctx, state, podWithTwoAffinityTerms, node2)
	if !status.IsSuccess() {
		t.Errorf("Expected scheduling on node-2 to SUCCEED due to fallback, but got %v", status.Message())
	}
}
