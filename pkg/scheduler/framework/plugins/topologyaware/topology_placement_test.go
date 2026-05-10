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

package topologyaware

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestGeneratePlacements(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	initialPlacementName := "test-placement"
	tests := map[string]struct {
		podGroup              *schedulingapi.PodGroup
		scheduledPodGroupPods map[string]string
		placementNodes        []*v1.Node
		otherNodes            []*v1.Node
		wantPlacementNodes    map[string][]string
		wantStatus            fwk.Code
	}{
		"without constraint returns placement matching all nodes": {
			podGroup: &schedulingapi.PodGroup{
				Spec: schedulingapi.PodGroupSpec{},
			},
			placementNodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Label("foo", "bar").Obj(),
			},
			otherNodes: []*v1.Node{
				st.MakeNode().Name("node3").Obj(),
			},
			wantPlacementNodes: map[string][]string{
				initialPlacementName: {"node1", "node2"},
			},
			wantStatus: fwk.Success,
		},
		"with topology key constraint, returns placement for each topology domain": {
			podGroup: makePodGroup("topology1"),
			placementNodes: []*v1.Node{
				st.MakeNode().Name("node0").Label("topology2", "d1").Obj(),
				st.MakeNode().Name("node1").Label("topology2", "d4").Obj(),
				st.MakeNode().Name("node2").Label("topology1", "d1").Obj(),
				st.MakeNode().Name("node3").Label("topology1", "d2").Obj(),
				st.MakeNode().Name("node4").Label("topology1", "d1").Obj(),
				st.MakeNode().Name("node5").Label("topology1", "d3").Obj(),
			},
			wantPlacementNodes: map[string][]string{
				"d1": {"node2", "node4"},
				"d2": {"node3"},
				"d3": {"node5"},
			},
			wantStatus: fwk.Success,
		},
		"without matching topology label, returns empty": {
			podGroup: makePodGroup("topology3"),
			placementNodes: []*v1.Node{
				st.MakeNode().Name("node0").Label("topology2", "d1").Obj(),
				st.MakeNode().Name("node1").Label("topology2", "d4").Obj(),
				st.MakeNode().Name("node2").Label("topology1", "d1").Obj(),
				st.MakeNode().Name("node3").Label("topology1", "d2").Obj(),
				st.MakeNode().Name("node4").Label("topology1", "d1").Obj(),
				st.MakeNode().Name("node5").Label("topology1", "d3").Obj(),
			},
			wantPlacementNodes: map[string][]string{},
			wantStatus:         fwk.Success,
		},
		"with pods already scheduled in a single domain, returns that domain": {
			podGroup: makePodGroup("topology"),
			scheduledPodGroupPods: map[string]string{
				"pod1": "node2",
				"pod2": "node3",
			},
			placementNodes: []*v1.Node{
				st.MakeNode().Name("node0").Label("topology", "d2").Obj(),
				st.MakeNode().Name("node1").Label("topology", "d1").Obj(),
			},
			otherNodes: []*v1.Node{
				st.MakeNode().Name("node2").Label("topology", "d1").Obj(),
				st.MakeNode().Name("node3").Label("topology", "d1").Obj(),
			},
			wantPlacementNodes: map[string][]string{
				"d1": {"node1"},
			},
			wantStatus: fwk.Success,
		},
		"with pods already scheduled in a single domain not present in current placement, returns empty": {
			podGroup: makePodGroup("topology"),
			scheduledPodGroupPods: map[string]string{
				"pod1": "node2",
				"pod2": "node3",
			},
			placementNodes: []*v1.Node{
				st.MakeNode().Name("node0").Label("topology", "d2").Obj(),
			},
			otherNodes: []*v1.Node{
				st.MakeNode().Name("node2").Label("topology", "d1").Obj(),
				st.MakeNode().Name("node3").Label("topology", "d1").Obj(),
			},
			wantPlacementNodes: map[string][]string{},
			wantStatus:         fwk.Success,
		},
		"with pods already scheduled in conflicting domains, returns error": {
			podGroup: makePodGroup("topology"),
			scheduledPodGroupPods: map[string]string{
				"pod1": "node2",
				"pod2": "node3",
			},
			placementNodes: []*v1.Node{
				st.MakeNode().Name("node0").Label("topology", "d2").Obj(),
				st.MakeNode().Name("node1").Label("topology", "d1").Obj(),
			},
			otherNodes: []*v1.Node{
				st.MakeNode().Name("node2").Label("topology", "d0").Obj(),
				st.MakeNode().Name("node3").Label("topology", "d1").Obj(),
			},
			wantStatus: fwk.Error,
		},
		"with already scheduled pod on node outside of snapshot, returns error": {
			podGroup: makePodGroup("topology"),
			scheduledPodGroupPods: map[string]string{
				"pod1": "node2",
				"pod2": "node4",
			},
			placementNodes: []*v1.Node{
				st.MakeNode().Name("node0").Label("topology", "d2").Obj(),
				st.MakeNode().Name("node1").Label("topology", "d1").Obj(),
			},
			otherNodes: []*v1.Node{
				st.MakeNode().Name("node2").Label("topology", "d1").Obj(),
				st.MakeNode().Name("node3").Label("topology", "d1").Obj(),
			},
			wantStatus: fwk.Error,
		},
		"with already scheduled pod on node without topology label, returns error": {
			podGroup: makePodGroup("topology"),
			scheduledPodGroupPods: map[string]string{
				"pod1": "node2",
			},
			placementNodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("topology", "d2").Obj(),
			},
			otherNodes: []*v1.Node{
				st.MakeNode().Name("node2").Label("foo", "bar").Obj(),
			},
			wantStatus: fwk.Error,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			_, tCtx := ktesting.NewTestContext(t)

			nodes := make([]v1.Node, 0, len(tt.placementNodes)+len(tt.otherNodes))
			for _, node := range append(tt.placementNodes, tt.otherNodes...) {
				nodes = append(nodes, *node)
			}

			cs := clientsetfake.NewClientset(
				&schedulingapi.PodGroupList{Items: []schedulingapi.PodGroup{*tt.podGroup}},
				&v1.NodeList{Items: nodes},
			)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			_ = informerFactory.Scheduling().V1alpha2().PodGroups().Informer()
			_ = informerFactory.Core().V1().Nodes().Informer()
			informerFactory.StartWithContext(tCtx)
			informerFactory.WaitForCacheSyncWithContext(tCtx)

			pods := make([]*v1.Pod, 0, len(tt.scheduledPodGroupPods)+1)
			pods = append(pods, st.MakePod().Name("unscheduled").UID("unscheduled").Namespace(tt.podGroup.Namespace).PodGroupName(tt.podGroup.Name).Obj())
			for podName, nodeName := range tt.scheduledPodGroupPods {
				pod := st.MakePod().Name(podName).UID(podName).Node(nodeName).Namespace(tt.podGroup.Namespace).PodGroupName(tt.podGroup.Name).Obj()
				pods = append(pods, pod)
			}
			snapshot := cache.NewSnapshot(pods, append(tt.placementNodes, tt.otherNodes...))

			fh, _ := runtime.NewFramework(tCtx, nil, nil,
				runtime.WithInformerFactory(informerFactory),
				runtime.WithSnapshotSharedLister(snapshot),
			)

			pl, err := New(tCtx, nil, fh, feature.Features{})
			if err != nil {
				t.Fatalf("failed when creating plugin: %v", err)
			}

			placement := &fwk.Placement{
				Name:  initialPlacementName,
				Nodes: make([]fwk.NodeInfo, len(tt.placementNodes)),
			}
			for i, node := range tt.placementNodes {
				ni := framework.NewNodeInfo()
				ni.SetNode(node)
				placement.Nodes[i] = ni
			}
			podGroupInfo := &framework.PodGroupInfo{
				Name:      tt.podGroup.Name,
				Namespace: tt.podGroup.Namespace,
			}

			result, status := pl.GeneratePlacements(tCtx, framework.NewCycleState(), podGroupInfo, placement)

			if status.Code() != tt.wantStatus {
				t.Fatalf("expected status %v, got %v", tt.wantStatus, status.AsError())
			}

			if status.IsSuccess() {
				gotPlacementNodes := make(map[string][]string)
				for _, placement := range result.Placements {
					gotPlacementNodes[placement.Name] = make([]string, len(placement.Nodes))
					for i, node := range placement.Nodes {
						gotPlacementNodes[placement.Name][i] = node.Node().Name
					}
				}

				if diff := cmp.Diff(tt.wantPlacementNodes, gotPlacementNodes, cmpopts.EquateEmpty()); diff != "" {
					t.Errorf("Unexpected placements (-want,+got):\n%s", diff)
				}
			}
		})
	}
}

func makePodGroup(topologyKey string) *schedulingapi.PodGroup {
	return &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pg1",
			Namespace: "default",
		},
		Spec: schedulingapi.PodGroupSpec{
			SchedulingConstraints: &schedulingapi.PodGroupSchedulingConstraints{
				Topology: []schedulingapi.TopologyConstraint{
					{
						Key: topologyKey,
					},
				},
			},
		},
	}
}
