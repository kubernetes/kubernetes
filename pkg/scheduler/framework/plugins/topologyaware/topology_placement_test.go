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
	"fmt"
	"slices"
	"testing"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/util/sets"
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
	type scheduledPod struct {
		pathToRoot   []string
		assignedNode string
	}

	initialPlacementName := "test-placement"
	tests := map[string]struct {
		podGroupInfo          fwk.PodGroupInfo
		scheduledPodGroupPods []scheduledPod
		placementNodes        []*v1.Node
		otherNodes            []*v1.Node
		wantPlacementNodes    map[string][]string
		wantStatus            fwk.Code
	}{
		"without constraint returns placement matching all nodes": {
			podGroupInfo: makePodGroupInfoFromPG(&schedulingapi.PodGroup{}),
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
			podGroupInfo: makePodGroup("topology1"),
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
			podGroupInfo: makePodGroup("topology3"),
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
			podGroupInfo: makePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{assignedNode: "node2"},
				{assignedNode: "node3"},
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
			podGroupInfo: makePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{assignedNode: "node2"},
				{assignedNode: "node3"},
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
			podGroupInfo: makePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{assignedNode: "node2"},
				{assignedNode: "node3"},
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
			podGroupInfo: makePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{assignedNode: "node2"},
				{assignedNode: "node4"},
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
			podGroupInfo: makePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{assignedNode: "node2"},
			},
			placementNodes: []*v1.Node{
				st.MakeNode().Name("node1").Label("topology", "d2").Obj(),
			},
			otherNodes: []*v1.Node{
				st.MakeNode().Name("node2").Label("foo", "bar").Obj(),
			},
			wantStatus: fwk.Error,
		},
		"for cpg without constraint returns placement matching all nodes": {
			podGroupInfo: makePodGroupInfoFromCPG(&schedulingapi.CompositePodGroup{}),
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
		"for cpg with topology key constraint, returns placement for each topology domain": {
			podGroupInfo: makeCompositePodGroup("topology1"),
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
		"for cpg without matching topology label, returns empty": {
			podGroupInfo: makeCompositePodGroup("topology3"),
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
		"for cpg with pods already scheduled in a single domain, returns that domain": {
			podGroupInfo: makeCompositePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{
					pathToRoot:   []string{"pg1", "cpg1"},
					assignedNode: "node2",
				},
				{
					pathToRoot:   []string{"pg2", "cpg2"},
					assignedNode: "node3",
				},
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
		"for cpg with pods already scheduled in a single domain not present in current placement, returns empty": {
			podGroupInfo: makeCompositePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{
					pathToRoot:   []string{"pg1", "cpg1"},
					assignedNode: "node2",
				},
				{
					pathToRoot:   []string{"pg2", "cpg2"},
					assignedNode: "node3",
				},
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
		"for cpg with pods already scheduled in conflicting domains across separate pod groups, returns error": {
			podGroupInfo: makeCompositePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{
					pathToRoot:   []string{"pg1", "cpg1"},
					assignedNode: "node2",
				},
				{
					pathToRoot:   []string{"pg2", "cpg2"},
					assignedNode: "node3",
				},
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
		"for cpg with pods already scheduled in conflicting domains in a single pod group, returns error": {
			podGroupInfo: makeCompositePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{
					pathToRoot:   []string{"pg1", "cpg1"},
					assignedNode: "node2",
				},
				{
					pathToRoot:   []string{"pg1", "cpg1"},
					assignedNode: "node3",
				},
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
		"for cpg with already scheduled pod on node outside of snapshot, returns error": {
			podGroupInfo: makeCompositePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{
					pathToRoot:   []string{"pg1", "cpg1"},
					assignedNode: "node2",
				},
				{
					pathToRoot:   []string{"pg1", "cpg1"},
					assignedNode: "node4",
				},
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
		"for cpg with already scheduled pod on node without topology label, returns error": {
			podGroupInfo: makeCompositePodGroup("topology"),
			scheduledPodGroupPods: []scheduledPod{
				{
					pathToRoot:   []string{"pg1", "cpg1"},
					assignedNode: "node2",
				},
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

	for _, cpgEnabled := range []bool{false, true} {
		for name, tt := range tests {
			if !cpgEnabled && tt.podGroupInfo.GetCompositePodGroup() != nil {
				continue
			}
			t.Run(fmt.Sprintf("%s (cpg=%v)", name, cpgEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.GenericWorkload:                 true,
					features.TopologyAwareWorkloadScheduling: true,
					features.CompositePodGroup:               cpgEnabled,
				})
				_, tCtx := ktesting.NewTestContext(t)

				nodes := make([]v1.Node, 0, len(tt.placementNodes)+len(tt.otherNodes))
				for _, node := range append(tt.placementNodes, tt.otherNodes...) {
					nodes = append(nodes, *node)
				}

				pgs := []schedulingapi.PodGroup{}
				cpgs := []schedulingapi.CompositePodGroup{}
				pods := []*v1.Pod{}
				alreadyAdded := sets.New[string]()
				namespace := tt.podGroupInfo.GetNamespace()

				for i, scheduledPod := range tt.scheduledPodGroupPods {
					parent := tt.podGroupInfo.GetName()
					for j, entityName := range slices.Backward(scheduledPod.pathToRoot) {
						if !alreadyAdded.Has(entityName) {
							if j == 0 {
								pgs = append(pgs, *st.MakePodGroup().Name(entityName).Namespace(namespace).ParentCompositePodGroup(parent).Obj())
							} else {
								cpgs = append(cpgs, *st.MakeCompositePodGroup().Name(entityName).Namespace(namespace).ParentCompositePodGroup(parent).Obj())
							}
							alreadyAdded.Insert(entityName)
						}
						parent = entityName
					}
					name := fmt.Sprintf("pod%v", i)
					pods = append(pods, st.MakePod().Name(name).UID(name).Namespace(namespace).Node(scheduledPod.assignedNode).PodGroupName(parent).Obj())
				}

				if cpg := tt.podGroupInfo.GetCompositePodGroup(); cpg != nil {
					cpgs = append(cpgs, *cpg)
				} else {
					pgs = append(pgs, *tt.podGroupInfo.GetPodGroup())
				}

				cs := clientsetfake.NewClientset(
					&schedulingapi.PodGroupList{Items: pgs},
					&schedulingapi.CompositePodGroupList{Items: cpgs},
					&v1.NodeList{Items: nodes},
				)
				informerFactory := informers.NewSharedInformerFactory(cs, 0)
				_ = informerFactory.Scheduling().V1alpha3().PodGroups().Informer()
				_ = informerFactory.Scheduling().V1alpha3().CompositePodGroups().Informer()
				_ = informerFactory.Core().V1().Nodes().Informer()
				informerFactory.StartWithContext(tCtx)
				informerFactory.WaitForCacheSyncWithContext(tCtx)

				pgPtrs := make([]*schedulingapi.PodGroup, len(pgs))
				cpgPtrs := make([]*schedulingapi.CompositePodGroup, len(cpgs))
				nodePtrs := make([]*v1.Node, len(nodes))
				for i := range pgs {
					pgPtrs[i] = &pgs[i]
				}
				for i := range cpgs {
					cpgPtrs[i] = &cpgs[i]
				}
				for i := range nodes {
					nodePtrs[i] = &nodes[i]
				}

				snapshot := cache.NewTestSnapshotWithCompositePodGroups(pods, nodePtrs, pgPtrs, cpgPtrs)

				fh, _ := runtime.NewFramework(tCtx, nil, nil,
					runtime.WithInformerFactory(informerFactory),
					runtime.WithSnapshotSharedLister(snapshot),
				)

				pl, err := New(tCtx, nil, fh, feature.NewSchedulerFeaturesFromGates(utilfeature.DefaultFeatureGate))
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

				result, status := pl.GeneratePlacements(tCtx, framework.NewCycleState(), tt.podGroupInfo, placement)

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
}

func makePodGroupInfoFromPG(pg *schedulingapi.PodGroup) fwk.PodGroupInfo {
	return &framework.PodGroupInfo{
		Name:      pg.Name,
		Namespace: pg.Namespace,
		PodGroup:  pg,
		Type:      fwk.PodGroupKeyType,
	}
}

func makePodGroupInfoFromCPG(cpg *schedulingapi.CompositePodGroup) fwk.PodGroupInfo {
	return &framework.PodGroupInfo{
		Name:              cpg.Name,
		Namespace:         cpg.Namespace,
		CompositePodGroup: cpg,
		Type:              fwk.CompositePodGroupKeyType,
	}
}

func makePodGroup(topologyKey string) fwk.PodGroupInfo {
	return makePodGroupInfoFromPG(st.MakePodGroup().Name("root").Namespace("default").TopologyKey(topologyKey).Obj())
}

func makeCompositePodGroup(topologyKey string) fwk.PodGroupInfo {
	return makePodGroupInfoFromCPG(st.MakeCompositePodGroup().Name("root").Namespace("default").TopologyKey(topologyKey).Obj())
}
