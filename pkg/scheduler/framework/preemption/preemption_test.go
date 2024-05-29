/*
Copyright 2021 The Kubernetes Authors.

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

package preemption

import (
	"context"
	"fmt"
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeunschedulable"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumezone"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

var (
	midPriority, highPriority = int32(100), int32(1000)

	veryLargeRes = map[v1.ResourceName]string{
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
)

type FakePostFilterPlugin struct {
	numViolatingVictim int
}

func (pl *FakePostFilterPlugin) SelectVictimsOnNode(
	ctx context.Context, state *framework.CycleState, pod *v1.Pod,
	nodeInfo *framework.NodeInfo, pdbs []*policy.PodDisruptionBudget) (victims []*v1.Pod, numViolatingVictim int, status *framework.Status) {
	return append(victims, nodeInfo.Pods[0].Pod), pl.numViolatingVictim, nil
}

func (pl *FakePostFilterPlugin) GetOffsetAndNumCandidates(nodes int32) (int32, int32) {
	return 0, nodes
}

func (pl *FakePostFilterPlugin) CandidatesToVictimsMap(candidates []Candidate) map[string]*extenderv1.Victims {
	return nil
}

func (pl *FakePostFilterPlugin) PodEligibleToPreemptOthers(pod *v1.Pod, nominatedNodeStatus *framework.Status) (bool, string) {
	return true, ""
}

func TestNodesWherePreemptionMightHelp(t *testing.T) {
	// Prepare 4 nodes names.
	nodeNames := []string{"node1", "node2", "node3", "node4"}
	tests := []struct {
		name          string
		nodesStatuses framework.NodeToStatusMap
		expected      sets.String // set of expected node names.
	}{
		{
			name: "No node should be attempted",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodeaffinity.ErrReasonPod),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
				"node3": framework.NewStatus(framework.UnschedulableAndUnresolvable, tainttoleration.ErrReasonNotMatch),
				"node4": framework.NewStatus(framework.UnschedulableAndUnresolvable, interpodaffinity.ErrReasonAffinityRulesNotMatch),
			},
			expected: sets.NewString(),
		},
		{
			name: "ErrReasonAntiAffinityRulesNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod anti-affinity",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable, interpodaffinity.ErrReasonAntiAffinityRulesNotMatch),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
				"node3": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodeunschedulable.ErrReasonUnschedulable),
				"node4": framework.NewStatus(framework.Unschedulable, "Unschedulable"),
			},
			expected: sets.NewString("node1", "node4"),
		},
		{
			name: "ErrReasonAffinityRulesNotMatch should not be tried as it indicates that the pod is unschedulable due to inter-pod affinity, but ErrReasonAntiAffinityRulesNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod anti-affinity",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, interpodaffinity.ErrReasonAffinityRulesNotMatch),
				"node2": framework.NewStatus(framework.Unschedulable, interpodaffinity.ErrReasonAntiAffinityRulesNotMatch),
				"node3": framework.NewStatus(framework.Unschedulable, "Unschedulable"),
				"node4": framework.NewStatus(framework.Unschedulable, "Unschedulable"),
			},
			expected: sets.NewString("node2", "node3", "node4"),
		},
		{
			name: "Mix of failed predicates works fine",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, volumerestrictions.ErrReasonDiskConflict),
				"node2": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("Insufficient %v", v1.ResourceMemory)),
				"node3": framework.NewStatus(framework.Unschedulable, "Unschedulable"),
				"node4": framework.NewStatus(framework.Unschedulable, "Unschedulable"),
			},
			expected: sets.NewString("node2", "node3", "node4"),
		},
		{
			name: "Node condition errors should be considered unresolvable",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodeunschedulable.ErrReasonUnknownCondition),
				"node2": framework.NewStatus(framework.Unschedulable, "Unschedulable"),
				"node3": framework.NewStatus(framework.Unschedulable, "Unschedulable"),
				"node4": framework.NewStatus(framework.Unschedulable, "Unschedulable"),
			},
			expected: sets.NewString("node2", "node3", "node4"),
		},
		{
			name: "ErrVolume... errors should not be tried as it indicates that the pod is unschedulable due to no matching volumes for pod on node",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, volumezone.ErrReasonConflict),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, string(volumebinding.ErrReasonNodeConflict)),
				"node3": framework.NewStatus(framework.UnschedulableAndUnresolvable, string(volumebinding.ErrReasonBindConflict)),
				"node4": framework.NewStatus(framework.Unschedulable, "Unschedulable"),
			},
			expected: sets.NewString("node4"),
		},
		{
			name: "ErrReasonConstraintsNotMatch should be tried as it indicates that the pod is unschedulable due to topology spread constraints",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable, podtopologyspread.ErrReasonConstraintsNotMatch),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
				"node3": framework.NewStatus(framework.Unschedulable, podtopologyspread.ErrReasonConstraintsNotMatch),
				"node4": framework.NewStatus(framework.Unschedulable, "Unschedulable"),
			},
			expected: sets.NewString("node1", "node3", "node4"),
		},
		{
			name: "UnschedulableAndUnresolvable status should be skipped but Unschedulable should be tried",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable, ""),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, ""),
				"node3": framework.NewStatus(framework.Unschedulable, ""),
				"node4": framework.NewStatus(framework.UnschedulableAndUnresolvable, ""),
			},
			expected: sets.NewString("node1", "node3"),
		},
		{
			name: "ErrReasonNodeLabelNotMatch should not be tried as it indicates that the pod is unschedulable due to node doesn't have the required label",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable, ""),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, podtopologyspread.ErrReasonNodeLabelNotMatch),
				"node3": framework.NewStatus(framework.Unschedulable, ""),
				"node4": framework.NewStatus(framework.UnschedulableAndUnresolvable, ""),
			},
			expected: sets.NewString("node1", "node3"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var nodeInfos []*framework.NodeInfo
			for _, name := range nodeNames {
				ni := framework.NewNodeInfo()
				ni.SetNode(st.MakeNode().Name(name).Obj())
				nodeInfos = append(nodeInfos, ni)
			}
			nodes, _ := nodesWherePreemptionMightHelp(nodeInfos, tt.nodesStatuses)
			if len(tt.expected) != len(nodes) {
				t.Errorf("number of nodes is not the same as expected. exptectd: %d, got: %d. Nodes: %v", len(tt.expected), len(nodes), nodes)
			}
			for _, node := range nodes {
				name := node.Node().Name
				if _, found := tt.expected[name]; !found {
					t.Errorf("node %v is not expected.", name)
				}
			}
		})
	}
}

func TestDryRunPreemption(t *testing.T) {
	tests := []struct {
		name               string
		nodes              []*v1.Node
		testPods           []*v1.Pod
		initPods           []*v1.Pod
		numViolatingVictim int
		expected           [][]Candidate
	}{
		{
			name: "no pdb violation",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(veryLargeRes).Obj(),
				st.MakeNode().Name("node2").Capacity(veryLargeRes).Obj(),
			},
			testPods: []*v1.Pod{
				st.MakePod().Name("p").UID("p").Priority(highPriority).Obj(),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
			},
			expected: [][]Candidate{
				{
					&candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj()},
						},
						name: "node1",
					},
					&candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj()},
						},
						name: "node2",
					},
				},
			},
		},
		{
			name: "pdb violation on each node",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(veryLargeRes).Obj(),
				st.MakeNode().Name("node2").Capacity(veryLargeRes).Obj(),
			},
			testPods: []*v1.Pod{
				st.MakePod().Name("p").UID("p").Priority(highPriority).Obj(),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
			},
			numViolatingVictim: 1,
			expected: [][]Candidate{
				{
					&candidate{
						victims: &extenderv1.Victims{
							Pods:             []*v1.Pod{st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj()},
							NumPDBViolations: 1,
						},
						name: "node1",
					},
					&candidate{
						victims: &extenderv1.Victims{
							Pods:             []*v1.Pod{st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj()},
							NumPDBViolations: 1,
						},
						name: "node2",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registeredPlugins := append([]st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New)},
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			var objs []runtime.Object
			for _, p := range append(tt.testPods, tt.initPods...) {
				objs = append(objs, p)
			}
			for _, n := range tt.nodes {
				objs = append(objs, n)
			}
			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewSimpleClientset(objs...), 0)
			parallelism := parallelize.DefaultParallelism
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			fwk, err := st.NewFramework(
				registeredPlugins, "", ctx.Done(),
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator(informerFactory.Core().V1().Pods().Lister())),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithParallelism(parallelism),
				frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot(tt.testPods, tt.nodes)),
			)
			if err != nil {
				t.Fatal(err)
			}

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			snapshot := internalcache.NewSnapshot(tt.initPods, tt.nodes)
			nodeInfos, err := snapshot.NodeInfos().List()
			if err != nil {
				t.Fatal(err)
			}
			sort.Slice(nodeInfos, func(i, j int) bool {
				return nodeInfos[i].Node().Name < nodeInfos[j].Node().Name
			})

			fakePostPlugin := &FakePostFilterPlugin{numViolatingVictim: tt.numViolatingVictim}

			for cycle, pod := range tt.testPods {
				state := framework.NewCycleState()
				pe := Evaluator{
					PluginName: "FakePostFilter",
					Handler:    fwk,
					Interface:  fakePostPlugin,
					State:      state,
				}
				got, _, _ := pe.DryRunPreemption(context.Background(), pod, nodeInfos, nil, 0, int32(len(nodeInfos)))
				// Sort the values (inner victims) and the candidate itself (by its NominatedNodeName).
				for i := range got {
					victims := got[i].Victims().Pods
					sort.Slice(victims, func(i, j int) bool {
						return victims[i].Name < victims[j].Name
					})
				}
				sort.Slice(got, func(i, j int) bool {
					return got[i].Name() < got[j].Name()
				})
				if diff := cmp.Diff(tt.expected[cycle], got, cmp.AllowUnexported(candidate{})); diff != "" {
					t.Errorf("cycle %d: unexpected candidates (-want, +got): %s", cycle, diff)
				}
			}
		})
	}
}
