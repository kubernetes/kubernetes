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

package defaultpreemption

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	volumescheduling "k8s.io/kubernetes/pkg/controller/volume/scheduling"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodelabel"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeunschedulable"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumerestrictions"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumezone"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

var (
	negPriority, lowPriority, midPriority, highPriority, veryHighPriority = int32(-100), int32(0), int32(100), int32(1000), int32(10000)
	smallCPU, mediumCPU, largeCPU, veryLargeCPU                           = "100m", "200m", "300m", "500m"
	smallMem, mediumMem, largeMem, veryLargeMem                           = "100", "200", "300", "500"

	epochTime  = metav1.NewTime(time.Unix(0, 0))
	epochTime1 = metav1.NewTime(time.Unix(0, 1))
	epochTime2 = metav1.NewTime(time.Unix(0, 2))
	epochTime3 = metav1.NewTime(time.Unix(0, 3))
	epochTime4 = metav1.NewTime(time.Unix(0, 4))
	epochTime5 = metav1.NewTime(time.Unix(0, 5))
	epochTime6 = metav1.NewTime(time.Unix(0, 6))
)

var _ framework.PreemptHandle = &preemptHandle{}

type preemptHandle struct {
	extenders []framework.Extender
	framework.PodNominator
	framework.PluginsRunner
}

// Extenders returns the registered extenders.
func (ph *preemptHandle) Extenders() []framework.Extender {
	return ph.extenders
}

func newPreemptHandle(es []framework.Extender, pn framework.PodNominator, f framework.Framework) *preemptHandle {
	ph := &preemptHandle{
		extenders:     es,
		PodNominator:  pn,
		PluginsRunner: f,
	}
	return ph
}

func TestPostFilter(t *testing.T) {
	tests := []struct {
		name                  string
		pod                   *v1.Pod
		pods                  []*v1.Pod
		nodes                 []*v1.Node
		filteredNodesStatuses framework.NodeToStatusMap
		extenders             []framework.Extender
		wantResult            *framework.PostFilterResult
		wantStatus            *framework.Status
	}{
		{
			name: "pod with higher priority can be made schedulable",
			pod:  st.MakePod().Name("p1").UID("p1").Priority(100).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("r1").UID("r1").Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity("", "", 1).Obj(),
			},
			filteredNodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable),
			},
			wantResult: &framework.PostFilterResult{
				NominatedNodeName: "node1",
				Victims: []*v1.Pod{
					st.MakePod().Name("r1").UID("r1").Node("node1").Obj(),
				},
			},
			wantStatus: framework.NewStatus(framework.Success),
		},
		{
			name: "pod with tied priority is still unschedulable",
			pod:  st.MakePod().Name("p1").UID("p1").Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("r1").UID("r1").Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity("", "", 1).Obj(),
			},
			filteredNodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable),
			},
			wantResult: nil,
			wantStatus: framework.NewStatus(framework.Unschedulable, NoPreemptionStrategy),
		},
		{
			name: "preemption should respect filteredNodesStatuses",
			pod:  st.MakePod().Name("p1").UID("p1").Priority(100).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("r1").UID("r1").Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity("", "", 1).Obj(),
			},
			filteredNodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable),
			},
			wantResult: nil,
			wantStatus: framework.NewStatus(framework.Unschedulable, NoPreemptionStrategy),
		},
		{
			name: "pod can be made schedulable on one node",
			pod:  st.MakePod().Name("p1").UID("p1").Priority(100).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("r1").UID("r1").Priority(200).Node("node1").Obj(),
				st.MakePod().Name("r2").UID("r2").Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity("", "", 1).Obj(),
				st.MakeNode().Name("node2").Capacity("", "", 1).Obj(),
			},
			filteredNodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable),
				"node2": framework.NewStatus(framework.Unschedulable),
			},
			wantResult: &framework.PostFilterResult{
				NominatedNodeName: "node2",
				Victims: []*v1.Pod{
					st.MakePod().Name("r2").UID("r2").Node("node2").Obj(),
				},
			},
			wantStatus: framework.NewStatus(framework.Success),
		},
		{
			name: "preemption result filtered out by extenders",
			pod:  st.MakePod().Name("p1").UID("p1").Priority(100).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("r1").UID("r1").Node("node1").Obj(),
				st.MakePod().Name("r2").UID("r2").Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity("", "", 1).Obj(),
				st.MakeNode().Name("node2").Capacity("", "", 1).Obj(),
			},
			filteredNodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable),
				"node2": framework.NewStatus(framework.Unschedulable),
			},
			extenders: []framework.Extender{&fakeExtender{}},
			wantResult: &framework.PostFilterResult{
				NominatedNodeName: "node1",
				Victims: []*v1.Pod{
					st.MakePod().Name("r1").UID("r1").Node("node1").Obj(),
				},
			},
			wantStatus: framework.NewStatus(framework.Success),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			apiObjs := mergeObjs(tt.pod, tt.pods, tt.nodes)
			cs := fake.NewSimpleClientset(apiObjs...)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			snapshot := cache.NewSnapshot(tt.pods, tt.nodes)
			podNominator := internalqueue.NewPodNominator()
			// Register NodeResourceFit as the Filter & PreFilter plugin.
			registeredPlugins := []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			f, err := st.NewFramework(registeredPlugins,
				framework.WithClientSet(cs),
				framework.WithInformerFactory(informerFactory),
				framework.WithPodNominator(podNominator),
				framework.WithExtenders(tt.extenders),
				framework.WithSnapshotSharedLister(snapshot),
			)
			if err != nil {
				t.Fatal(err)
			}
			p := DefaultPreemption{fh: f}

			state := framework.NewCycleState()
			// Ensure <state> is populated.
			if status := f.RunPreFilterPlugins(context.Background(), state, tt.pod); !status.IsSuccess() {
				t.Errorf("Unexpected PreFilter Status: %v", status)
			}

			gotResult, gotStatus := p.PostFilter(context.TODO(), state, tt.pod, tt.filteredNodesStatuses)
			if !reflect.DeepEqual(gotStatus, tt.wantStatus) {
				t.Errorf("Status does not match: %v, want: %v", gotStatus, tt.wantStatus)
			}
			if diff := cmp.Diff(gotResult, tt.wantResult); diff != "" {
				t.Errorf("Unexpected postFilterResult (-want, +got): %s", diff)
			}
		})
	}
}

func mergeObjs(pod *v1.Pod, pods []*v1.Pod, nodes []*v1.Node) []runtime.Object {
	var objs []runtime.Object
	if pod != nil {
		objs = append(objs, pod)
	}
	for i := range pods {
		objs = append(objs, pods[i])
	}
	for i := range nodes {
		objs = append(objs, nodes[i])
	}
	return objs
}

func TestNodesWherePreemptionMightHelp(t *testing.T) {
	// Prepare 4 nodes names.
	nodeNames := []string{"node1", "node2", "node3", "node4"}

	tests := []struct {
		name          string
		nodesStatuses framework.NodeToStatusMap
		expected      map[string]bool // set of expected node names. Value is ignored.
	}{
		{
			name: "No node should be attempted",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodeaffinity.ErrReason),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
				"node3": framework.NewStatus(framework.UnschedulableAndUnresolvable, tainttoleration.ErrReasonNotMatch),
				"node4": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodelabel.ErrReasonPresenceViolated),
			},
			expected: map[string]bool{},
		},
		{
			name: "ErrReasonAffinityNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod affinity or anti-affinity",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable, interpodaffinity.ErrReasonAffinityNotMatch),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
				"node3": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodeunschedulable.ErrReasonUnschedulable),
			},
			expected: map[string]bool{"node1": true, "node4": true},
		},
		{
			name: "pod with both pod affinity and anti-affinity should be tried",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable, interpodaffinity.ErrReasonAffinityNotMatch),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
			},
			expected: map[string]bool{"node1": true, "node3": true, "node4": true},
		},
		{
			name: "ErrReasonAffinityRulesNotMatch should not be tried as it indicates that the pod is unschedulable due to inter-pod affinity, but ErrReasonAffinityNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod affinity or anti-affinity",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, interpodaffinity.ErrReasonAffinityRulesNotMatch),
				"node2": framework.NewStatus(framework.Unschedulable, interpodaffinity.ErrReasonAffinityNotMatch),
			},
			expected: map[string]bool{"node2": true, "node3": true, "node4": true},
		},
		{
			name: "Mix of failed predicates works fine",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, volumerestrictions.ErrReasonDiskConflict),
				"node2": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("Insufficient %v", v1.ResourceMemory)),
			},
			expected: map[string]bool{"node2": true, "node3": true, "node4": true},
		},
		{
			name: "Node condition errors should be considered unresolvable",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodeunschedulable.ErrReasonUnknownCondition),
			},
			expected: map[string]bool{"node2": true, "node3": true, "node4": true},
		},
		{
			name: "ErrVolume... errors should not be tried as it indicates that the pod is unschedulable due to no matching volumes for pod on node",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, volumezone.ErrReasonConflict),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, string(volumescheduling.ErrReasonNodeConflict)),
				"node3": framework.NewStatus(framework.UnschedulableAndUnresolvable, string(volumescheduling.ErrReasonBindConflict)),
			},
			expected: map[string]bool{"node4": true},
		},
		{
			name: "ErrTopologySpreadConstraintsNotMatch should be tried as it indicates that the pod is unschedulable due to topology spread constraints",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable, podtopologyspread.ErrReasonConstraintsNotMatch),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, nodename.ErrReason),
				"node3": framework.NewStatus(framework.Unschedulable, podtopologyspread.ErrReasonConstraintsNotMatch),
			},
			expected: map[string]bool{"node1": true, "node3": true, "node4": true},
		},
		{
			name: "UnschedulableAndUnresolvable status should be skipped but Unschedulable should be tried",
			nodesStatuses: framework.NodeToStatusMap{
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, ""),
				"node3": framework.NewStatus(framework.Unschedulable, ""),
				"node4": framework.NewStatus(framework.UnschedulableAndUnresolvable, ""),
			},
			expected: map[string]bool{"node1": true, "node3": true},
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
			nodes := nodesWherePreemptionMightHelp(nodeInfos, tt.nodesStatuses)
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

// TestDryRunPreemption tests dryRunPreemption. This test assumes
// that PodPassesFiltersOnNode works correctly and is tested separately.
func TestDryRunPreemption(t *testing.T) {
	tests := []struct {
		name                    string
		nodeNames               []string
		pod                     *v1.Pod
		pods                    []*v1.Pod
		registerPlugins         []st.RegisterPluginFunc
		pdbs                    []*policy.PodDisruptionBudget
		fakeFilterRC            framework.Code // return code for fake filter plugin
		expected                []Strategy
		expectedNumFilterCalled int32
	}{
		{
			name: "a pod that does not fit on any machine",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("FalseFilter", NewFalseFilterPlugin),
			},
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
			},
			expected:                nil,
			expectedNumFilterCalled: 2,
		},
		{
			name: "a pod that fits with no preemption",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
			},
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
			},
			expected: []Strategy{
				&strategy{victims: &extenderv1.Victims{}, nominatedNodeName: "node1"},
				&strategy{victims: &extenderv1.Victims{}, nominatedNodeName: "node2"},
			},
			expectedNumFilterCalled: 4,
		},
		{
			name: "a pod that fits on one machine with no preemption",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("MatchFilter", NewMatchFilterPlugin),
			},
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("node1").UID("node1").Priority(highPriority).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
			},
			expected: []Strategy{
				&strategy{victims: &extenderv1.Victims{}, nominatedNodeName: "node1"},
			},
			expectedNumFilterCalled: 3,
		},
		{
			name: "a pod that fits on both machines when lower priority pods are preempted",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeCPU, largeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeCPU, largeMem).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).Obj(),
			},
			expected: []Strategy{
				&strategy{
					victims: &extenderv1.Victims{
						Pods: []*v1.Pod{st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeCPU, largeMem).Obj()},
					},
					nominatedNodeName: "node1",
				},
				&strategy{
					victims: &extenderv1.Victims{
						Pods: []*v1.Pod{st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).Obj()},
					},
					nominatedNodeName: "node2",
				},
			},
			expectedNumFilterCalled: 4,
		},
		{
			name: "a pod that would fit on the machines, but other pods running are higher priority",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("p").UID("p").Priority(lowPriority).Req(largeCPU, largeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeCPU, largeMem).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).Obj(),
			},
			expected:                nil,
			expectedNumFilterCalled: 2,
		},
		{
			name: "medium priority pod is preempted, but lower priority one stays as it is small",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeCPU, largeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(lowPriority).Req(smallCPU, smallMem).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(largeCPU, largeMem).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).Obj(),
			},
			expected: []Strategy{
				&strategy{
					victims: &extenderv1.Victims{
						Pods: []*v1.Pod{st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(largeCPU, largeMem).Obj()},
					},
					nominatedNodeName: "node1",
				},
				&strategy{
					victims: &extenderv1.Victims{
						Pods: []*v1.Pod{st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).Obj()},
					},
					nominatedNodeName: "node2",
				},
			},
			expectedNumFilterCalled: 5,
		},
		{
			name: "mixed priority pods are preempted",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeCPU, largeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(smallCPU, smallMem).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(lowPriority).Req(smallCPU, smallMem).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).Obj(),
				st.MakePod().Name("p1.4").UID("p1.4").Node("node1").Priority(highPriority).Req(smallCPU, smallMem).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(highPriority).Req(largeCPU, largeMem).Obj(),
			},
			expected: []Strategy{
				&strategy{
					victims: &extenderv1.Victims{
						Pods: []*v1.Pod{
							st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(lowPriority).Req(smallCPU, smallMem).Obj(),
							st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).Obj(),
						},
					},
					nominatedNodeName: "node1",
				},
			},
			expectedNumFilterCalled: 5,
		},
		{
			name: "mixed priority pods are preempted, pick later StartTime one when priorities are equal",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeCPU, largeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(lowPriority).Req(smallCPU, smallMem).StartTime(epochTime5).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(lowPriority).Req(smallCPU, smallMem).StartTime(epochTime4).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime3).Obj(),
				st.MakePod().Name("p1.4").UID("p1.4").Node("node1").Priority(highPriority).Req(smallCPU, smallMem).StartTime(epochTime2).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(highPriority).Req(largeCPU, largeMem).StartTime(epochTime1).Obj(),
			},
			expected: []Strategy{
				&strategy{
					victims: &extenderv1.Victims{
						Pods: []*v1.Pod{
							st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(lowPriority).Req(smallCPU, smallMem).StartTime(epochTime5).Obj(),
							st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime3).Obj(),
						},
					},
					nominatedNodeName: "node1",
				},
			},
			expectedNumFilterCalled: 5,
		},
		{
			name: "pod with anti-affinity is preempted",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterPluginAsExtensions(interpodaffinity.Name, interpodaffinity.New, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("p").UID("p").Label("foo", "").Priority(highPriority).Req(smallCPU, smallMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("foo", "").Priority(lowPriority).Req(smallCPU, smallMem).
					PodAntiAffinityExists("foo", "hostname", st.PodAntiAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(smallCPU, smallMem).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(highPriority).Req(smallCPU, smallMem).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(highPriority).Req(smallCPU, smallMem).Obj(),
			},
			expected: []Strategy{
				&strategy{
					victims: &extenderv1.Victims{
						Pods: []*v1.Pod{
							st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("foo", "").Priority(lowPriority).Req(smallCPU, smallMem).
								PodAntiAffinityExists("foo", "hostname", st.PodAntiAffinityWithRequiredReq).Obj(),
						},
					},
					nominatedNodeName: "node1",
				},
				&strategy{victims: &extenderv1.Victims{}, nominatedNodeName: "node2"},
			},
			expectedNumFilterCalled: 4,
		},
		{
			name: "preemption to resolve pod topology spread filter failure",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterPluginAsExtensions(podtopologyspread.Name, podtopologyspread.New, "PreFilter", "Filter"),
			},
			nodeNames: []string{"node-a/zone1", "node-b/zone1", "node-x/zone2"},
			pod: st.MakePod().Name("p").UID("p").Label("foo", "").Priority(highPriority).
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "hostname", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("pod-a1").UID("pod-a1").Node("node-a").Label("foo", "").Priority(midPriority).Obj(),
				st.MakePod().Name("pod-a2").UID("pod-a2").Node("node-a").Label("foo", "").Priority(lowPriority).Obj(),
				st.MakePod().Name("pod-b1").UID("pod-b1").Node("node-b").Label("foo", "").Priority(lowPriority).Obj(),
				st.MakePod().Name("pod-x1").UID("pod-x1").Node("node-x").Label("foo", "").Priority(highPriority).Obj(),
				st.MakePod().Name("pod-x2").UID("pod-x2").Node("node-x").Label("foo", "").Priority(highPriority).Obj(),
			},
			expected: []Strategy{
				&strategy{
					victims: &extenderv1.Victims{
						Pods: []*v1.Pod{st.MakePod().Name("pod-a2").UID("pod-a2").Node("node-a").Label("foo", "").Priority(lowPriority).Obj()},
					},
					nominatedNodeName: "node-a",
				},
				&strategy{
					victims: &extenderv1.Victims{
						Pods: []*v1.Pod{st.MakePod().Name("pod-b1").UID("pod-b1").Node("node-b").Label("foo", "").Priority(lowPriority).Obj()},
					},
					nominatedNodeName: "node-b",
				},
			},
			expectedNumFilterCalled: 6,
		},
		{
			name: "get Unschedulable in the preemption phase when the filter plugins filtering the nodes",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeCPU, largeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeCPU, largeMem).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).Obj(),
			},
			fakeFilterRC:            framework.Unschedulable,
			expected:                nil,
			expectedNumFilterCalled: 2,
		},
		{
			name: "preemption with violation of same pdb",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeCPU, veryLargeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumCPU, mediumMem).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumCPU, mediumMem).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{Spec: policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}}, Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 1}},
			},
			expected: []Strategy{
				&strategy{
					victims: &extenderv1.Victims{
						Pods: []*v1.Pod{
							st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumCPU, mediumMem).Obj(),
							st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumCPU, mediumMem).Obj(),
						},
						NumPDBViolations: 1,
					},
					nominatedNodeName: "node1",
				},
			},
			expectedNumFilterCalled: 3,
		},
	}

	labelKeys := []string{"hostname", "zone", "region"}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodes := make([]*v1.Node, len(tt.nodeNames))
			fakeFilterRCMap := make(map[string]framework.Code, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodeWrapper := st.MakeNode().Capacity(veryLargeCPU, veryLargeMem, 100)
				// Split node name by '/' to form labels in a format of
				// {"hostname": tpKeys[0], "zone": tpKeys[1], "region": tpKeys[2]}
				tpKeys := strings.Split(nodeName, "/")
				nodeWrapper.Name(tpKeys[0])
				for i, labelVal := range strings.Split(nodeName, "/") {
					nodeWrapper.Label(labelKeys[i], labelVal)
				}
				nodes[i] = nodeWrapper.Obj()
				fakeFilterRCMap[nodeName] = tt.fakeFilterRC
			}
			snapshot := cache.NewSnapshot(tt.pods, nodes)
			podNominator := internalqueue.NewPodNominator()

			// For each test, register a FakeFilterPlugin along with essential plugins and tt.registerPlugins.
			fakePlugin := fakeFilterPlugin{
				failedNodeReturnCodeMap: fakeFilterRCMap,
			}
			registeredPlugins := append([]st.RegisterPluginFunc{
				st.RegisterFilterPlugin(
					"FakeFilter",
					func(_ runtime.Object, fh framework.FrameworkHandle) (framework.Plugin, error) {
						return &fakePlugin, nil
					},
				)},
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			registeredPlugins = append(registeredPlugins, tt.registerPlugins...)
			f, err := st.NewFramework(
				registeredPlugins,
				framework.WithPodNominator(podNominator),
				framework.WithSnapshotSharedLister(snapshot),
			)
			if err != nil {
				t.Fatal(err)
			}
			ph := newPreemptHandle(nil, podNominator, f)

			state := framework.NewCycleState()
			// Some tests rely on PreFilter plugin to compute its CycleState.
			if status := f.RunPreFilterPlugins(context.Background(), state, tt.pod); !status.IsSuccess() {
				t.Errorf("Unexpected PreFilter Status: %v", status)
			}

			nodeInfos, err := snapshot.NodeInfos().List()
			if err != nil {
				t.Fatal(err)
			}
			got := dryRunPreemption(context.Background(), ph, state, tt.pod, nodeInfos, tt.pdbs)
			// Sort both the inner victims and strategy itself.
			for i := range got {
				victims := got[i].Victims().Pods
				sort.Slice(victims, func(i, j int) bool {
					return victims[i].Name < victims[j].Name
				})
			}
			sort.Slice(got, func(i, j int) bool {
				return got[i].NominatedNodeName() < got[j].NominatedNodeName()
			})

			if tt.expectedNumFilterCalled != fakePlugin.numFilterCalled {
				t.Errorf("expected fakePlugin.numFilterCalled is %d, but got %d", tt.expectedNumFilterCalled, fakePlugin.numFilterCalled)
			}
			if diff := cmp.Diff(tt.expected, got, cmp.AllowUnexported(strategy{})); diff != "" {
				t.Errorf("Unexpected strategies (-want, +got): %s", diff)
			}
		})
	}
}

func TestPickStrategy(t *testing.T) {
	tests := []struct {
		name      string
		nodeNames []string
		pod       *v1.Pod
		pods      []*v1.Pod
		expected  []string // any of the items is valid
	}{
		{
			name:      "No node needs preemption",
			nodeNames: []string{"node1"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeCPU, largeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
			},
			expected: []string{"node1"},
		},
		{
			name:      "a pod that fits on both nodes when lower priority pods are preempted",
			nodeNames: []string{"node1", "node2"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeCPU, largeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(largeCPU, largeMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).StartTime(epochTime).Obj(),
			},
			expected: []string{"node1", "node2"},
		},
		{
			name:      "a pod that fits on a node with no preemption",
			nodeNames: []string{"node1", "node2", "node3"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeCPU, largeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(largeCPU, largeMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).StartTime(epochTime).Obj(),
			},
			expected: []string{"node3"},
		},
		{
			name:      "node with min highest priority pod is picked",
			nodeNames: []string{"node1", "node2", "node3"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeCPU, veryLargeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(largeCPU, largeMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(lowPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(lowPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(lowPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
			},
			expected: []string{"node3"},
		},
		{
			name:      "when highest priorities are the same, minimum sum of priorities is picked",
			nodeNames: []string{"node1", "node2", "node3"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeCPU, veryLargeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(largeCPU, largeMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(lowPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
			},
			expected: []string{"node2"},
		},
		{
			name:      "when highest priority and sum are the same, minimum number of pods is picked",
			nodeNames: []string{"node1", "node2", "node3"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeCPU, veryLargeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(negPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(midPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.4").UID("p1.4").Node("node1").Priority(negPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(negPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(negPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.3").UID("p3.3").Node("node3").Priority(lowPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
			},
			expected: []string{"node2"},
		},
		{
			// pickOneNodeForPreemption adjusts pod priorities when finding the sum of the victims. This
			// test ensures that the logic works correctly.
			name:      "sum of adjusted priorities is considered",
			nodeNames: []string{"node1", "node2", "node3"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeCPU, veryLargeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(negPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(negPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(largeCPU, largeMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(negPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(negPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.3").UID("p3.3").Node("node3").Priority(lowPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
			},
			expected: []string{"node2"},
		},
		{
			name:      "non-overlapping lowest high priority, sum priorities, and number of pods",
			nodeNames: []string{"node1", "node2", "node3", "node4"},
			pod:       st.MakePod().Name("p").UID("p").Priority(veryHighPriority).Req(veryLargeCPU, veryLargeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(lowPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(lowPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(highPriority).Req(largeCPU, largeMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(lowPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.3").UID("p3.3").Node("node3").Priority(lowPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.4").UID("p3.4").Node("node3").Priority(lowPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p4.1").UID("p4.1").Node("node4").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p4.2").UID("p4.2").Node("node4").Priority(midPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p4.3").UID("p4.3").Node("node4").Priority(midPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
				st.MakePod().Name("p4.4").UID("p4.4").Node("node4").Priority(negPriority).Req(smallCPU, smallMem).StartTime(epochTime).Obj(),
			},
			expected: []string{"node1"},
		},
		{
			name:      "same priority, same number of victims, different start time for each node's pod",
			nodeNames: []string{"node1", "node2", "node3"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeCPU, veryLargeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime2).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime2).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime3).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime3).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime1).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime1).Obj(),
			},
			expected: []string{"node2"},
		},
		{
			name:      "same priority, same number of victims, different start time for all pods",
			nodeNames: []string{"node1", "node2", "node3"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeCPU, veryLargeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime4).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime2).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime5).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime1).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime3).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime6).Obj(),
			},
			expected: []string{"node3"},
		},
		{
			name:      "different priority, same number of victims, different start time for all pods",
			nodeNames: []string{"node1", "node2", "node3"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeCPU, veryLargeMem).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(lowPriority).Req(mediumCPU, mediumMem).StartTime(epochTime4).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime2).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime6).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(lowPriority).Req(mediumCPU, mediumMem).StartTime(epochTime1).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(lowPriority).Req(mediumCPU, mediumMem).StartTime(epochTime3).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(midPriority).Req(mediumCPU, mediumMem).StartTime(epochTime5).Obj(),
			},
			expected: []string{"node2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodes := make([]*v1.Node, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodes[i] = st.MakeNode().Name(nodeName).Capacity(veryLargeCPU, veryLargeMem, 100).Obj()
			}
			snapshot := cache.NewSnapshot(tt.pods, nodes)
			podNominator := internalqueue.NewPodNominator()
			// Register NodeResourceFit as the Filter & PreFilter plugin.
			registeredPlugins := []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			f, err := st.NewFramework(
				registeredPlugins,
				framework.WithPodNominator(podNominator),
				framework.WithSnapshotSharedLister(snapshot),
			)
			if err != nil {
				t.Fatal(err)
			}
			ph := newPreemptHandle(nil, podNominator, f)

			state := framework.NewCycleState()
			// Some tests rely on PreFilter plugin to compute its CycleState.
			if status := f.RunPreFilterPlugins(context.Background(), state, tt.pod); !status.IsSuccess() {
				t.Errorf("Unexpected PreFilter Status: %v", status)
			}

			nodeInfos, err := snapshot.NodeInfos().List()
			if err != nil {
				t.Fatal(err)
			}
			ss := dryRunPreemption(context.Background(), ph, state, tt.pod, nodeInfos, nil)
			s := PickStrategy(ss)
			found := false
			for _, nodeName := range tt.expected {
				if nodeName == s.NominatedNodeName() {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("expect any node in %v, but got %v", tt.expected, s.NominatedNodeName())
			}
		})
	}
}

var _ framework.Extender = &fakeExtender{}

type fakeExtender struct{}

func (f *fakeExtender) Name() string             { return "fakeExtender" }
func (f *fakeExtender) IsIgnorable() bool        { return false }
func (f *fakeExtender) SupportsPreemption() bool { return true }
func (f *fakeExtender) ProcessPreemption(_ *v1.Pod, nodeNameToVictims map[string]*extenderv1.Victims, _ framework.NodeInfoLister) (map[string]*extenderv1.Victims, error) {
	if len(nodeNameToVictims) <= 1 {
		return nodeNameToVictims, nil
	}
	// Return the first entry sort by nodeName.
	finalCandidate, finalVictims := "", &extenderv1.Victims{}
	for nodeName := range nodeNameToVictims {
		if finalCandidate == "" || finalCandidate > nodeName {
			finalCandidate, finalVictims = nodeName, nodeNameToVictims[nodeName]
			continue
		}
	}
	return map[string]*extenderv1.Victims{finalCandidate: finalVictims}, nil
}
func (f *fakeExtender) Filter(pod *v1.Pod, nodes []*v1.Node) ([]*v1.Node, extenderv1.FailedNodesMap, error) {
	return nil, nil, nil
}
func (f *fakeExtender) Prioritize(pod *v1.Pod, nodes []*v1.Node) (*extenderv1.HostPriorityList, int64, error) {
	return nil, 0, nil
}
func (f *fakeExtender) Bind(binding *v1.Binding) error {
	return nil
}
func (f *fakeExtender) IsBinder() bool { return false }

func (f *fakeExtender) IsInterested(_ *v1.Pod) bool { return true }

/*
 * Fake filter plugins below.
 * TODO(Huang-Wei): De-duplicate them with pkg/scheduler/core/generic_scheduler_test.go
 */

const ErrReasonFake = "Nodes failed the fake predicate"

type falseFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *falseFilterPlugin) Name() string {
	return "FalseFilter"
}

// Filter invoked at the filter extension point.
func (pl *falseFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	return framework.NewStatus(framework.Unschedulable, ErrReasonFake)
}

// NewFalseFilterPlugin initializes a falseFilterPlugin and returns it.
func NewFalseFilterPlugin(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &falseFilterPlugin{}, nil
}

type trueFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *trueFilterPlugin) Name() string {
	return "TrueFilter"
}

// Filter invoked at the filter extension point.
func (pl *trueFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	return nil
}

// NewTrueFilterPlugin initializes a trueFilterPlugin and returns it.
func NewTrueFilterPlugin(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &trueFilterPlugin{}, nil
}

// fakeFilterPlugin is a test filter plugin to record how many times its Filter() function have
// been called, and it returns different 'Code' depending on its internal 'failedNodeReturnCodeMap'.
type fakeFilterPlugin struct {
	numFilterCalled         int32
	failedNodeReturnCodeMap map[string]framework.Code
}

// Name returns name of the plugin.
func (pl *fakeFilterPlugin) Name() string {
	return "FakeFilter"
}

// Filter invoked at the filter extension point.
func (pl *fakeFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	atomic.AddInt32(&pl.numFilterCalled, 1)

	if returnCode, ok := pl.failedNodeReturnCodeMap[nodeInfo.Node().Name]; ok {
		return framework.NewStatus(returnCode, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}

	return nil
}

type matchFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *matchFilterPlugin) Name() string {
	return "MatchFilter"
}

// Filter invoked at the filter extension point.
func (pl *matchFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}
	if pod.Name == node.Name {
		return nil
	}
	return framework.NewStatus(framework.Unschedulable, ErrReasonFake)
}

// NewMatchFilterPlugin initializes a matchFilterPlugin and returns it.
func NewMatchFilterPlugin(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &matchFilterPlugin{}, nil
}
