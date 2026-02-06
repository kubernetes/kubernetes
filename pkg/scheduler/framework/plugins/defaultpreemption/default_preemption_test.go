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
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/events"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	apipod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	configv1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1"
	apicache "k8s.io/kubernetes/pkg/scheduler/backend/api_cache"
	apidispatcher "k8s.io/kubernetes/pkg/scheduler/backend/api_dispatcher"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	apicalls "k8s.io/kubernetes/pkg/scheduler/framework/api_calls"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

var (
	negPriority, lowPriority, midPriority, highPriority, veryHighPriority = int32(-100), int32(0), int32(100), int32(1000), int32(10000)

	smallRes = map[v1.ResourceName]string{
		v1.ResourceCPU:    "100m",
		v1.ResourceMemory: "100",
	}
	mediumRes = map[v1.ResourceName]string{
		v1.ResourceCPU:    "200m",
		v1.ResourceMemory: "200",
	}
	largeRes = map[v1.ResourceName]string{
		v1.ResourceCPU:    "300m",
		v1.ResourceMemory: "300",
	}
	veryLargeRes = map[v1.ResourceName]string{
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}

	epochTime  = metav1.NewTime(time.Unix(0, 0))
	epochTime1 = metav1.NewTime(time.Unix(0, 1))
	epochTime2 = metav1.NewTime(time.Unix(0, 2))
	epochTime3 = metav1.NewTime(time.Unix(0, 3))
	epochTime4 = metav1.NewTime(time.Unix(0, 4))
	epochTime5 = metav1.NewTime(time.Unix(0, 5))
	epochTime6 = metav1.NewTime(time.Unix(0, 6))
)

func init() {
	metrics.Register()
}

func getDefaultDefaultPreemptionArgs() *config.DefaultPreemptionArgs {
	v1dpa := &kubeschedulerconfigv1.DefaultPreemptionArgs{}
	configv1.SetDefaults_DefaultPreemptionArgs(v1dpa)
	dpa := &config.DefaultPreemptionArgs{}
	configv1.Convert_v1_DefaultPreemptionArgs_To_config_DefaultPreemptionArgs(v1dpa, dpa, nil)
	return dpa
}

var nodeResourcesFitFunc = frameworkruntime.FactoryAdapter(feature.Features{}, noderesources.NewFit)
var podTopologySpreadFunc = frameworkruntime.FactoryAdapter(feature.Features{}, podtopologyspread.New)

// TestPlugin returns Error status when trying to `AddPod` or `RemovePod` on the nodes which have the {k,v} label pair defined on the nodes.
type TestPlugin struct {
	name string
}

func newTestPlugin(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	return &TestPlugin{name: "test-plugin"}, nil
}

func (pl *TestPlugin) AddPod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	if nodeInfo.Node().GetLabels()["error"] == "true" {
		return fwk.AsStatus(fmt.Errorf("failed to add pod: %v", podToSchedule.Name))
	}
	return nil
}

func (pl *TestPlugin) RemovePod(ctx context.Context, state fwk.CycleState, podToSchedule *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	if nodeInfo.Node().GetLabels()["error"] == "true" {
		return fwk.AsStatus(fmt.Errorf("failed to remove pod: %v", podToSchedule.Name))
	}
	return nil
}

func (pl *TestPlugin) Name() string {
	return pl.name
}

func (pl *TestPlugin) PreFilterExtensions() fwk.PreFilterExtensions {
	return pl
}

func (pl *TestPlugin) PreFilter(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	return nil, nil
}

func (pl *TestPlugin) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	return nil
}

const (
	LabelKeyIsViolatingPDB    = "test.kubernetes.io/is-violating-pdb"
	LabelValueViolatingPDB    = "violating"
	LabelValueNonViolatingPDB = "non-violating"
)

type blockingRule struct {
	nodeName        string
	blockingVictims []string
	capacity        int
}

type podGroupPreemptor struct {
	preemption.Preemptor
	priority         int32
	pods             []*v1.Pod
	isPodGroup       bool
	preemptionPolicy *v1.PreemptionPolicy
}

func (p *podGroupPreemptor) Priority() int32 {
	return p.priority
}

func (p *podGroupPreemptor) IsPodGroup() bool {
	return p.isPodGroup
}

func (p *podGroupPreemptor) Members() []*v1.Pod {
	return p.pods
}

func (p *podGroupPreemptor) IsEligibleToPreemptOthers() bool {
	return p.preemptionPolicy == nil || *p.preemptionPolicy != v1.PreemptNever
}

func (p *podGroupPreemptor) SupportExtenders() bool {
	return !p.isPodGroup
}

func (p *podGroupPreemptor) GetNamespace() string {
	if len(p.pods) > 0 {
		return p.pods[0].Namespace
	}
	return ""
}

func (p *podGroupPreemptor) GetName() string {
	if len(p.pods) == 0 {
		return "unknown"
	}

	firstPod := p.GetRepresentativePod()

	if p.isPodGroup {
		ref := firstPod.Spec.WorkloadRef

		// Start with the Workload Name (e.g., "my-job")
		name := ref.Name

		// Append PodGroup if distinct (e.g., "my-job/group-1")
		if ref.PodGroup != "" {
			name = name + "/" + ref.PodGroup
		}

		// Append ReplicaKey if present (e.g., "my-job/group-1/idx-0")
		// This is crucial for distinguishing between retries of the same job.
		if ref.PodGroupReplicaKey != "" {
			name = name + "/" + ref.PodGroupReplicaKey
		}

		return name
	}

	return firstPod.Name
}

func (p *podGroupPreemptor) GetRepresentativePod() *v1.Pod {
	if len(p.pods) == 0 {
		return nil
	}

	return p.pods[0]
}

func newPodGroupPreemptor(priority int32, members []*v1.Pod, preemptionPolicy *v1.PreemptionPolicy) preemption.Preemptor {
	return &podGroupPreemptor{
		priority:         priority,
		pods:             members,
		isPodGroup:       true,
		preemptionPolicy: preemptionPolicy,
	}
}

func TestPostFilter(t *testing.T) {
	onePodRes := map[v1.ResourceName]string{v1.ResourcePods: "1"}
	nodeRes := map[v1.ResourceName]string{v1.ResourceCPU: "200m", v1.ResourceMemory: "400"}
	tests := []struct {
		name                  string
		pod                   *v1.Pod
		pods                  []*v1.Pod
		pdbs                  []*policy.PodDisruptionBudget
		nodes                 []*v1.Node
		filteredNodesStatuses *framework.NodeToStatus
		extender              fwk.Extender
		wantResult            *fwk.PostFilterResult
		wantStatus            *fwk.Status
	}{
		{
			name: "pod with higher priority can be made schedulable",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(onePodRes).Obj(),
			},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			wantResult: framework.NewPostFilterResultWithNominatedNode("node1"),
			wantStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name: "pod with tied priority is still unschedulable",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(onePodRes).Obj(),
			},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			wantResult: framework.NewPostFilterResultWithNominatedNode(""),
			wantStatus: fwk.NewStatus(fwk.Unschedulable, "preemption: 0/1 nodes are available: 1 No preemption victims found for incoming pod."),
		},
		{
			name: "preemption should respect filteredNodesStatuses",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(onePodRes).Obj(),
			},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			wantResult: framework.NewPostFilterResultWithNominatedNode(""),
			wantStatus: fwk.NewStatus(fwk.Unschedulable, "preemption: 0/1 nodes are available: 1 Preemption is not helpful for scheduling."),
		},
		{
			name: "preemption should respect absent NodeToStatusReader entry meaning UnschedulableAndUnresolvable",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Node("node1").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(onePodRes).Obj(),
			},
			filteredNodesStatuses: framework.NewDefaultNodeToStatus(),
			wantResult:            framework.NewPostFilterResultWithNominatedNode(""),
			wantStatus:            fwk.NewStatus(fwk.Unschedulable, "preemption: 0/1 nodes are available: 1 Preemption is not helpful for scheduling."),
		},
		{
			name: "pod can be made schedulable on one node",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(midPriority).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Priority(highPriority).Node("node1").Obj(),
				st.MakePod().Name("p2").UID("p2").Namespace(v1.NamespaceDefault).Priority(lowPriority).Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(onePodRes).Obj(),
				st.MakeNode().Name("node2").Capacity(onePodRes).Obj(),
			},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
				"node2": fwk.NewStatus(fwk.Unschedulable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			wantResult: framework.NewPostFilterResultWithNominatedNode("node2"),
			wantStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name: "pod can be made schedulable on minHighestPriority node",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(veryHighPriority).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Label(LabelKeyIsViolatingPDB, LabelValueNonViolatingPDB).Namespace(v1.NamespaceDefault).Priority(highPriority).Node("node1").Obj(),
				st.MakePod().Name("p2").UID("p2").Label(LabelKeyIsViolatingPDB, LabelValueViolatingPDB).Namespace(v1.NamespaceDefault).Priority(lowPriority).Node("node1").Obj(),
				st.MakePod().Name("p3").UID("p3").Label(LabelKeyIsViolatingPDB, LabelValueViolatingPDB).Namespace(v1.NamespaceDefault).Priority(midPriority).Node("node2").Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				st.MakePDB().Name("violating-pdb").Namespace(v1.NamespaceDefault).MatchLabel(LabelKeyIsViolatingPDB, LabelValueViolatingPDB).MinAvailable("100%").Obj(),
				st.MakePDB().Name("non-violating-pdb").Namespace(v1.NamespaceDefault).MatchLabel(LabelKeyIsViolatingPDB, LabelValueNonViolatingPDB).MinAvailable("0").DisruptionsAllowed(math.MaxInt32).Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(onePodRes).Obj(),
				st.MakeNode().Name("node2").Capacity(onePodRes).Obj(),
			},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
				"node2": fwk.NewStatus(fwk.Unschedulable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			wantResult: framework.NewPostFilterResultWithNominatedNode("node2"),
			wantStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name: "preemption result filtered out by extenders",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Node("node1").Obj(),
				st.MakePod().Name("p2").UID("p2").Namespace(v1.NamespaceDefault).Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(onePodRes).Obj(),
				st.MakeNode().Name("node2").Capacity(onePodRes).Obj(),
			},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
				"node2": fwk.NewStatus(fwk.Unschedulable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			extender: &tf.FakeExtender{
				ExtenderName: "FakeExtender1",
				Predicates:   []tf.FitPredicate{tf.Node1PredicateExtender},
			},
			wantResult: framework.NewPostFilterResultWithNominatedNode("node1"),
			wantStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name: "no candidate nodes found, no enough resource after removing low priority pods",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(largeRes).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Node("node1").Obj(),
				st.MakePod().Name("p2").UID("p2").Namespace(v1.NamespaceDefault).Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(nodeRes).Obj(), // no enough CPU resource
				st.MakeNode().Name("node2").Capacity(nodeRes).Obj(), // no enough CPU resource
			},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
				"node2": fwk.NewStatus(fwk.Unschedulable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			wantResult: framework.NewPostFilterResultWithNominatedNode(""),
			wantStatus: fwk.NewStatus(fwk.Unschedulable, "preemption: 0/2 nodes are available: 2 Insufficient cpu."),
		},
		{
			name: "no candidate nodes found with mixed reasons, no lower priority pod and no enough CPU resource",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(largeRes).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Node("node1").Priority(highPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Namespace(v1.NamespaceDefault).Node("node2").Obj(),
				st.MakePod().Name("p3").UID("p3").Namespace(v1.NamespaceDefault).Node("node3").Priority(highPriority).Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(onePodRes).Obj(), // no pod will be preempted
				st.MakeNode().Name("node2").Capacity(nodeRes).Obj(),   // no enough CPU resource
				st.MakeNode().Name("node3").Capacity(onePodRes).Obj(), // no pod will be preempted
			},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
				"node2": fwk.NewStatus(fwk.Unschedulable),
				"node3": fwk.NewStatus(fwk.Unschedulable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			wantResult: framework.NewPostFilterResultWithNominatedNode(""),
			wantStatus: fwk.NewStatus(fwk.Unschedulable, "preemption: 0/3 nodes are available: 1 Insufficient cpu, 2 No preemption victims found for incoming pod."),
		},
		{
			name: "no candidate nodes found with mixed reason, 2 UnschedulableAndUnresolvable nodes and 2 nodes don't have enough CPU resource",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(largeRes).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Node("node1").Obj(),
				st.MakePod().Name("p2").UID("p2").Namespace(v1.NamespaceDefault).Node("node2").Obj(),
			},
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(nodeRes).Obj(),
				st.MakeNode().Name("node2").Capacity(nodeRes).Obj(),
				st.MakeNode().Name("node3").Capacity(nodeRes).Obj(),
				st.MakeNode().Name("node4").Capacity(nodeRes).Obj(),
			},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
				"node2": fwk.NewStatus(fwk.Unschedulable),
				"node4": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			wantResult: framework.NewPostFilterResultWithNominatedNode(""),
			wantStatus: fwk.NewStatus(fwk.Unschedulable, "preemption: 0/4 nodes are available: 2 Insufficient cpu, 2 Preemption is not helpful for scheduling."),
		},
		{
			name: "only one node but failed with TestPlugin",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(largeRes).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Node("node1").Obj(),
			},
			// label the node with key as "error" so that the TestPlugin will fail with error.
			nodes: []*v1.Node{st.MakeNode().Name("node1").Capacity(largeRes).Label("error", "true").Obj()},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			wantResult: nil,
			wantStatus: fwk.AsStatus(errors.New("preemption: running RemovePod on PreFilter plugin \"test-plugin\": failed to remove pod: p")),
		},
		{
			name: "one failed with TestPlugin and the other pass",
			pod:  st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(largeRes).Obj(),
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace(v1.NamespaceDefault).Node("node1").Obj(),
				st.MakePod().Name("p2").UID("p2").Namespace(v1.NamespaceDefault).Node("node2").Req(mediumRes).Obj(),
			},
			// even though node1 will fail with error but node2 will still be returned as a valid nominated node.
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(largeRes).Label("error", "true").Obj(),
				st.MakeNode().Name("node2").Capacity(largeRes).Obj(),
			},
			filteredNodesStatuses: framework.NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
				"node2": fwk.NewStatus(fwk.Unschedulable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			wantResult: framework.NewPostFilterResultWithNominatedNode("node2"),
			wantStatus: fwk.NewStatus(fwk.Success),
		},
	}

	for _, asyncAPICallsEnabled := range []bool{true, false} {
		for _, tt := range tests {
			t.Run(fmt.Sprintf("%s (Async API calls enabled: %v)", tt.name, asyncAPICallsEnabled), func(t *testing.T) {
				// index the potential victim pods in the fake client so that the victims deletion logic does not fail
				podItems := []v1.Pod{}
				for _, pod := range tt.pods {
					podItems = append(podItems, *pod)
				}
				cs := clientsetfake.NewClientset(&v1.PodList{Items: podItems})
				informerFactory := informers.NewSharedInformerFactory(cs, 0)
				podInformer := informerFactory.Core().V1().Pods().Informer()
				if err := podInformer.GetStore().Add(tt.pod); err != nil {
					t.Fatal(err)
				}
				for i := range tt.pods {
					if err := podInformer.GetStore().Add(tt.pods[i]); err != nil {
						t.Fatal(err)
					}
				}
				pdbInformer := informerFactory.Policy().V1().PodDisruptionBudgets().Informer()
				for i := range tt.pdbs {
					if err := pdbInformer.GetStore().Add(tt.pdbs[i]); err != nil {
						t.Fatal(err)
					}
				}

				// Register NodeResourceFit as the Filter & PreFilter plugin.
				registeredPlugins := []tf.RegisterPluginFunc{
					tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
					tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
					tf.RegisterPluginAsExtensions("test-plugin", newTestPlugin, "PreFilter"),
					tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				}
				var extenders []fwk.Extender
				if tt.extender != nil {
					extenders = append(extenders, tt.extender)
				}
				logger, ctx := ktesting.NewTestContext(t)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()
				var apiDispatcher *apidispatcher.APIDispatcher
				if asyncAPICallsEnabled {
					apiDispatcher = apidispatcher.New(cs, 16, apicalls.Relevances)
					apiDispatcher.Run(logger)
					defer apiDispatcher.Close()
				}

				f, err := tf.NewFramework(ctx, registeredPlugins, "",
					frameworkruntime.WithClientSet(cs),
					frameworkruntime.WithAPIDispatcher(apiDispatcher),
					frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
					frameworkruntime.WithInformerFactory(informerFactory),
					frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
					frameworkruntime.WithExtenders(extenders),
					frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot(tt.pods, tt.nodes)),
					frameworkruntime.WithLogger(logger),
					frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
				)
				if err != nil {
					t.Fatal(err)
				}
				if asyncAPICallsEnabled {
					cache := internalcache.New(ctx, apiDispatcher)
					f.SetAPICacher(apicache.New(nil, cache))
				}

				p, err := New(ctx, getDefaultDefaultPreemptionArgs(), f, feature.Features{})
				if err != nil {
					t.Fatal(err)
				}

				state := framework.NewCycleState()
				// Ensure <state> is populated.
				if _, status, _ := f.RunPreFilterPlugins(ctx, state, tt.pod); !status.IsSuccess() {
					t.Errorf("Unexpected PreFilter Status: %v", status)
				}

				gotResult, gotStatus := p.PostFilter(ctx, state, tt.pod, tt.filteredNodesStatuses)
				// As we cannot compare two errors directly due to miss the equal method for how to compare two errors, so just need to compare the reasons.
				if gotStatus.Code() == fwk.Error {
					if diff := cmp.Diff(tt.wantStatus.Reasons(), gotStatus.Reasons()); diff != "" {
						t.Errorf("Unexpected status (-want, +got):\n%s", diff)
					}
				} else {
					if diff := cmp.Diff(tt.wantStatus, gotStatus); diff != "" {
						t.Errorf("Unexpected status (-want, +got):\n%s", diff)
					}
				}
				if diff := cmp.Diff(tt.wantResult, gotResult); diff != "" {
					t.Errorf("Unexpected postFilterResult (-want, +got):\n%s", diff)
				}
			})
		}
	}
}

type candidate struct {
	victims *extenderv1.Victims
	name    string
}

func getMockCanPlacePodsFunc(blockingRules []blockingRule) CanPlacePodsFunc {
	mockCanPlacePods := func(ctx context.Context,
		state fwk.CycleState,
		pods []*v1.Pod,
		nodes []fwk.NodeInfo) *fwk.Status {
		// 1. Determine Goal: How many pods need to fit?
		neededSlots := len(pods)
		availableSlots := 0

		nodeMap := make(map[string]fwk.NodeInfo)
		for _, n := range nodes {
			nodeMap[n.Node().Name] = n
		}

		// 2. Evaluate Rules
		for _, rule := range blockingRules {
			node, exists := nodeMap[rule.nodeName]
			if !exists {
				continue // Node not in this snapshot, skip
			}

			// Check if ANY blocking victim is still on the node
			isBlocked := false
			for _, pod := range node.GetPods() {
				isBlocked = slices.Contains(rule.blockingVictims, pod.GetPod().Name)
				if isBlocked {
					break
				}
			}

			// If not blocked, this node contributes its capacity
			if !isBlocked {
				cap := rule.capacity
				if cap == 0 {
					cap = 1
				} // Default to 1 if not set
				availableSlots += cap
			}
		}

		// 3. Decision
		if availableSlots >= neededSlots {
			return fwk.NewStatus(fwk.Success)
		}

		msg := fmt.Sprintf("Need %d slots, found %d (Blocked)", neededSlots, availableSlots)
		return fwk.NewStatus(fwk.Unschedulable, msg)
	}
	return mockCanPlacePods
}

func TestDryRunPreemption(t *testing.T) {
	var (
		w1 = &v1.WorkloadReference{PodGroup: "pg1"}
	)

	tests := []struct {
		name                    string
		args                    *config.DefaultPreemptionArgs
		nodeNames               []string
		preemptors              []preemption.Preemptor
		initPods                []*v1.Pod
		registerPlugins         []tf.RegisterPluginFunc
		pdbs                    []*policy.PodDisruptionBudget
		fakeFilterRC            fwk.Code // return code for fake filter plugin
		disableParallelism      bool
		expected                [][]candidate
		expectedNumFilterCalled []int32
		blockingRules           []blockingRule
		workloadAwarePreemption bool
	}{
		{
			name: "a pod that does not fit on any node",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterFilterPlugin("FalseFilter", tf.NewFalseFilterPlugin),
			},
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
			},
			expected:                [][]candidate{{}},
			expectedNumFilterCalled: []int32{2},
		},
		{
			name: "a pod that fits with no preemption",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterFilterPlugin("TrueFilter", tf.NewTrueFilterPlugin),
			},
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
			},
			expected:                [][]candidate{{}},
			fakeFilterRC:            fwk.Unschedulable,
			expectedNumFilterCalled: []int32{2},
		},
		{
			name: "a pod that fits on one node with no preemption",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterFilterPlugin("MatchFilter", tf.NewMatchFilterPlugin),
			},
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("node1").UID("node1").Priority(highPriority).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
			},
			expected:                [][]candidate{{}},
			fakeFilterRC:            fwk.Unschedulable,
			expectedNumFilterCalled: []int32{2},
		},
		{
			name: "a pod that fits on both nodes when lower priority pods are preempted",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeRes).Obj(),
			},
			expected: [][]candidate{
				{
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeRes).Obj()},
						},
						name: "node1",
					},
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeRes).Obj()},
						},
						name: "node2",
					},
				},
			},
			expectedNumFilterCalled: []int32{8},
		},
		{
			name: "a pod that would fit on the nodes, but other pods running are higher priority, no preemption would happen",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(lowPriority).Req(largeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeRes).Obj(),
			},
			expected: [][]candidate{{}},
		},
		{
			name: "medium priority pod is preempted, but lower priority one stays as it is small",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeRes).Obj(),
			},
			expected: [][]candidate{
				{
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(largeRes).Obj(),
							},
						},
						name: "node1",
					},
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeRes).Obj()},
						},
						name: "node2",
					},
				},
			},
			expectedNumFilterCalled: []int32{9},
		},
		{
			name: "mixed priority pods are preempted",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(midPriority).Req(mediumRes).Obj(),
				st.MakePod().Name("p1.4").UID("p1.4").Node("node1").Priority(highPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(highPriority).Req(largeRes).Obj(),
			},
			expected: [][]candidate{
				{
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
								st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(midPriority).Req(mediumRes).Obj(),
							},
						},
						name: "node1",
					},
				},
			},
			expectedNumFilterCalled: []int32{7},
		},
		{
			name: "mixed priority pods are preempted, pick later StartTime one when priorities are equal",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime5).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime4).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(midPriority).Req(mediumRes).StartTime(epochTime3).Obj(),
				st.MakePod().Name("p1.4").UID("p1.4").Node("node1").Priority(highPriority).Req(smallRes).StartTime(epochTime2).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(highPriority).Req(largeRes).StartTime(epochTime1).Obj(),
			},
			expected: [][]candidate{
				{
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime5).Obj(),
								st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(midPriority).Req(mediumRes).StartTime(epochTime3).Obj(),
							},
						},
						name: "node1",
					},
				},
			},
			expectedNumFilterCalled: []int32{7}, // no preemption would happen on node2 and no filter call is counted.
		},
		{
			name: "pod with anti-affinity is preempted",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
				tf.RegisterPluginAsExtensions(interpodaffinity.Name, frameworkruntime.FactoryAdapter(feature.Features{}, interpodaffinity.New), "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Label("foo", "").Priority(highPriority).Req(smallRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("foo", "").Priority(lowPriority).Req(smallRes).
					PodAntiAffinityExists("foo", "hostname", st.PodAntiAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(highPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(highPriority).Req(smallRes).Obj(),
			},
			expected: [][]candidate{
				{
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("foo", "").Priority(lowPriority).Req(smallRes).
									PodAntiAffinityExists("foo", "hostname", st.PodAntiAffinityWithRequiredReq).Obj(),
							},
						},
						name: "node1",
					},
				},
			},
			expectedNumFilterCalled: []int32{4}, // no preemption would happen on node2 and no filter call is counted.
		},
		{
			name: "preemption to resolve pod topology spread filter failure",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(podtopologyspread.Name, podTopologySpreadFunc, "PreFilter", "Filter"),
			},
			nodeNames: []string{"node-a/zone1", "node-b/zone1", "node-x/zone2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Label("foo", "").Priority(highPriority).
					SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, nil).
					SpreadConstraint(1, "hostname", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, nil).
					Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("pod-a1").UID("pod-a1").Node("node-a").Label("foo", "").Priority(midPriority).Obj(),
				st.MakePod().Name("pod-a2").UID("pod-a2").Node("node-a").Label("foo", "").Priority(lowPriority).Obj(),
				st.MakePod().Name("pod-b1").UID("pod-b1").Node("node-b").Label("foo", "").Priority(lowPriority).Obj(),
				st.MakePod().Name("pod-x1").UID("pod-x1").Node("node-x").Label("foo", "").Priority(highPriority).Obj(),
				st.MakePod().Name("pod-x2").UID("pod-x2").Node("node-x").Label("foo", "").Priority(highPriority).Obj(),
			},
			expected: [][]candidate{
				{
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("pod-a2").UID("pod-a2").Node("node-a").Label("foo", "").Priority(lowPriority).Obj()},
						},
						name: "node-a",
					},
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("pod-b1").UID("pod-b1").Node("node-b").Label("foo", "").Priority(lowPriority).Obj()},
						},
						name: "node-b",
					},
				},
			},
			expectedNumFilterCalled: []int32{8},
		},
		{
			name: "get Unschedulable in the preemption phase when the filter plugins filtering the nodes",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeRes).Obj(),
			},
			fakeFilterRC:            fwk.Unschedulable,
			expected:                [][]candidate{{}},
			expectedNumFilterCalled: []int32{2},
		},
		{
			name: "preemption with violation of same pdb",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 1},
				},
			},
			expected: [][]candidate{
				{
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
								st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
							},
							NumPDBViolations: 1,
						},
						name: "node1",
					},
				},
			},
			expectedNumFilterCalled: []int32{6},
		},
		{
			name: "preemption with violation of the pdb with pod whose eviction was processed, the victim doesn't belong to DisruptedPods",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 1, DisruptedPods: map[string]metav1.Time{"p2": {Time: time.Now()}}},
				},
			},
			expected: [][]candidate{
				{
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
								st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
							},
							NumPDBViolations: 1,
						},
						name: "node1",
					},
				},
			},
			expectedNumFilterCalled: []int32{6},
		},
		{
			name: "preemption with violation of the pdb with pod whose eviction was processed, the victim belongs to DisruptedPods",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 1, DisruptedPods: map[string]metav1.Time{"p1.2": {Time: time.Now()}}},
				},
			},
			expected: [][]candidate{
				{
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
								st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
							},
							NumPDBViolations: 0,
						},
						name: "node1",
					},
				},
			},
			expectedNumFilterCalled: []int32{5},
		},
		{
			name: "preemption with violation of the pdb with pod whose eviction was processed, the victim which belongs to DisruptedPods is treated as 'nonViolating'",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 1, DisruptedPods: map[string]metav1.Time{"p1.3": {Time: time.Now()}}},
				},
			},
			expected: [][]candidate{
				{
					candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
								st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
								st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Label("app", "foo").Priority(midPriority).Req(mediumRes).Obj(),
							},
							NumPDBViolations: 1,
						},
						name: "node1",
					},
				},
			},
			expectedNumFilterCalled: []int32{7},
		},
		{
			name: "all nodes are possible candidates, but DefaultPreemptionArgs limits to 2",
			args: &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 40, MinCandidateNodesAbsolute: 1},
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2", "node3", "node4", "node5"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p3").UID("p3").Node("node3").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p4").UID("p4").Node("node4").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p5").UID("p5").Node("node5").Priority(midPriority).Req(largeRes).Obj(),
			},
			disableParallelism: true,
			expected: [][]candidate{
				{
					// cycle=0 => offset=4 => node5 (yes), node1 (yes)
					candidate{
						name: "node1",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
					candidate{
						name: "node5",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p5").UID("p5").Node("node5").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
				},
			},
			expectedNumFilterCalled: []int32{8},
		},
		{
			name: "some nodes are not possible candidates, DefaultPreemptionArgs limits to 2",
			args: &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 40, MinCandidateNodesAbsolute: 1},
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2", "node3", "node4", "node5"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(veryHighPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p3").UID("p3").Node("node3").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p4").UID("p4").Node("node4").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p5").UID("p5").Node("node5").Priority(veryHighPriority).Req(largeRes).Obj(),
			},
			disableParallelism: true,
			expected: [][]candidate{
				{
					// cycle=0 => offset=4 => node5 (no), node1 (yes), node2 (no), node3 (yes)
					candidate{
						name: "node1",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
					candidate{
						name: "node3",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p3").UID("p3").Node("node3").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
				},
			},
			expectedNumFilterCalled: []int32{8},
		},
		{
			name: "preemption offset across multiple scheduling cycles and wrap around",
			args: &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 40, MinCandidateNodesAbsolute: 1},
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2", "node3", "node4", "node5"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("tp1").UID("tp1").Priority(highPriority).Req(largeRes).Obj()),
				preemption.NewPodPreemptor(st.MakePod().Name("tp2").UID("tp2").Priority(highPriority).Req(largeRes).Obj()),
				preemption.NewPodPreemptor(st.MakePod().Name("tp3").UID("tp3").Priority(highPriority).Req(largeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p3").UID("p3").Node("node3").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p4").UID("p4").Node("node4").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p5").UID("p5").Node("node5").Priority(midPriority).Req(largeRes).Obj(),
			},
			disableParallelism: true,
			expected: [][]candidate{
				{
					// cycle=0 => offset=4 => node5 (yes), node1 (yes)
					candidate{
						name: "node1",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
					candidate{
						name: "node5",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p5").UID("p5").Node("node5").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
				},
				{
					// cycle=1 => offset=1 => node2 (yes), node3 (yes)
					candidate{
						name: "node2",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
					candidate{
						name: "node3",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p3").UID("p3").Node("node3").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
				},
				{
					// cycle=2 => offset=3 => node4 (yes), node5 (yes)
					candidate{
						name: "node4",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p4").UID("p4").Node("node4").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
					candidate{
						name: "node5",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p5").UID("p5").Node("node5").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
				},
			},
			expectedNumFilterCalled: []int32{8, 8, 8},
		},
		{
			name: "preemption looks past numCandidates until a non-PDB violating node is found",
			args: &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 40, MinCandidateNodesAbsolute: 2},
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2", "node3", "node4", "node5"},
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Label("app", "foo").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Label("app", "foo").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p3").UID("p3").Node("node3").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p4").UID("p4").Node("node4").Priority(midPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p5").UID("p5").Node("node5").Label("app", "foo").Priority(midPriority).Req(largeRes).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
				},
			},
			disableParallelism: true,
			expected: [][]candidate{
				{
					// Even though the DefaultPreemptionArgs constraints suggest that the
					// minimum number of candidates is 2, we get three candidates here
					// because we're okay with being a little over (in production, if a
					// non-PDB violating candidate isn't found close to the offset, the
					// number of additional candidates returned will be at most
					// approximately equal to the parallelism in dryRunPreemption).
					// cycle=0 => offset=4 => node5 (yes, pdb), node1 (yes, pdb), node2 (no, pdb), node3 (yes)
					candidate{
						name: "node1",
						victims: &extenderv1.Victims{
							Pods:             []*v1.Pod{st.MakePod().Name("p1").UID("p1").Node("node1").Label("app", "foo").Priority(midPriority).Req(largeRes).Obj()},
							NumPDBViolations: 1,
						},
					},
					candidate{
						name: "node3",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{st.MakePod().Name("p3").UID("p3").Node("node3").Priority(midPriority).Req(largeRes).Obj()},
						},
					},
					candidate{
						name: "node5",
						victims: &extenderv1.Victims{
							Pods:             []*v1.Pod{st.MakePod().Name("p5").UID("p5").Node("node5").Label("app", "foo").Priority(midPriority).Req(largeRes).Obj()},
							NumPDBViolations: 1,
						},
					},
				},
			},
			expectedNumFilterCalled: []int32{13},
		},
		{
			name: "gang of 2 pods fits by preempting victims on two different nodes",
			registerPlugins: []tf.RegisterPluginFunc{
				tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			},
			nodeNames: []string{"node1", "node2"},
			// The Gang: 2 High Prio pods, each requiring Large Resources
			preemptors: []preemption.Preemptor{
				newPodGroupPreemptor(highPriority,
					[]*v1.Pod{
						st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
						st.MakePod().Name("gang-p2").UID("gang-p2").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
					},
					nil,
				),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim-n1").UID("victim-n1").Node("node1").Priority(lowPriority).Req(largeRes).Obj(),
				st.MakePod().Name("victim-n2").UID("victim-n2").Node("node2").Priority(lowPriority).Req(largeRes).Obj(),
			},
			// Expectation: Both nodes are returned as candidates with their respective victims
			expected: [][]candidate{
				{
					candidate{
						name: "Cluster-Scope-pg1",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("victim-n1").UID("victim-n1").Node("node1").Priority(lowPriority).Req(largeRes).Obj(),
								st.MakePod().Name("victim-n2").UID("victim-n2").Node("node2").Priority(lowPriority).Req(largeRes).Obj(),
							},
						},
					},
				},
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"victim-n1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"victim-n2"}, capacity: 1},
			},
			workloadAwarePreemption: true,
		},
		{
			name:      "gang of 2 pods fails because one member is blocked by non-preemptible pod",
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				newPodGroupPreemptor(highPriority,
					[]*v1.Pod{
						st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
						st.MakePod().Name("gang-p2").UID("gang-p2").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
					},
					nil,
				),
			},
			initPods: []*v1.Pod{
				// Node 1: Valid victim (Low Priority)
				st.MakePod().Name("victim-n1").UID("victim-n1").Node("node1").Priority(lowPriority).Req(largeRes).Obj(),
				// Node 2: Invalid victim (Very High Priority - cannot be preempted)
				st.MakePod().Name("blocker-n2").UID("blocker-n2").Node("node2").Priority(veryHighPriority).Req(largeRes).Obj(),
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"victim-n1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"blocker-n2"}, capacity: 1},
			},
			// Expected: Empty (Failure). Gang needs 2 slots, only found 1 feasible slot.
			expected: [][]candidate{{}},
		},
		{
			name:      "gang of 2 pods fits on a SINGLE node by preempting multiple victims there",
			nodeNames: []string{"node1"},
			preemptors: []preemption.Preemptor{
				newPodGroupPreemptor(highPriority,
					[]*v1.Pod{
						st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(smallRes).Obj(),
						st.MakePod().Name("gang-p2").UID("gang-p2").WorkloadRef(w1).Priority(highPriority).Req(smallRes).Obj(),
					},
					nil,
				),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim-a").UID("victim-a").Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("victim-b").UID("victim-b").Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
			},
			blockingRules: []blockingRule{
				// Key Rule: Removing BOTH victims unlocks capacity 2
				{nodeName: "node1", blockingVictims: []string{"victim-a", "victim-b"}, capacity: 2},
			},
			expected: [][]candidate{
				{
					candidate{
						name: "Cluster-Scope-pg1",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("victim-a").UID("victim-a").Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
								st.MakePod().Name("victim-b").UID("victim-b").Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
							},
						},
					},
				},
			},
			workloadAwarePreemption: true,
		},
		{
			name:      "gang of 2 pods requires preempting a Medium priority victim because Low alone is insufficient",
			nodeNames: []string{"node1", "node2"},
			preemptors: []preemption.Preemptor{
				newPodGroupPreemptor(highPriority,
					[]*v1.Pod{
						st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
						st.MakePod().Name("gang-p2").UID("gang-p2").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
					},
					nil,
				),
			},
			initPods: []*v1.Pod{
				// Node 1: Low Priority (Will be chosen first)
				st.MakePod().Name("low-p-victim").UID("low-p-victim").Node("node1").Priority(lowPriority).Req(largeRes).Obj(),
				// Node 2: Mid Priority (Must also be chosen to reach capacity 2)
				st.MakePod().Name("mid-p-victim").UID("mid-p-victim").Node("node2").Priority(midPriority).Req(largeRes).Obj(),
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"low-p-victim"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"mid-p-victim"}, capacity: 1},
			},
			workloadAwarePreemption: true,
			expected: [][]candidate{
				{
					candidate{
						name: "Cluster-Scope-pg1",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("low-p-victim").UID("low-p-victim").Node("node1").Priority(lowPriority).Req(largeRes).Obj(),
								st.MakePod().Name("mid-p-victim").UID("mid-p-victim").Node("node2").Priority(midPriority).Req(largeRes).Obj(),
							},
						},
					},
				},
			},
		},
		{
			name:      "single pod preemptor triggers atomic preemption of a distributed workload victim",
			nodeNames: []string{"node1", "node2"},

			// Preemptor: Single High Priority Pod
			// It only needs to find ONE node to run on.
			preemptors: []preemption.Preemptor{
				preemption.NewPodPreemptor(
					st.MakePod().Name("preemptor").UID("preemptor").Priority(highPriority).Req(largeRes).Obj(),
				),
			},

			// Init: Workload w1 split across two nodes
			initPods: []*v1.Pod{
				// Pod 1 on Node 1 (Direct Blocker)
				st.MakePod().Name("w1-p1").UID("w1-p1").Node("node1").WorkloadRef(w1).Priority(lowPriority).Req(smallRes).Obj(),
				// Pod 2 on Node 2 (Collateral Damage)
				st.MakePod().Name("w1-p2").UID("w1-p2").Node("node2").WorkloadRef(w1).Priority(lowPriority).Req(smallRes).Obj(),
			},

			// Rules:
			// - If checking Node 1: "w1-p1" blocks it. Removing it (and its gang) opens the node.
			// - If checking Node 2: "w1-p2" blocks it. Removing it (and its gang) opens the node.
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"w1-p1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"w1-p2"}, capacity: 1},
			},
			workloadAwarePreemption: true,
			expected: [][]candidate{
				{
					// Scenario A: Scheduler tries to fit preemptor on Node 1
					candidate{
						name: "node1",
						victims: &extenderv1.Victims{
							// Result: MUST return BOTH pods in the victim list.
							// Even though w1-p2 is on a different node, it is part of the atomic unit.
							Pods: []*v1.Pod{
								st.MakePod().Name("w1-p1").UID("w1-p1").Node("node1").WorkloadRef(w1).Priority(lowPriority).Req(smallRes).Obj(),
								st.MakePod().Name("w1-p2").UID("w1-p2").Node("node2").WorkloadRef(w1).Priority(lowPriority).Req(smallRes).Obj(),
							},
						},
					},
					// Scenario B: Scheduler tries to fit preemptor on Node 2
					// (Both are valid candidates since removing the gang clears both)
					candidate{
						name: "node2",
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("w1-p1").UID("w1-p1").Node("node1").WorkloadRef(w1).Priority(lowPriority).Req(smallRes).Obj(),
								st.MakePod().Name("w1-p2").UID("w1-p2").Node("node2").WorkloadRef(w1).Priority(lowPriority).Req(smallRes).Obj(),
							},
						},
					},
				},
			},
		},
	}

	labelKeys := []string{"hostname", "zone", "region"}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodes := make([]*v1.Node, len(tt.nodeNames))
			fakeFilterRCMap := make(map[string]fwk.Code, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodeWrapper := st.MakeNode().Capacity(veryLargeRes)
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
			snapshot := internalcache.NewSnapshot(tt.initPods, nodes)

			// For each test, register a FakeFilterPlugin along with essential plugins and tt.registerPlugins.
			fakePlugin := tf.FakeFilterPlugin{
				FailedNodeReturnCodeMap: fakeFilterRCMap,
			}
			registeredPlugins := append([]tf.RegisterPluginFunc{
				tf.RegisterFilterPlugin(
					"FakeFilter",
					func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
						return &fakePlugin, nil
					},
				)},
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			registeredPlugins = append(registeredPlugins, tt.registerPlugins...)
			var preemptorPods []*v1.Pod
			for _, pod := range tt.preemptors {
				preemptorPods = append(preemptorPods, pod.Members()...)
			}

			var objs []runtime.Object
			for _, p := range append(preemptorPods, tt.initPods...) {
				objs = append(objs, p)
			}
			for _, n := range nodes {
				objs = append(objs, n)
			}
			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(objs...), 0)
			parallelism := parallelize.DefaultParallelism
			if tt.disableParallelism {
				// We need disableParallelism because of the non-deterministic nature
				// of the results of tests that set custom minCandidateNodesPercentage
				// or minCandidateNodesAbsolute. This is only done in a handful of tests.
				parallelism = 1
			}

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			testingFwk, err := tf.NewFramework(
				ctx,
				registeredPlugins, "",
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithParallelism(parallelism),
				frameworkruntime.WithLogger(logger),
			)
			if err != nil {
				t.Fatal(err)
			}

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			nodeInfos, err := snapshot.NodeInfos().List()
			if err != nil {
				t.Fatal(err)
			}
			sort.Slice(nodeInfos, func(i, j int) bool {
				return nodeInfos[i].Node().Name < nodeInfos[j].Node().Name
			})

			if tt.args == nil {
				tt.args = getDefaultDefaultPreemptionArgs()
			}
			pl, err := New(ctx, tt.args, testingFwk, feature.Features{
				EnableWorkloadAwarePreemption: tt.workloadAwarePreemption,
			})
			if err != nil {
				t.Fatal(err)
			}

			// Using 4 as a seed source to test getOffsetAndNumCandidates() deterministically.
			// However, we need to do it after informerFactory.WaitforCacheSync() which might
			// set a seed.
			getOffsetRand = rand.New(rand.NewSource(4)).Int31n
			var prevNumFilterCalled int32
			for cycle, preemptor := range tt.preemptors {
				state := framework.NewCycleState()
				for _, pod := range preemptor.Members() {
					// Some tests rely on PreFilter plugin to compute its CycleState.
					if _, status, _ := testingFwk.RunPreFilterPlugins(ctx, state, pod); !status.IsSuccess() {
						t.Errorf("cycle %d: Unexpected PreFilter Status: %v", cycle, status)
					}
				}

				offset, numCandidates := pl.GetOffsetAndNumCandidates(int32(len(nodeInfos)))

				domains := pl.Evaluator.NewDomains(preemptor, nodeInfos)

				if tt.blockingRules != nil {
					pl.CanPlacePods = getMockCanPlacePodsFunc(tt.blockingRules)
				}

				got, _, _ := pl.Evaluator.DryRunPreemption(ctx, state, preemptor, domains, tt.pdbs, offset, numCandidates)
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
				candidates := []candidate{}
				for i := range got {
					candidates = append(candidates, candidate{victims: got[i].Victims(), name: got[i].Name()})
				}
				if tt.expectedNumFilterCalled != nil && fakePlugin.NumFilterCalled-prevNumFilterCalled != tt.expectedNumFilterCalled[cycle] {
					t.Errorf("cycle %d: got NumFilterCalled=%d, want %d", cycle, fakePlugin.NumFilterCalled-prevNumFilterCalled, tt.expectedNumFilterCalled[cycle])
				}
				prevNumFilterCalled = fakePlugin.NumFilterCalled
				if diff := cmp.Diff(tt.expected[cycle], candidates, cmp.AllowUnexported(candidate{})); diff != "" {
					t.Errorf("cycle %d: unexpected candidates (-want, +got): %s", cycle, diff)
				}
			}
		})
	}
}

func TestSelectBestCandidate(t *testing.T) {
	var (
		w1 = &v1.WorkloadReference{PodGroup: "pg1"}
		w2 = &v1.WorkloadReference{PodGroup: "pg2"}
		w3 = &v1.WorkloadReference{PodGroup: "pg3"}
	)
	tests := []struct {
		name                    string
		registerPlugin          tf.RegisterPluginFunc
		nodeNames               []string
		preemptor               preemption.Preemptor
		pods                    []*v1.Pod
		expected                []string
		blockingRules           []blockingRule
		workloadAwarePreemption bool
	}{
		{
			name:           "node with min highest priority pod is picked",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2", "node3"},
			preemptor:      preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(lowPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(lowPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(lowPriority).Req(mediumRes).StartTime(epochTime).Obj(),
			},
			expected: []string{"node3"},
		},
		{
			name:           "when highest priorities are the same, minimum sum of priorities is picked",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2", "node3"},
			preemptor:      preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(lowPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(midPriority).Req(mediumRes).StartTime(epochTime).Obj(),
			},
			expected: []string{"node2"},
		},
		{
			name:           "when highest priority and sum are the same, minimum number of pods is picked",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2", "node3"},
			preemptor:      preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(negPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(midPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.4").UID("p1.4").Node("node1").Priority(negPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(negPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(negPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.3").UID("p3.3").Node("node3").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
			},
			expected: []string{"node2"},
		},
		{
			// pickOneNodeForPreemption adjusts pod priorities when finding the sum of the victims. This
			// test ensures that the logic works correctly.
			name:           "sum of adjusted priorities is considered",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2", "node3"},
			preemptor:      preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(negPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(negPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(negPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(negPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.3").UID("p3.3").Node("node3").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
			},
			expected: []string{"node2"},
		},
		{
			name:           "non-overlapping lowest high priority, sum priorities, and number of pods",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2", "node3", "node4"},
			preemptor:      preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(veryHighPriority).Req(veryLargeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.3").UID("p1.3").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(highPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.3").UID("p3.3").Node("node3").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p3.4").UID("p3.4").Node("node3").Priority(lowPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p4.1").UID("p4.1").Node("node4").Priority(midPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p4.2").UID("p4.2").Node("node4").Priority(midPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p4.3").UID("p4.3").Node("node4").Priority(midPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p4.4").UID("p4.4").Node("node4").Priority(negPriority).Req(smallRes).StartTime(epochTime).Obj(),
			},
			expected: []string{"node1"},
		},
		{
			name:           "same priority, same number of victims, different start time for each node's pod",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2", "node3"},
			preemptor:      preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(mediumRes).StartTime(epochTime2).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(mediumRes).StartTime(epochTime2).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(mediumRes).StartTime(epochTime3).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(midPriority).Req(mediumRes).StartTime(epochTime3).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumRes).StartTime(epochTime1).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(midPriority).Req(mediumRes).StartTime(epochTime1).Obj(),
			},
			expected: []string{"node2"},
		},
		{
			name:           "same priority, same number of victims, different start time for all pods",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2", "node3"},
			preemptor:      preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Req(mediumRes).StartTime(epochTime4).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(mediumRes).StartTime(epochTime2).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(mediumRes).StartTime(epochTime5).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(midPriority).Req(mediumRes).StartTime(epochTime1).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumRes).StartTime(epochTime3).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(midPriority).Req(mediumRes).StartTime(epochTime6).Obj(),
			},
			expected: []string{"node3"},
		},
		{
			name:           "different priority, same number of victims, different start time for all pods",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2", "node3"},
			preemptor:      preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(lowPriority).Req(mediumRes).StartTime(epochTime4).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(midPriority).Req(mediumRes).StartTime(epochTime2).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Req(mediumRes).StartTime(epochTime6).Obj(),
				st.MakePod().Name("p2.2").UID("p2.2").Node("node2").Priority(lowPriority).Req(mediumRes).StartTime(epochTime1).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(lowPriority).Req(mediumRes).StartTime(epochTime3).Obj(),
				st.MakePod().Name("p3.2").UID("p3.2").Node("node3").Priority(midPriority).Req(mediumRes).StartTime(epochTime5).Obj(),
			},
			expected: []string{"node2"},
		},
		{
			name:           "gang preemptor prefers node with lower priority victim (Workload vs Workload)",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2"},
			// Preemptor: Gang of 2 High Priority Pods
			preemptor: newPodGroupPreemptor(highPriority,
				[]*v1.Pod{
					st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
					st.MakePod().Name("gang-p2").UID("gang-p2").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
				}, nil),
			pods: []*v1.Pod{
				// Node 1: Occupied by Mid-Priority Workload (w2)
				st.MakePod().Name("w2-p1").UID("w2-p1").Node("node1").WorkloadRef(w2).Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("w2-p2").UID("w2-p2").Node("node1").WorkloadRef(w2).Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				// Node 2: Occupied by Low-Priority Workload (w3)
				st.MakePod().Name("w3-p1").UID("w3-p1").Node("node2").WorkloadRef(w3).Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("w3-p2").UID("w3-p2").Node("node2").WorkloadRef(w3).Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"w2-p1", "w2-p2"}, capacity: 2},
				{nodeName: "node2", blockingVictims: []string{"w3-p1", "w3-p2"}, capacity: 2},
			},
			workloadAwarePreemption: true,
			expected:                []string{"Cluster-Scope-pg1"},
		},
		{
			name:           "gang preemptor prefers node with FEWER victims (when priorities equal)",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2"},
			preemptor: newPodGroupPreemptor(highPriority,
				[]*v1.Pod{
					st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
				}, nil),
			pods: []*v1.Pod{
				// Node 1: Two small pods (Same Low Priority)
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
				// Node 2: One larger pod (Same Low Priority)
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},

			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"p1.1", "p1.2"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"p2.1"}, capacity: 1},
			},
			workloadAwarePreemption: true,
			expected:                []string{"Cluster-Scope-pg1"},
		},
		{
			name:           "gang preemptor prefers node with NEWER victims (break tie with StartTime)",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2"},
			preemptor: newPodGroupPreemptor(highPriority,
				[]*v1.Pod{
					st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
				}, nil),

			pods: []*v1.Pod{
				// Node 1: Old Workload (Running for a long time) - epochTime1 is older
				st.MakePod().Name("old-p1").UID("old-p1").Node("node1").Priority(lowPriority).Req(largeRes).StartTime(epochTime1).Obj(),
				// Node 2: New Workload (Just started) - epochTime2 is newer
				st.MakePod().Name("new-p1").UID("new-p1").Node("node2").Priority(lowPriority).Req(largeRes).StartTime(epochTime2).Obj(),
			},

			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"old-p1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"new-p1"}, capacity: 1},
			},
			workloadAwarePreemption: true,
			expected:                []string{"Cluster-Scope-pg1"},
		},
		{
			name:           "distributed gang victim: selection accounts for TOTAL victims across cluster (Side-Effect Check)",
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			nodeNames:      []string{"node1", "node2", "node3"},
			// Preemptor: Single Pod needs Node1 or Node2
			preemptor: preemption.NewPodPreemptor(
				st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj(),
			),
			pods: []*v1.Pod{
				// Victim A (Node 1): Single standalone pod
				st.MakePod().Name("standalone").UID("standalone").Node("node1").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),

				// Victim B (Node 2): Member of atomic workload w2. Peer is on Node 3 (simulated impact).
				st.MakePod().Name("w2-p1").UID("w2-p1").Node("node2").WorkloadRef(w2).Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("w2-p2").UID("w2-p2").Node("node3").WorkloadRef(w2).Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			// Node 1 cost: 1 pod.
			// Node 2 cost: 2 pods (w2-p1 + w2-p2).
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"standalone"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"w2-p1", "w2-p2"}, capacity: 1},
			},
			// EXPECTED: Pick Node1 (1 Victim) over Node2 (2 Victims).
			expected: []string{"node1"},

			workloadAwarePreemption: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			getOffsetRand = rand.New(rand.NewSource(4)).Int31n
			nodes := make([]*v1.Node, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodes[i] = st.MakeNode().Name(nodeName).Capacity(veryLargeRes).Obj()
			}

			var objs []runtime.Object
			var preemptorPods []*v1.Pod
			preemptorPods = append(preemptorPods, tt.preemptor.Members()...)
			for _, pod := range preemptorPods {
				objs = append(objs, pod)
			}

			for _, pod := range tt.pods {
				objs = append(objs, pod)
			}
			cs := clientsetfake.NewClientset(objs...)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			snapshot := internalcache.NewSnapshot(tt.pods, nodes)
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fwk, err := tf.NewFramework(
				ctx,
				[]tf.RegisterPluginFunc{
					tt.registerPlugin,
					tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
					tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				},
				"",
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithLogger(logger),
			)
			if err != nil {
				t.Fatal(err)
			}
			nodeInfos, err := fwk.SnapshotSharedLister().NodeInfos().List()
			if err != nil {
				t.Fatal(err)
			}

			state := framework.NewCycleState()

			for _, pod := range preemptorPods {
				// Some tests rely on PreFilter plugin to compute its CycleState.
				if _, status, _ := fwk.RunPreFilterPlugins(ctx, state, pod); !status.IsSuccess() {
					t.Errorf("Unexpected PreFilter Status: %v", status)
				}
			}

			pl, err := New(ctx, getDefaultDefaultPreemptionArgs(), fwk, feature.Features{
				EnableWorkloadAwarePreemption: tt.workloadAwarePreemption,
			})
			if err != nil {
				t.Fatal(err)
			}
			if tt.blockingRules != nil {
				pl.CanPlacePods = getMockCanPlacePodsFunc(tt.blockingRules)
			}

			offset, numCandidates := pl.GetOffsetAndNumCandidates(int32(len(nodeInfos)))
			domains := pl.Evaluator.NewDomains(tt.preemptor, nodeInfos)
			candidates, _, _ := pl.Evaluator.DryRunPreemption(ctx, state, tt.preemptor, domains, nil, offset, numCandidates)
			s := pl.Evaluator.SelectCandidate(ctx, candidates)
			if s == nil || len(s.Name()) == 0 {
				t.Fatalf("expected any node in %v, but candidate is missing", tt.expected)
			}
			found := false
			for _, nodeName := range tt.expected {
				if nodeName == s.Name() {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("expect any domain in %v, but got %v", tt.expected, s.Name())
			}
		})
	}
}

func TestCustomSelection(t *testing.T) {
	var (
		w1 = &v1.WorkloadReference{PodGroup: "pg1"}
	)

	victimLabelsAreEligible := func(key, val string) IsEligiblePreemptorFunc {
		return func(domain preemption.Domain, victim preemption.PreemptionUnit, preemptor preemption.Preemptor) bool {
			for _, pod := range victim.Pods() {
				pval, ok := pod.GetPod().Labels[key]
				if !ok {
					return false
				}
				if pval != val {
					return false
				}
			}
			return true
		}
	}
	domainNodeNamesAreEligible := func(names []string) IsEligiblePreemptorFunc {
		containsName := make(map[string]bool, len(names))
		for _, name := range names {
			containsName[name] = true
		}
		return func(domain preemption.Domain, victim preemption.PreemptionUnit, preemptor preemption.Preemptor) bool {
			for name := range victim.AffectedNodes() {
				if !containsName[name] {
					return false
				}
			}
			return true
		}
	}
	priorityBelowThresholdCannotPreempt := func(minPreempting int32) IsEligiblePreemptorFunc {
		return func(domain preemption.Domain, victim preemption.PreemptionUnit, preemptor preemption.Preemptor) bool {
			for _, pod := range preemptor.Members() {
				if corev1helpers.PodPriority(pod) < minPreempting {
					return false
				}
			}
			return true
		}
	}
	priorityAboveThresholdCannotBePreempted := func(maxPreemptible int32) IsEligiblePreemptorFunc {
		return func(domain preemption.Domain, victim preemption.PreemptionUnit, preemptor preemption.Preemptor) bool {
			for _, pod := range victim.Pods() {
				if corev1helpers.PodPriority(pod.GetPod()) > maxPreemptible {
					return false
				}
			}
			return true
		}
	}

	tests := []struct {
		name                    string
		eligiblePreemptor       IsEligiblePreemptorFunc
		nodeNames               []string
		preemptor               preemption.Preemptor
		pods                    []*v1.Pod
		expected                map[string][]string
		blockingRules           []blockingRule
		workloadAwarePreemption bool
	}{
		{
			name:              "filter for matching pod label: high priority",
			eligiblePreemptor: victimLabelsAreEligible("preemptible", "yes"),
			nodeNames:         []string{"node1", "node2", "node3", "node4"},
			preemptor:         preemption.NewPodPreemptor(st.MakePod().Name("p1").UID("p1").Priority(highPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Label("preemptible", "no").Node("node1").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Label("preemptible", "yes").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Label("preemptible", "yes").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node4").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			expected: map[string][]string{"node2": {"v2"}, "node3": {"v3"}},
		},
		{
			name:              "filter for matching pod label: mid priority",
			eligiblePreemptor: victimLabelsAreEligible("preemptible", "yes"),
			nodeNames:         []string{"node1", "node2", "node3", "node4"},
			preemptor:         preemption.NewPodPreemptor(st.MakePod().Name("p2").UID("p2").Priority(midPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Label("preemptible", "no").Node("node1").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Label("preemptible", "yes").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Label("preemptible", "yes").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node4").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			expected: map[string][]string{"node3": {"v3"}},
		},
		{
			name:              "filter for matching pod label: low priority",
			eligiblePreemptor: victimLabelsAreEligible("preemptible", "yes"),
			nodeNames:         []string{"node1", "node2", "node3", "node4"},
			preemptor:         preemption.NewPodPreemptor(st.MakePod().Name("p3").UID("p3").Priority(lowPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Label("preemptible", "no").Node("node1").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Label("preemptible", "yes").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Label("preemptible", "yes").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node4").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			expected: map[string][]string{},
		},
		{
			name:              "filter for matching victim node: high priority",
			eligiblePreemptor: domainNodeNamesAreEligible([]string{"node1"}),
			nodeNames:         []string{"node1", "node2", "node3"},
			preemptor:         preemption.NewPodPreemptor(st.MakePod().Name("p3").UID("p3").Priority(highPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(highPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node1").Priority(midPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v5").UID("v5").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			expected: map[string][]string{"node1": {"v2", "v3"}},
		},
		{
			name:              "filter for matching victim node: mid priority",
			eligiblePreemptor: domainNodeNamesAreEligible([]string{"node1"}),
			nodeNames:         []string{"node1", "node2", "node3"},
			preemptor:         preemption.NewPodPreemptor(st.MakePod().Name("p3").UID("p3").Priority(midPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(highPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node1").Priority(midPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v5").UID("v5").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			expected: map[string][]string{"node1": {"v3"}},
		},
		{
			name:              "filter for matching victim node: low priority",
			eligiblePreemptor: domainNodeNamesAreEligible([]string{"node1"}),
			nodeNames:         []string{"node1", "node2", "node3"},
			preemptor:         preemption.NewPodPreemptor(st.MakePod().Name("p3").UID("p3").Priority(lowPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(highPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node1").Priority(midPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v5").UID("v5").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			expected: map[string][]string{},
		},
		{
			name:              "only pods at or above specified priority can preempted: high priority",
			eligiblePreemptor: priorityBelowThresholdCannotPreempt(highPriority),
			nodeNames:         []string{"node1", "node2", "node3"},
			preemptor:         preemption.NewPodPreemptor(st.MakePod().Name("p1").UID("p1").Priority(highPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(highPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			// highPriority can preempt anything, but not other highPriority
			expected: map[string][]string{"node2": {"v2"}, "node3": {"v3"}},
		},
		{
			name:              "only pods at or above specified priority can preempted: mid priority",
			eligiblePreemptor: priorityBelowThresholdCannotPreempt(highPriority),
			nodeNames:         []string{"node1", "node2", "node3"},
			preemptor:         preemption.NewPodPreemptor(st.MakePod().Name("p2").UID("p2").Priority(midPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(highPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			// midPriority can't preempt anything
			expected: map[string][]string{},
		},
		{
			name:              "only pods at or below specified priority can be preempted: high priority",
			eligiblePreemptor: priorityAboveThresholdCannotBePreempted(midPriority),
			nodeNames:         []string{"node1", "node2", "node3"},
			preemptor:         preemption.NewPodPreemptor(st.MakePod().Name("p1").UID("p1").Priority(highPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(highPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			// the lowPriority pod can be preempted but not the midPriority pod
			expected:                map[string][]string{"node2": {"v2"}, "node3": {"v3"}},
			workloadAwarePreemption: true,
		},
		{
			name:              "workload: filter for matching pod label: high priority",
			eligiblePreemptor: victimLabelsAreEligible("preemptible", "yes"),
			nodeNames:         []string{"node1", "node2", "node3", "node4"},
			preemptor: newPodGroupPreemptor(highPriority,
				[]*v1.Pod{st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj()},
				nil),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Label("preemptible", "no").Node("node1").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Label("preemptible", "yes").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Label("preemptible", "yes").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node4").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"v1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"v2"}, capacity: 1},
				{nodeName: "node3", blockingVictims: []string{"v3"}, capacity: 1},
				{nodeName: "node4", blockingVictims: []string{"v4"}, capacity: 1},
			},
			expected:                map[string][]string{"Cluster-Scope-pg1": {"v3"}},
			workloadAwarePreemption: true,
		},
		{
			name:              "workload: filter for matching pod label: mid priority",
			eligiblePreemptor: victimLabelsAreEligible("preemptible", "yes"),
			nodeNames:         []string{"node1", "node2", "node3", "node4"},
			// Preemptor: Mid Priority Workload
			preemptor: newPodGroupPreemptor(midPriority,
				[]*v1.Pod{st.MakePod().Name("gang-p2").UID("gang-p2").WorkloadRef(w1).Priority(midPriority).Req(largeRes).Obj()},
				nil),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Label("preemptible", "no").Node("node1").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Label("preemptible", "yes").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Label("preemptible", "yes").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node4").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"v1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"v2"}, capacity: 1},
				{nodeName: "node3", blockingVictims: []string{"v3"}, capacity: 1},
				{nodeName: "node4", blockingVictims: []string{"v4"}, capacity: 1},
			},
			workloadAwarePreemption: true,
			expected:                map[string][]string{"Cluster-Scope-pg1": {"v3"}},
		},
		{
			name:              "workload: filter for matching victim node: high priority",
			eligiblePreemptor: domainNodeNamesAreEligible([]string{"node1"}),
			nodeNames:         []string{"node1", "node2", "node3"},
			preemptor: newPodGroupPreemptor(highPriority,
				[]*v1.Pod{st.MakePod().Name("gang-p3").UID("gang-p3").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj()},
				nil),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(highPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node1").Priority(midPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node1").Priority(lowPriority).Req(smallRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v5").UID("v5").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"v2", "v3"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"v4"}, capacity: 1},
				{nodeName: "node3", blockingVictims: []string{"v5"}, capacity: 1},
			},
			workloadAwarePreemption: true,
			expected:                map[string][]string{"Cluster-Scope-pg1": {"v2", "v3"}},
		},
		{
			name:              "workload: only pods at or above specified priority can be preemptor",
			eligiblePreemptor: priorityBelowThresholdCannotPreempt(highPriority),
			nodeNames:         []string{"node1", "node2", "node3"},
			preemptor: newPodGroupPreemptor(highPriority,
				[]*v1.Pod{st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj()},
				nil),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(highPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"v1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"v2"}, capacity: 1},
				{nodeName: "node3", blockingVictims: []string{"v3"}, capacity: 1},
			},
			expected:                map[string][]string{"Cluster-Scope-pg1": {"v3"}},
			workloadAwarePreemption: true,
		},
		{
			name:              "workload: only pods at or below specified priority can be preempted",
			eligiblePreemptor: priorityAboveThresholdCannotBePreempted(midPriority),
			nodeNames:         []string{"node1", "node2", "node3"},
			preemptor: newPodGroupPreemptor(highPriority,
				[]*v1.Pod{st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj()},
				nil),
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(highPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node2").Priority(midPriority).Req(largeRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node3").Priority(lowPriority).Req(largeRes).StartTime(epochTime).Obj(),
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"v1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"v2"}, capacity: 1},
				{nodeName: "node3", blockingVictims: []string{"v3"}, capacity: 1},
			},
			expected:                map[string][]string{"Cluster-Scope-pg1": {"v3"}},
			workloadAwarePreemption: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodes := make([]*v1.Node, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodes[i] = st.MakeNode().Name(nodeName).Capacity(veryLargeRes).Obj()
			}

			var objs []runtime.Object

			var premptorPods []*v1.Pod
			premptorPods = append(premptorPods, tt.preemptor.Members()...)

			for _, pod := range premptorPods {
				objs = append(objs, pod)

			}
			for _, pod := range tt.pods {
				objs = append(objs, pod)
			}
			cs := clientsetfake.NewClientset(objs...)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			snapshot := internalcache.NewSnapshot(tt.pods, nodes)
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fwk, err := tf.NewFramework(
				ctx,
				[]tf.RegisterPluginFunc{
					tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
					tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
					tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				},
				"",
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithLogger(logger),
			)
			if err != nil {
				t.Fatal(err)
			}

			state := framework.NewCycleState()
			for _, pod := range premptorPods {
				// Some tests rely on PreFilter plugin to compute its CycleState.
				if _, status, _ := fwk.RunPreFilterPlugins(ctx, state, pod); !status.IsSuccess() {
					t.Errorf("Unexpected PreFilter Status: %v", status)
				}
			}
			nodeInfos, err := snapshot.NodeInfos().List()
			if err != nil {
				t.Fatal(err)
			}

			pl, err := New(ctx, getDefaultDefaultPreemptionArgs(), fwk, feature.Features{
				EnableWorkloadAwarePreemption: tt.workloadAwarePreemption,
			})
			if err != nil {
				t.Fatal(err)
			}
			if tt.blockingRules != nil {
				pl.CanPlacePods = getMockCanPlacePodsFunc(tt.blockingRules)
			}

			// Override eligibility logic
			if tt.eligiblePreemptor != nil {
				pl.IsEligiblePreemptor = tt.eligiblePreemptor
			}
			offset, numCandidates := pl.GetOffsetAndNumCandidates(int32(len(nodeInfos)))
			domains := pl.Evaluator.NewDomains(tt.preemptor, nodeInfos)
			candidates, _, _ := pl.Evaluator.DryRunPreemption(ctx, state, tt.preemptor, domains, nil, offset, numCandidates)
			// check that the candidates match what's expected
			if len(tt.expected) != len(candidates) {
				candidateNames := []string{}
				for _, c := range candidates {
					candidateNames = append(candidateNames, c.Name())
				}
				t.Fatalf("expected %d candidates (%+v) but got %d: %+v", len(tt.expected), tt.expected, len(candidates), candidateNames)
			}
			for len(candidates) > 0 {
				selected := pl.Evaluator.SelectCandidate(ctx, candidates)

				expectVictims, ok := tt.expected[selected.Name()]
				if !ok {
					t.Fatalf("got unexpected candidate %+v, when expected is %+v", selected, tt.expected)
				}

				gotVictims := []string{}
				for _, p := range selected.Victims().Pods {
					gotVictims = append(gotVictims, p.Name)
				}
				if diff := cmp.Diff(expectVictims, gotVictims); diff != "" {
					t.Errorf("Unexpected victims on node %s (-want,+got):\n%s", selected.Name(), diff)
				}

				// remove selected from candidates
				notSelected := []preemption.Candidate{}
				for _, c := range candidates {
					if c.Name() != selected.Name() {
						notSelected = append(notSelected, c)
					}
				}
				candidates = notSelected
			}
		})
	}
}

func TestCustomOrdering(t *testing.T) {
	var (
		w1 = &v1.WorkloadReference{PodGroup: "pg1"}
		w2 = &v1.WorkloadReference{PodGroup: "pg2"}
	)

	// Two arbitrary examples of custom selection ordering to check that they behave as expected
	orderByOldestStart := func(pod1, pod2 *v1.Pod) bool {
		return util.GetPodStartTime(pod1).Before(util.GetPodStartTime(pod2))
	}
	orderByOldestStartVictim := func(vi1, vi2 []*v1.Pod, _ bool) bool {

		sort.Slice(vi1, func(i, j int) bool {
			return orderByOldestStart(vi1[i], vi1[j])
		})
		sort.Slice(vi2, func(i, j int) bool {
			return orderByOldestStart(vi2[i], vi2[j])
		})

		return util.GetPodStartTime(vi1[0]).Before(util.GetPodStartTime(vi2[0]))
	}

	tests := []struct {
		name          string
		orderVictims  MoreImportantVictimFunc
		nodeNames     []string
		preemptor     preemption.Preemptor
		pods          []*v1.Pod
		blockingRules []blockingRule
		expectedPods  []string
	}{
		{
			name:         "select newest pods",
			orderVictims: orderByOldestStartVictim,
			nodeNames:    []string{"node1"},
			preemptor:    preemption.NewPodPreemptor(st.MakePod().Name("p2").UID("p2").Priority(highPriority).Req(largeRes).Obj()),
			// size victims to require at least two to be preempted
			pods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(lowPriority).Req(mediumRes).StartTime(epochTime2).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node1").Priority(lowPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node1").Priority(lowPriority).Req(mediumRes).StartTime(epochTime1).Obj(),
			},
			// the newest two pods are selected, despite one with higher priority
			expectedPods: []string{"v3", "v1"},
		},
		{
			name:         "workload: select newest workload (gang) when capacity can be satisfied by either",
			orderVictims: orderByOldestStartVictim, // Strategy: Sort victims so Newest are first (to be killed)
			nodeNames:    []string{"node1"},
			// Preemptor: High Priority Pod needing LargeRes
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				// Workload 1: OLD (Start = epochTime) -> "Senior"
				st.MakePod().Name("old-w1-p1").UID("w1-p1").Node("node1").WorkloadRef(w1).Priority(lowPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("old-w1-p2").UID("w1-p2").Node("node1").WorkloadRef(w1).Priority(lowPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				// Workload 2: NEW (Start = epochTime2) -> "Junior"
				st.MakePod().Name("new-w2-p1").UID("w2-p1").Node("node1").WorkloadRef(w2).Priority(lowPriority).Req(mediumRes).StartTime(epochTime2).Obj(),
				st.MakePod().Name("new-w2-p2").UID("w2-p2").Node("node1").WorkloadRef(w2).Priority(lowPriority).Req(mediumRes).StartTime(epochTime2).Obj(),
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"old-w1-p1", "old-w1-p2"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"new-w2-p1", "new-w2-p2"}, capacity: 1},
			},
			// Expectation: The NEWER gang (w2) is selected for preemption.
			expectedPods: []string{"new-w2-p1", "new-w2-p2"},
		},
		{
			name:         "workload: select lower priority workload even if it is older",
			orderVictims: orderByOldestStartVictim,
			nodeNames:    []string{"node1"},
			preemptor:    preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Req(largeRes).Obj()),
			pods: []*v1.Pod{
				// Workload 1: LOW Priority but OLD (Start = epochTime)
				st.MakePod().Name("low-old-p1").UID("low-p1").Node("node1").WorkloadRef(w1).Priority(lowPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				st.MakePod().Name("low-old-p2").UID("low-p2").Node("node1").WorkloadRef(w1).Priority(lowPriority).Req(mediumRes).StartTime(epochTime).Obj(),
				// Workload 2: MID Priority but NEW (Start = epochTime2)
				st.MakePod().Name("mid-new-p1").UID("mid-p1").Node("node1").WorkloadRef(w2).Priority(midPriority).Req(mediumRes).StartTime(epochTime2).Obj(),
				st.MakePod().Name("mid-new-p2").UID("mid-p2").Node("node1").WorkloadRef(w2).Priority(midPriority).Req(mediumRes).StartTime(epochTime2).Obj(),
			},
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"low-old-p1", "low-old-p2"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"mid-new-p1", "mid-new-p2"}, capacity: 1},
			},
			// Expectation: Priority (Low) is a stronger signal than Age. Preempt w1.
			expectedPods: []string{"low-old-p1", "low-old-p2"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodes := make([]*v1.Node, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodes[i] = st.MakeNode().Name(nodeName).Capacity(veryLargeRes).Obj()
			}

			var objs []runtime.Object
			var podsToPreempt []*v1.Pod
			podsToPreempt = append(podsToPreempt, tt.preemptor.Members()...)
			for _, pod := range podsToPreempt {
				objs = append(objs, pod)
			}
			for _, pod := range tt.pods {
				objs = append(objs, pod)
			}
			cs := clientsetfake.NewClientset(objs...)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			snapshot := internalcache.NewSnapshot(tt.pods, nodes)
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fwk, err := tf.NewFramework(
				ctx,
				[]tf.RegisterPluginFunc{
					tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
					tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
					tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
				},
				"",
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithLogger(logger),
			)
			if err != nil {
				t.Fatal(err)
			}

			state := framework.NewCycleState()
			for _, pod := range podsToPreempt {
				// Some tests rely on PreFilter plugin to compute its CycleState.
				if _, status, _ := fwk.RunPreFilterPlugins(ctx, state, pod); !status.IsSuccess() {
					t.Errorf("Unexpected PreFilter Status: %v", status)
				}
			}
			nodeInfos, err := snapshot.NodeInfos().List()
			if err != nil {
				t.Fatal(err)
			}

			pl, err := New(ctx, getDefaultDefaultPreemptionArgs(), fwk, feature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			if tt.blockingRules != nil {
				pl.CanPlacePods = getMockCanPlacePodsFunc(tt.blockingRules)
			}

			if tt.orderVictims != nil {
				pl.MoreImportantVictim = tt.orderVictims
			}
			offset, numCandidates := pl.GetOffsetAndNumCandidates(int32(len(nodeInfos)))
			domains := pl.Evaluator.NewDomains(tt.preemptor, nodeInfos)
			candidates, _, _ := pl.Evaluator.DryRunPreemption(ctx, state, tt.preemptor, domains, nil, offset, numCandidates)
			if len(candidates) != 1 {
				t.Fatalf("expected exactly one node but got %+v", candidates)
			}
			podNames := []string{}
			for _, p := range candidates[0].Victims().Pods {
				podNames = append(podNames, p.Name)
			}
			if diff := cmp.Diff(tt.expectedPods, podNames); diff != "" {
				t.Errorf("expect pods %+v, but got pods %+v", tt.expectedPods, podNames)
			}
		})
	}
}

func TestPodEligibleToPreemptOthers(t *testing.T) {
	tests := []struct {
		name                string
		pod                 *v1.Pod
		pods                []*v1.Pod
		nodes               []string
		nominatedNodeStatus *fwk.Status
		expected            bool
	}{
		{
			name:                "Pod with nominated node",
			pod:                 st.MakePod().Name("p_with_nominated_node").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj(),
			pods:                []*v1.Pod{st.MakePod().Name("p1").UID("p1").Priority(lowPriority).Node("node1").Terminating().Obj()},
			nodes:               []string{"node1"},
			nominatedNodeStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, tainttoleration.ErrReasonNotMatch),
			expected:            true,
		},
		{
			name:                "Pod without nominated node",
			pod:                 st.MakePod().Name("p_without_nominated_node").UID("p").Priority(highPriority).Obj(),
			pods:                []*v1.Pod{},
			nodes:               []string{},
			nominatedNodeStatus: nil,
			expected:            true,
		},
		{
			name:                "Pod with 'PreemptNever' preemption policy",
			pod:                 st.MakePod().Name("p_with_preempt_never_policy").UID("p").Priority(highPriority).PreemptionPolicy(v1.PreemptNever).Obj(),
			pods:                []*v1.Pod{},
			nodes:               []string{},
			nominatedNodeStatus: nil,
			expected:            false,
		},
		{
			name: "preemption victim pod terminating, as indicated by the dedicated DisruptionTarget condition",
			pod:  st.MakePod().Name("p_with_nominated_node").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj(),
			pods: []*v1.Pod{st.MakePod().Name("p1").UID("p1").Priority(lowPriority).Node("node1").Terminating().
				Condition(v1.DisruptionTarget, v1.ConditionTrue, v1.PodReasonPreemptionByScheduler).Obj()},
			nodes:    []string{"node1"},
			expected: false,
		},
		{
			name:     "non-victim Pods terminating",
			pod:      st.MakePod().Name("p_with_nominated_node").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj(),
			pods:     []*v1.Pod{st.MakePod().Name("p1").UID("p1").Priority(lowPriority).Node("node1").Terminating().Obj()},
			nodes:    []string{"node1"},
			expected: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			var nodes []*v1.Node
			for _, n := range test.nodes {
				nodes = append(nodes, st.MakeNode().Name(n).Obj())
			}
			var pods []runtime.Object
			pods = append(pods, test.pod)
			for _, pod := range test.pods {
				pods = append(pods, pod)
			}
			cs := clientsetfake.NewClientset(pods...)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			registeredPlugins := []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			f, err := tf.NewFramework(ctx, registeredPlugins, "",
				frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot(test.pods, nodes)),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithLogger(logger),
			)
			if err != nil {
				t.Fatal(err)
			}
			pl, err := New(ctx, getDefaultDefaultPreemptionArgs(), f, feature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			if got, _ := pl.PodEligibleToPreemptOthers(ctx, test.pod, test.nominatedNodeStatus); got != test.expected {
				t.Errorf("expected %t, got %t for pod: %s", test.expected, got, test.pod.Name)
			}
		})
	}
}

func TestPreempt(t *testing.T) {
	var (
		w1 = &v1.WorkloadReference{PodGroup: "pg1"}
	)

	metrics.Register()
	tests := []struct {
		name                    string
		preemptor               preemption.Preemptor
		pods                    []*v1.Pod
		extenders               []*tf.FakeExtender
		nodeNames               []string
		registerPlugin          tf.RegisterPluginFunc
		want                    *fwk.PostFilterResult
		blockingRules           []blockingRule
		expectedPods            []string // list of preempted pods
		workloadAwarePreemption bool
	}{
		{
			name:      "basic preemption logic",
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(veryLargeRes).PreemptionPolicy(v1.PreemptLowerPriority).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(highPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Req(mediumRes).Obj(),
			},
			nodeNames:      []string{"node1", "node2", "node3"},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			want:           framework.NewPostFilterResultWithNominatedNode("node1"),
			expectedPods:   []string{"p1.1", "p1.2"},
		},
		{
			name: "preemption for topology spread constraints",
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Label("foo", "").Priority(highPriority).
				SpreadConstraint(1, "zone", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, nil).
				SpreadConstraint(1, "hostname", v1.DoNotSchedule, st.MakeLabelSelector().Exists("foo").Obj(), nil, nil, nil, nil).
				Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p-a1").UID("p-a1").Namespace(v1.NamespaceDefault).Node("node-a").Label("foo", "").Priority(highPriority).Obj(),
				st.MakePod().Name("p-a2").UID("p-a2").Namespace(v1.NamespaceDefault).Node("node-a").Label("foo", "").Priority(highPriority).Obj(),
				st.MakePod().Name("p-b1").UID("p-b1").Namespace(v1.NamespaceDefault).Node("node-b").Label("foo", "").Priority(lowPriority).Obj(),
				st.MakePod().Name("p-x1").UID("p-x1").Namespace(v1.NamespaceDefault).Node("node-x").Label("foo", "").Priority(highPriority).Obj(),
				st.MakePod().Name("p-x2").UID("p-x2").Namespace(v1.NamespaceDefault).Node("node-x").Label("foo", "").Priority(highPriority).Obj(),
			},
			nodeNames:      []string{"node-a/zone1", "node-b/zone1", "node-x/zone2"},
			registerPlugin: tf.RegisterPluginAsExtensions(podtopologyspread.Name, podTopologySpreadFunc, "PreFilter", "Filter"),
			want:           framework.NewPostFilterResultWithNominatedNode("node-b"),
			expectedPods:   []string{"p-b1"},
		},
		{
			name:      "Scheduler extenders allow only node1, otherwise node3 would have been chosen",
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(veryLargeRes).PreemptionPolicy(v1.PreemptLowerPriority).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Namespace(v1.NamespaceDefault).Node("node1").Priority(midPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Namespace(v1.NamespaceDefault).Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Namespace(v1.NamespaceDefault).Node("node3").Priority(midPriority).Req(largeRes).Obj(),
			},
			nodeNames: []string{"node1", "node2", "node3"},
			extenders: []*tf.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Predicates:   []tf.FitPredicate{tf.TruePredicateExtender},
				},
				{
					ExtenderName: "FakeExtender2",
					Predicates:   []tf.FitPredicate{tf.Node1PredicateExtender},
				},
			},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			want:           framework.NewPostFilterResultWithNominatedNode("node1"),
			expectedPods:   []string{"p1.1", "p1.2"},
		},
		{
			name:      "Scheduler extenders do not allow any preemption",
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(veryLargeRes).PreemptionPolicy(v1.PreemptLowerPriority).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Namespace(v1.NamespaceDefault).Node("node1").Priority(midPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Namespace(v1.NamespaceDefault).Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Namespace(v1.NamespaceDefault).Node("node2").Priority(midPriority).Req(largeRes).Obj(),
			},
			nodeNames: []string{"node1", "node2", "node3"},
			extenders: []*tf.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Predicates:   []tf.FitPredicate{tf.FalsePredicateExtender},
				},
			},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			want:           nil,
			expectedPods:   []string{},
		},
		{
			name:      "One scheduler extender allows only node1, the other returns error but ignorable. Only node1 would be chosen",
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(veryLargeRes).PreemptionPolicy(v1.PreemptLowerPriority).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Namespace(v1.NamespaceDefault).Node("node1").Priority(midPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Namespace(v1.NamespaceDefault).Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Namespace(v1.NamespaceDefault).Node("node2").Priority(midPriority).Req(largeRes).Obj(),
			},
			nodeNames: []string{"node1", "node2", "node3"},
			extenders: []*tf.FakeExtender{
				{
					Predicates:   []tf.FitPredicate{tf.ErrorPredicateExtender},
					Ignorable:    true,
					ExtenderName: "FakeExtender1",
				},
				{
					Predicates:   []tf.FitPredicate{tf.Node1PredicateExtender},
					ExtenderName: "FakeExtender2",
				},
			},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			want:           framework.NewPostFilterResultWithNominatedNode("node1"),
			expectedPods:   []string{"p1.1", "p1.2"},
		},
		{
			name:      "One scheduler extender allows only node1, but it is not interested in given pod, otherwise node1 would have been chosen",
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(veryLargeRes).PreemptionPolicy(v1.PreemptLowerPriority).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Namespace(v1.NamespaceDefault).Node("node1").Priority(midPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Namespace(v1.NamespaceDefault).Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Namespace(v1.NamespaceDefault).Node("node2").Priority(midPriority).Req(largeRes).Obj(),
			},
			nodeNames: []string{"node1", "node2"},
			extenders: []*tf.FakeExtender{
				{
					ExtenderName: "FakeExtender1",
					Predicates:   []tf.FitPredicate{tf.Node1PredicateExtender},
					UnInterested: true,
				},
				{
					ExtenderName: "FakeExtender2",
					Predicates:   []tf.FitPredicate{tf.TruePredicateExtender},
				},
			},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			// sum of priorities of all victims on node1 is larger than node2, node2 is chosen.
			want:         framework.NewPostFilterResultWithNominatedNode("node2"),
			expectedPods: []string{"p2.1"},
		},
		{
			name:      "no preempting in pod",
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(veryLargeRes).PreemptionPolicy(v1.PreemptNever).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Namespace(v1.NamespaceDefault).Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Namespace(v1.NamespaceDefault).Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Namespace(v1.NamespaceDefault).Node("node2").Priority(highPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Namespace(v1.NamespaceDefault).Node("node3").Priority(midPriority).Req(mediumRes).Obj(),
			},
			nodeNames:      []string{"node1", "node2", "node3"},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			want:           nil,
			expectedPods:   nil,
		},
		{
			name:      "PreemptionPolicy is nil",
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Namespace(v1.NamespaceDefault).Priority(highPriority).Req(veryLargeRes).Obj()),
			pods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Namespace(v1.NamespaceDefault).Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p1.2").UID("p1.2").Namespace(v1.NamespaceDefault).Node("node1").Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Namespace(v1.NamespaceDefault).Node("node2").Priority(highPriority).Req(largeRes).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Namespace(v1.NamespaceDefault).Node("node3").Priority(midPriority).Req(mediumRes).Obj(),
			},
			nodeNames:      []string{"node1", "node2", "node3"},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			want:           framework.NewPostFilterResultWithNominatedNode("node1"),
			expectedPods:   []string{"p1.1", "p1.2"},
		},
		{
			name: "workload: basic success - gang of 2 fits on a single node by preempting local victims",
			// Preemptor: Gang of 2 High Prio Pods
			preemptor: newPodGroupPreemptor(highPriority,
				[]*v1.Pod{
					st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
					st.MakePod().Name("gang-p2").UID("gang-p2").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
				}, nil),
			pods: []*v1.Pod{
				// Node 1: Filled with Low Prio pods
				st.MakePod().Name("victim-n1-1").UID("v1").Node("node1").Priority(lowPriority).Req(largeRes).Obj(),
				st.MakePod().Name("victim-n1-2").UID("v2").Node("node1").Priority(lowPriority).Req(largeRes).Obj(),
				// Other nodes full or irrelevant
				st.MakePod().Name("safe-n2").UID("s1").Node("node2").Priority(highPriority).Req(largeRes).Obj(),
			},
			nodeNames:      []string{"node1", "node2"},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			// Rules: Node1 has capacity if both victims are removed.
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"victim-n1-1", "victim-n1-2"}, capacity: 2},
				{nodeName: "node2", blockingVictims: []string{}, capacity: 0},
			},
			// Expected: Node1 nominated. Both victims preempted.
			want:                    framework.NewPostFilterResultWithNominatedNode("Cluster-Scope-pg1"),
			expectedPods:            []string{"victim-n1-1", "victim-n1-2"},
			workloadAwarePreemption: true,
		},
		{
			name: "workload: distributed success - gang fits by clearing TWO different nodes (atomic check)",
			// Preemptor: Gang of 2 High Prio Pods
			// Requirement: Need 2 slots.
			preemptor: newPodGroupPreemptor(highPriority,
				[]*v1.Pod{
					st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
					st.MakePod().Name("gang-p2").UID("gang-p2").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
				}, nil),
			pods: []*v1.Pod{
				// Node 1: One victim
				st.MakePod().Name("victim-n1").UID("v1").Node("node1").Priority(lowPriority).Req(largeRes).Obj(),
				// Node 2: One victim
				st.MakePod().Name("victim-n2").UID("v2").Node("node2").Priority(lowPriority).Req(largeRes).Obj(),
			},
			nodeNames:      []string{"node1", "node2"},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			// Rules:
			// Node1 capacity=1 if victim-n1 removed.
			// Node2 capacity=1 if victim-n2 removed.
			// Total Capacity = 1 + 1 = 2 (Fits the gang!)
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"victim-n1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"victim-n2"}, capacity: 1},
			},
			want:                    framework.NewPostFilterResultWithNominatedNode("Cluster-Scope-pg1"),
			expectedPods:            []string{"victim-n1", "victim-n2"},
			workloadAwarePreemption: true,
		},
		{
			name: "workload: partial failure - gang needs 2 slots, only 1 node can be cleared",
			preemptor: newPodGroupPreemptor(highPriority,
				[]*v1.Pod{
					st.MakePod().Name("gang-p1").UID("gang-p1").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
					st.MakePod().Name("gang-p2").UID("gang-p2").WorkloadRef(w1).Priority(highPriority).Req(largeRes).Obj(),
				}, nil),
			pods: []*v1.Pod{
				// Node 1: Cleared by removing low prio victim
				st.MakePod().Name("victim-n1").UID("v1").Node("node1").Priority(lowPriority).Req(largeRes).Obj(),
				// Node 2: Blocked by HIGH prio pod (Cannot preempt)
				st.MakePod().Name("blocker-n2").UID("b2").Node("node2").Priority(highPriority).Req(largeRes).Obj(),
			},
			nodeNames:      []string{"node1", "node2"},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			// Rules:
			// Node 1 -> Capacity 1 (if v1 gone)
			// Node 2 -> Capacity 0 (Cannot remove b2)
			// Total = 1. Needed = 2. -> FAIL.
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"victim-n1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"blocker-n2"}, capacity: 0},
			},
			// Expected: No preemption happens because the gang doesn't fit fully.
			want:         framework.NewPostFilterResultWithNominatedNode(""),
			expectedPods: nil,
		},
		{
			name: "workload: atomic victim - single pod preemptor evicts entire gang",
			// Preemptor: Single High Prio Pod
			preemptor: preemption.NewPodPreemptor(
				st.MakePod().Name("single-p").UID("p").Priority(highPriority).Req(largeRes).Obj(),
			),
			pods: []*v1.Pod{
				// Victim: Gang of 2 (distributed)
				st.MakePod().Name("w1-p1").UID("w1-p1").Node("node1").WorkloadRef(w1).Priority(lowPriority).Req(smallRes).Obj(),
				st.MakePod().Name("w1-p2").UID("w1-p2").Node("node2").WorkloadRef(w1).Priority(lowPriority).Req(smallRes).Obj(),
			},
			nodeNames:      []string{"node1", "node2"},
			registerPlugin: tf.RegisterPluginAsExtensions(noderesources.Name, nodeResourcesFitFunc, "Filter", "PreFilter"),
			// Rules:
			// To clear space on Node1, we must remove w1-p1.
			// Because w1-p1 is part of a Workload, w1-p2 is ALSO removed.
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"w1-p1", "w1-p2"}, capacity: 1},
			},
			// Expected: Node1 selected. BOTH pods preempted (including collateral damage on Node2).
			want:                    framework.NewPostFilterResultWithNominatedNode("node1"),
			expectedPods:            []string{"w1-p1", "w1-p2"},
			workloadAwarePreemption: true,
		},
	}

	labelKeys := []string{"hostname", "zone", "region"}
	for _, asyncPreemptionEnabled := range []bool{true, false} {
		for _, asyncAPICallsEnabled := range []bool{true, false} {
			for _, test := range tests {
				t.Run(fmt.Sprintf("%s (Async preemption enabled: %v, Async API calls enabled: %v)", test.name, asyncPreemptionEnabled, asyncAPICallsEnabled), func(t *testing.T) {
					client := clientsetfake.NewClientset()
					informerFactory := informers.NewSharedInformerFactory(client, 0)
					podInformer := informerFactory.Core().V1().Pods().Informer()
					var preemptorPods []*v1.Pod
					for _, pod := range test.preemptor.Members() {
						preemptorPods = append(preemptorPods, pod.DeepCopy())
					}
					var testPods []*v1.Pod
					for _, pod := range test.pods {
						testPods = append(testPods, pod.DeepCopy())
					}

					for i := range preemptorPods {
						if err := podInformer.GetStore().Add(preemptorPods[i]); err != nil {
							t.Fatalf("Failed to add test pod %s: %v", preemptorPods[i], err)
						}
					}
					for i := range testPods {
						if err := podInformer.GetStore().Add(testPods[i]); err != nil {
							t.Fatalf("Failed to add test pod %s: %v", testPods[i], err)
						}
					}

					// Need to protect deletedPodNames and patchedPodNames to prevent DATA RACE panic.
					var mu sync.RWMutex
					deletedPodNames := sets.New[string]()
					patchedPodNames := sets.New[string]()
					patchedPods := []*v1.Pod{}
					client.PrependReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
						patchAction := action.(clienttesting.PatchAction)
						podName := patchAction.GetName()
						namespace := patchAction.GetNamespace()
						patch := patchAction.GetPatch()
						pod, err := informerFactory.Core().V1().Pods().Lister().Pods(namespace).Get(podName)
						if err != nil {
							t.Fatalf("Failed to get the original pod %s/%s before patching: %v\n", namespace, podName, err)
						}
						marshalledPod, err := json.Marshal(pod)
						if err != nil {
							t.Fatalf("Failed to marshal the original pod %s/%s: %v", namespace, podName, err)
						}
						updated, err := strategicpatch.StrategicMergePatch(marshalledPod, patch, v1.Pod{})
						if err != nil {
							t.Fatalf("Failed to apply strategic merge patch %q on pod %#v: %v", patch, marshalledPod, err)
						}
						updatedPod := &v1.Pod{}
						if err := json.Unmarshal(updated, updatedPod); err != nil {
							t.Fatalf("Failed to unmarshal updated pod %q: %v", updated, err)
						}
						patchedPods = append(patchedPods, updatedPod)
						mu.Lock()
						defer mu.Unlock()
						patchedPodNames.Insert(podName)
						return true, nil, nil
					})
					client.PrependReactor("delete", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
						mu.Lock()
						defer mu.Unlock()
						deletedPodNames.Insert(action.(clienttesting.DeleteAction).GetName())
						return true, nil, nil
					})

					logger, ctx := ktesting.NewTestContext(t)
					ctx, cancel := context.WithCancel(ctx)
					defer cancel()

					waitingPods := frameworkruntime.NewWaitingPodsMap()

					var apiDispatcher *apidispatcher.APIDispatcher
					if asyncAPICallsEnabled {
						apiDispatcher = apidispatcher.New(client, 16, apicalls.Relevances)
						apiDispatcher.Run(logger)
						defer apiDispatcher.Close()
					}

					cache := internalcache.New(ctx, apiDispatcher)
					for _, pod := range testPods {
						if err := cache.AddPod(logger, pod.DeepCopy()); err != nil {
							t.Fatalf("Failed to add pod %s: %v", pod.Name, err)
						}
					}
					cachedNodeInfoMap := map[string]*framework.NodeInfo{}
					nodes := make([]*v1.Node, len(test.nodeNames))
					for i, name := range test.nodeNames {
						node := st.MakeNode().Name(name).Capacity(veryLargeRes).Obj()
						// Split node name by '/' to form labels in a format of
						// {"hostname": node.Name[0], "zone": node.Name[1], "region": node.Name[2]}
						node.Labels = make(map[string]string)
						for i, label := range strings.Split(node.Name, "/") {
							node.Labels[labelKeys[i]] = label
						}
						node.Name = node.Labels["hostname"]
						t.Logf("node is added: %v. labels: %#v", node.Name, node.Labels)
						cache.AddNode(logger, node)
						nodes[i] = node

						// Set nodeInfo to extenders to mock extenders' cache for preemption.
						cachedNodeInfo := framework.NewNodeInfo()
						cachedNodeInfo.SetNode(node)
						cachedNodeInfoMap[node.Name] = cachedNodeInfo
					}
					var extenders []fwk.Extender
					for _, extender := range test.extenders {
						// Set nodeInfoMap as extenders cached node information.
						extender.CachedNodeNameToInfo = cachedNodeInfoMap
						extenders = append(extenders, extender)
					}
					schedFramework, err := tf.NewFramework(
						ctx,
						[]tf.RegisterPluginFunc{
							test.registerPlugin,
							tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
							tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
						},
						"",
						frameworkruntime.WithClientSet(client),
						frameworkruntime.WithAPIDispatcher(apiDispatcher),
						frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
						frameworkruntime.WithExtenders(extenders),
						frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
						frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot(testPods, nodes)),
						frameworkruntime.WithInformerFactory(informerFactory),
						frameworkruntime.WithWaitingPods(waitingPods),
						frameworkruntime.WithLogger(logger),
						frameworkruntime.WithPodActivator(&fakePodActivator{}),
					)
					if err != nil {
						t.Fatal(err)
					}
					if asyncAPICallsEnabled {
						schedFramework.SetAPICacher(apicache.New(nil, cache))
					}

					state := framework.NewCycleState()
					for _, pod := range preemptorPods {
						// Some tests rely on PreFilter plugin to compute its CycleState.
						if _, s, _ := schedFramework.RunPreFilterPlugins(ctx, state, pod); !s.IsSuccess() {
							t.Errorf("Unexpected preFilterStatus: %v", s)
						}
					}

					// Call preempt and check the expected results.
					features := feature.Features{
						EnableAsyncPreemption:         asyncPreemptionEnabled,
						EnableWorkloadAwarePreemption: test.workloadAwarePreemption,
					}
					pl, err := New(ctx, getDefaultDefaultPreemptionArgs(), schedFramework, features)
					if err != nil {
						t.Fatal(err)
					}
					if test.blockingRules != nil {
						pl.CanPlacePods = getMockCanPlacePodsFunc(test.blockingRules)
					}

					// so that these nodes are eligible for preemption, we set their status
					// to Unschedulable.

					nodeToStatusMap := framework.NewDefaultNodeToStatus()
					for _, n := range nodes {
						nodeToStatusMap.Set(n.Name, fwk.NewStatus(fwk.Unschedulable))
					}
					res, status := pl.Evaluator.Preempt(ctx, state, test.preemptor, nodeToStatusMap)
					if !status.IsSuccess() && !status.IsRejected() {
						t.Errorf("unexpected error in preemption: %v", status.AsError())
					}
					if diff := cmp.Diff(test.want, res); diff != "" {
						t.Errorf("Unexpected status (-want, +got):\n%s", diff)
					}

					if asyncPreemptionEnabled {
						// Wait for the pod to be deleted.
						if err := wait.PollUntilContextTimeout(ctx, time.Millisecond*200, wait.ForeverTestTimeout, false, func(ctx context.Context) (bool, error) {
							mu.RLock()
							defer mu.RUnlock()
							return len(deletedPodNames) == len(test.expectedPods), nil
						}); err != nil {
							t.Errorf("expected %v pods to be deleted, got %v.", len(test.expectedPods), len(deletedPodNames))
						}
					} else {
						mu.RLock()
						// If async preemption is disabled, the pod should be deleted immediately.
						if len(deletedPodNames) != len(test.expectedPods) {
							t.Errorf("expected %v pods to be deleted, got %v.", len(test.expectedPods), len(deletedPodNames))
						}
						mu.RUnlock()
					}

					mu.RLock()
					if diff := cmp.Diff(sets.List(patchedPodNames), sets.List(deletedPodNames)); diff != "" {
						t.Errorf("unexpected difference in the set of patched and deleted pods: %s", diff)
					}

					// Make sure that the DisruptionTarget condition has been added to the pod status
					for _, patchedPod := range patchedPods {
						var message string
						if test.preemptor.IsPodGroup() {
							message = fmt.Sprintf("%s: preempting to accommodate a higher priority pod group", patchedPod.Spec.SchedulerName)
						} else {
							message = fmt.Sprintf("%s: preempting to accommodate a higher priority pod", patchedPod.Spec.SchedulerName)
						}
						expectedPodCondition := &v1.PodCondition{
							Type:    v1.DisruptionTarget,
							Status:  v1.ConditionTrue,
							Reason:  v1.PodReasonPreemptionByScheduler,
							Message: message,
						}

						_, condition := apipod.GetPodCondition(&patchedPod.Status, v1.DisruptionTarget)
						if diff := cmp.Diff(condition, expectedPodCondition, cmpopts.IgnoreFields(v1.PodCondition{}, "LastTransitionTime")); diff != "" {
							t.Fatalf("unexpected difference in the pod %q DisruptionTarget condition: %s", patchedPod.Name, diff)
						}
					}

					for victimName := range deletedPodNames {
						found := false
						for _, expPod := range test.expectedPods {
							if expPod == victimName {
								found = true
								break
							}
						}
						if !found {
							t.Errorf("pod %v is not expected to be a victim.", victimName)
						}
					}

					for _, testPod := range preemptorPods {
						if res != nil && res.NominatingInfo != nil {
							testPod.Status.NominatedNodeName = res.NominatedNodeName
						}
					}
					// Manually set the deleted Pods' deletionTimestamp to non-nil.
					for _, pod := range testPods {
						if deletedPodNames.Has(pod.Name) {
							now := metav1.Now()
							pod.DeletionTimestamp = &now
							deletedPodNames.Delete(pod.Name)
						}
					}
					mu.RUnlock()

					// Call preempt again and make sure it doesn't preempt any more pods.

					res, status = pl.Evaluator.Preempt(ctx, state, test.preemptor, framework.NewDefaultNodeToStatus())
					if !status.IsSuccess() && !status.IsRejected() {
						t.Errorf("unexpected error in preemption: %v", status.AsError())
					}
					if res != nil && res.NominatingInfo != nil && len(deletedPodNames) > 0 {
						t.Errorf("didn't expect any more preemption. Node %v is selected for preemption.", res.NominatedNodeName)
					}
				})
			}
		}
	}
}

type fakePodActivator struct {
}

func (f *fakePodActivator) Activate(logger klog.Logger, pods map[string]*v1.Pod) {}

func TestSelectVictimsOnDomain(t *testing.T) {
	tests := []struct {
		name                       string
		nodeNames                  []string
		initPods                   []*v1.Pod
		preemptor                  preemption.Preemptor
		workloadAwarePreemption    bool
		pdbs                       []*policy.PodDisruptionBudget
		blockingRules              []blockingRule
		expectedPods               [][]string
		expectedNumViolatingVictim []int
		expectedStatus             []*fwk.Status
	}{
		{
			name:      "Basic: Preempt single lower priority pod",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim").UID("v1").Node("node1").Priority(lowPriority).Obj(),
			},
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"victim"}, capacity: 1},
			},
			expectedPods:               [][]string{{"victim"}},
			expectedNumViolatingVictim: []int{0},
			expectedStatus:             []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:      "Priority: Prefer lower priority victim",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("high-prio").UID("v3").Node("node1").Priority(highPriority).Obj(),
				st.MakePod().Name("mid-prio").UID("v2").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("low-prio").UID("v1").Node("node1").Priority(lowPriority).Obj(),
			},
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"mid-prio"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"low-prio"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"high-prio"}, capacity: 1},
			},
			expectedPods:               [][]string{{"low-prio"}},
			expectedNumViolatingVictim: []int{0},
			expectedStatus:             []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:      "Efficiency: Preempt minimum number of victims (Binary Search)",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node1").Priority(lowPriority).Obj(),
			},
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"v1", "v2"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"v1"}, capacity: 1},
			},
			expectedPods:               [][]string{{"v1"}},
			expectedNumViolatingVictim: []int{0},
			expectedStatus:             []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:      "PDB: Prefer non-violating victim",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim-pdb").UID("v1").Node("node1").Label("app", "foo").Priority(lowPriority).Obj(),
				st.MakePod().Name("victim-no-pdb").UID("v2").Node("node1").Priority(lowPriority).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
				},
			},
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"victim-pdb"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"victim-no-pdb"}, capacity: 1},
			},
			expectedPods:               [][]string{{"victim-no-pdb"}},
			expectedNumViolatingVictim: []int{0},
			expectedStatus:             []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:      "PDB: Prefer lower prioirity pod for preemption, when preemption without pdb violation is not possible",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Label("app", "foo").Priority(lowPriority).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node1").Label("app", "foo").Priority(midPriority).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
				},
			},
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"v1"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"v2"}, capacity: 1},
			},
			expectedPods:               [][]string{{"v1"}},
			expectedNumViolatingVictim: []int{1},
			expectedStatus:             []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:                    "Workload Aware: Atomic preemption of PodGroup",
			nodeNames:               []string{"node1"},
			workloadAwarePreemption: true,
			initPods: []*v1.Pod{
				st.MakePod().Name("g1-1").UID("g1").Node("node1").WorkloadRef(&v1.WorkloadReference{PodGroup: "wg1"}).Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-2").UID("g2").Node("node1").WorkloadRef(&v1.WorkloadReference{PodGroup: "wg1"}).Priority(lowPriority).Obj(),
			},
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"g1-1", "g1-2"}, capacity: 1},
			},
			expectedPods:               [][]string{{"g1-1", "g1-2"}},
			expectedNumViolatingVictim: []int{0},
			expectedStatus:             []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:                    "Workload Aware: prefer single pod over podGroup for preemption candidate",
			nodeNames:               []string{"node1"},
			workloadAwarePreemption: true,
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-1").UID("g1").Node("node1").WorkloadRef(&v1.WorkloadReference{PodGroup: "wg1"}).Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-2").UID("g2").Node("node1").WorkloadRef(&v1.WorkloadReference{PodGroup: "wg1"}).Priority(lowPriority).Obj(),
			},
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"g1-1", "g1-2"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"p1"}, capacity: 1},
			},
			expectedPods:               [][]string{{"p1"}},
			expectedNumViolatingVictim: []int{0},
			expectedStatus:             []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:                    "Workload Aware: prefer single pod over podGroup for preemption candidate, on corresponding node",
			nodeNames:               []string{"node1", "node2"},
			workloadAwarePreemption: true,
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-1").UID("g1").Node("node1").WorkloadRef(&v1.WorkloadReference{PodGroup: "wg1"}).Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-2").UID("g2").Node("node1").WorkloadRef(&v1.WorkloadReference{PodGroup: "wg1"}).Priority(lowPriority).Obj(),
			},
			preemptor: newPodGroupPreemptor(highPriority, []*v1.Pod{
				st.MakePod().Name("p").UID("p1").WorkloadRef(&v1.WorkloadReference{PodGroup: "wg1"}).Priority(highPriority).Obj(),
			}, nil),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"g1-1", "g1-2"}, capacity: 1},
				{nodeName: "node1", blockingVictims: []string{"p1"}, capacity: 1},
				{nodeName: "node2", blockingVictims: []string{"p2"}, capacity: 1},
			},
			expectedPods:               [][]string{{"p2"}},
			expectedNumViolatingVictim: []int{0},
			expectedStatus:             []*fwk.Status{fwk.NewStatus(fwk.Success)},
		},
		{
			name:      "Failure: Cannot preempt the victim with higher priority",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim").UID("v1").Node("node1").Priority(highPriority).Obj(),
			},
			preemptor: preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(midPriority).Obj()),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: []string{"victim"}},
			},
			expectedPods:               [][]string{nil},
			expectedNumViolatingVictim: []int{0},
			expectedStatus:             []*fwk.Status{fwk.NewStatus(fwk.UnschedulableAndUnresolvable)},
		},
		{
			name:                       "Failure: Cannot preempt if node is empty",
			nodeNames:                  []string{"node1"},
			initPods:                   []*v1.Pod{},
			preemptor:                  preemption.NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(midPriority).Obj()),
			blockingRules:              []blockingRule{},
			expectedPods:               [][]string{nil},
			expectedNumViolatingVictim: []int{0},
			expectedStatus:             []*fwk.Status{fwk.NewStatus(fwk.UnschedulableAndUnresolvable)},
		},
	}

	labelKeys := []string{"hostname", "zone", "region"}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodes := make([]*v1.Node, len(tt.nodeNames))
			fakeFilterRCMap := make(map[string]fwk.Code, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodeWrapper := st.MakeNode().Capacity(largeRes)
				tpKeys := strings.Split(nodeName, "/")
				nodeWrapper.Name(tpKeys[0])
				for i, labelVal := range strings.Split(nodeName, "/") {
					nodeWrapper.Label(labelKeys[i], labelVal)
				}
				nodes[i] = nodeWrapper.Obj()
			}
			snapshot := internalcache.NewSnapshot(tt.initPods, nodes)

			fakePlugin := tf.FakeFilterPlugin{
				FailedNodeReturnCodeMap: fakeFilterRCMap,
			}
			registeredPlugins := append([]tf.RegisterPluginFunc{
				tf.RegisterFilterPlugin(
					"FakeFilter",
					func(_ context.Context, _ runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
						return &fakePlugin, nil
					},
				)},
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			preemptorPods := tt.preemptor.Members()

			var objs []runtime.Object
			for _, p := range append(preemptorPods, tt.initPods...) {
				objs = append(objs, p)
			}
			for _, n := range nodes {
				objs = append(objs, n)
			}
			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(objs...), 0)
			parallelism := parallelize.DefaultParallelism

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			testingFwk, err := tf.NewFramework(
				ctx,
				registeredPlugins, "",
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithParallelism(parallelism),
				frameworkruntime.WithLogger(logger),
			)
			if err != nil {
				t.Fatal(err)
			}

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			nodeInfos, err := snapshot.NodeInfos().List()
			if err != nil {
				t.Fatal(err)
			}
			sort.Slice(nodeInfos, func(i, j int) bool {
				return nodeInfos[i].Node().Name < nodeInfos[j].Node().Name
			})

			pl, err := New(ctx, getDefaultDefaultPreemptionArgs(), testingFwk, feature.Features{
				EnableWorkloadAwarePreemption: tt.workloadAwarePreemption,
			})
			if err != nil {
				t.Fatal(err)
			}
			pl.CanPlacePods = getMockCanPlacePodsFunc(tt.blockingRules)

			state := framework.NewCycleState()
			domains := pl.Evaluator.NewDomains(tt.preemptor, nodeInfos)

			for i, domain := range domains {
				t.Logf("Checking Domain: %s", domain.GetName())

				gotPods, gotNumViolating, gotStatus := pl.SelectVictimsOnDomain(ctx, state, tt.preemptor, domain, tt.pdbs)

				wantStatus := tt.expectedStatus[i]
				wantCode := fwk.Success
				if wantStatus != nil {
					wantCode = wantStatus.Code()
				}

				gotCode := fwk.Success
				if gotStatus != nil {
					gotCode = gotStatus.Code()
				}

				if gotCode != wantCode {
					t.Errorf("Domain %s: Status mismatch. Want %v, Got %v", domain.GetName(), wantCode, gotCode)
				}

				if wantCode != fwk.Success {
					continue
				}

				wantViolating := 0
				if i < len(tt.expectedNumViolatingVictim) {
					wantViolating = tt.expectedNumViolatingVictim[i]
				}
				if gotNumViolating != wantViolating {
					t.Errorf("Domain %s: Violating victim count mismatch. Want %d, Got %d", domain.GetName(), wantViolating, gotNumViolating)
				}

				var gotNames []string
				for _, p := range gotPods {
					gotNames = append(gotNames, p.Name)
				}
				sort.Strings(gotNames)

				wantNames := tt.expectedPods[i]
				sort.Strings(wantNames)

				if diff := cmp.Diff(wantNames, gotNames); diff != "" {
					t.Errorf("Domain %s: Victims mismatch (-want +got):\n%s", domain.GetName(), diff)
				}
			}
		})
	}

}
