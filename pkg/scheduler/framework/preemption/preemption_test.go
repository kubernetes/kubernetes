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
	"errors"
	"fmt"
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	apicache "k8s.io/kubernetes/pkg/scheduler/backend/api_cache"
	apidispatcher "k8s.io/kubernetes/pkg/scheduler/backend/api_dispatcher"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	apicalls "k8s.io/kubernetes/pkg/scheduler/framework/api_calls"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

var (
	w1 = &v1.WorkloadReference{PodGroup: "pg1"}
	w2 = &v1.WorkloadReference{PodGroup: "pg2"}

	midPriority, highPriority = int32(100), int32(1000)

	veryLargeRes = map[v1.ResourceName]string{
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
)

func init() {
	metrics.Register()
}

type FakePostFilterPlugin struct {
	numViolatingVictim int
}

func (pl *FakePostFilterPlugin) SelectVictimsOnDomain(ctx context.Context, state fwk.CycleState, preemptor Preemptor, domain Domain, pdbs []*policy.PodDisruptionBudget) (victims []*v1.Pod, numViolatingVictim int, status *fwk.Status) {
	for _, node := range domain.Nodes() {
		victims = append(victims, node.GetPods()[0].GetPod())
	}
	return victims, pl.numViolatingVictim, nil
}

func (pl *FakePostFilterPlugin) GetOffsetAndNumCandidates(nodes int32) (int32, int32) {
	return 0, nodes
}

func (pl *FakePostFilterPlugin) CandidatesToVictimsMap(candidates []Candidate) map[string]*extenderv1.Victims {
	return nil
}

func (pl *FakePostFilterPlugin) PodEligibleToPreemptOthers(_ context.Context, pod *v1.Pod, nominatedNodeStatus *fwk.Status) (bool, string) {
	return true, ""
}

func (pl *FakePostFilterPlugin) OrderedScoreFuncs(ctx context.Context, nodesToVictims map[string]*extenderv1.Victims) []func(node string) int64 {
	return nil
}

type FakePreemptionScorePostFilterPlugin struct{}

func (pl *FakePreemptionScorePostFilterPlugin) SelectVictimsOnDomain(ctx context.Context, state fwk.CycleState, preemptor Preemptor, domain Domain, pdbs []*policy.PodDisruptionBudget) (victims []*v1.Pod, numViolatingVictim int, status *fwk.Status) {
	for _, node := range domain.Nodes() {
		victims = append(victims, node.GetPods()[0].GetPod())
	}
	return victims, 1, nil
}

func (pl *FakePreemptionScorePostFilterPlugin) GetOffsetAndNumCandidates(nodes int32) (int32, int32) {
	return 0, nodes
}

func (pl *FakePreemptionScorePostFilterPlugin) CandidatesToVictimsMap(candidates []Candidate) map[string]*extenderv1.Victims {
	m := make(map[string]*extenderv1.Victims, len(candidates))
	for _, c := range candidates {
		m[c.Name()] = c.Victims()
	}
	return m
}

func (pl *FakePreemptionScorePostFilterPlugin) PodEligibleToPreemptOthers(_ context.Context, pod *v1.Pod, nominatedNodeStatus *fwk.Status) (bool, string) {
	return true, ""
}

func (pl *FakePreemptionScorePostFilterPlugin) OrderedScoreFuncs(ctx context.Context, nodesToVictims map[string]*extenderv1.Victims) []func(node string) int64 {
	return []func(string) int64{
		func(node string) int64 {
			var sumContainers int64
			for _, pod := range nodesToVictims[node].Pods {
				sumContainers += int64(len(pod.Spec.Containers) + len(pod.Spec.InitContainers))
			}
			// The smaller the sumContainers, the higher the score.
			return -sumContainers
		},
	}
}

func newPodGroupPreemptor(priority int32, members []*v1.Pod, policy *v1.PreemptionPolicy) Preemptor {
	return &preemptor{
		priority:         priority,
		pods:             members,
		isPodGroup:       true,
		preemptionPolicy: policy,
	}
}

func TestDryRunPreemption(t *testing.T) {
	tests := []struct {
		name                    string
		nodes                   []*v1.Node
		preemptors              []Preemptor
		initPods                []*v1.Pod
		numViolatingVictim      int
		expected                [][]Candidate
		workloadAwarePreemption bool
	}{
		{
			name: "no pdb violation",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(veryLargeRes).Obj(),
				st.MakeNode().Name("node2").Capacity(veryLargeRes).Obj(),
			},
			preemptors: []Preemptor{
				NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
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
			preemptors: []Preemptor{
				NewPodPreemptor(st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()),
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
		{
			name: "pod group as preemptor and whole cluster as domain",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(veryLargeRes).Obj(),
				st.MakeNode().Name("node2").Capacity(veryLargeRes).Obj(),
			},
			preemptors: []Preemptor{
				newPodGroupPreemptor(highPriority,
					[]*v1.Pod{
						st.MakePod().Name("pr1").UID("pr1").WorkloadRef(w1).Priority(highPriority).Obj(),
						st.MakePod().Name("pr2").UID("pr2").WorkloadRef(w1).Priority(highPriority).Obj(),
					}, nil),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
			},
			numViolatingVictim: 0,
			expected: [][]Candidate{
				{
					&candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
								st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
							},
						},
						name: "Cluster-Scope-pg1",
					},
				},
			},
			workloadAwarePreemption: true,
		},
		{
			name: "pod group as preemptor and whole cluster as domain and has pod group for preempion",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(veryLargeRes).Obj(),
				st.MakeNode().Name("node2").Capacity(veryLargeRes).Obj(),
			},
			preemptors: []Preemptor{
				newPodGroupPreemptor(highPriority,
					[]*v1.Pod{
						st.MakePod().Name("pr1").UID("pr1").WorkloadRef(w1).Priority(highPriority).Obj(),
						st.MakePod().Name("pr2").UID("pr2").WorkloadRef(w1).Priority(highPriority).Obj(),
					}, nil),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
			},
			numViolatingVictim: 0,
			expected: [][]Candidate{
				{
					&candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
								st.MakePod().Name("p2").UID("p2").Node("node2").Priority(midPriority).Obj(),
							},
						},
						name: "Cluster-Scope-pg1",
					},
				},
			},
			workloadAwarePreemption: true,
		},
		{
			name: "pod group as preemptor and whole cluster as domain with pod group victim",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(veryLargeRes).Obj(),
				st.MakeNode().Name("node2").Capacity(veryLargeRes).Obj(),
			},
			preemptors: []Preemptor{
				newPodGroupPreemptor(highPriority,
					[]*v1.Pod{
						st.MakePod().Name("pr1").UID("pr1").WorkloadRef(w1).Priority(highPriority).Obj(),
						st.MakePod().Name("pr2").UID("pr2").WorkloadRef(w1).Priority(highPriority).Obj(),
					}, nil),
			},
			initPods: []*v1.Pod{
				// Victim PodGroup (Workload w2) spread across nodes
				st.MakePod().Name("p1").UID("p1").Node("node1").WorkloadRef(w2).Priority(midPriority).Obj(),
				st.MakePod().Name("p2").UID("p2").Node("node2").WorkloadRef(w2).Priority(midPriority).Obj(),
			},
			numViolatingVictim: 0,
			expected: [][]Candidate{
				{
					&candidate{
						victims: &extenderv1.Victims{
							Pods: []*v1.Pod{
								st.MakePod().Name("p1").UID("p1").Node("node1").WorkloadRef(w2).Priority(midPriority).Obj(),
								st.MakePod().Name("p2").UID("p2").Node("node2").WorkloadRef(w2).Priority(midPriority).Obj(),
							},
						},
						name: "Cluster-Scope-pg1",
					},
				},
			},
			workloadAwarePreemption: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			registeredPlugins := append([]tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New)},
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			var objs []runtime.Object
			var preemptorPods []*v1.Pod
			for _, preemptor := range tt.preemptors {
				preemptorPods = append(preemptorPods, preemptor.Members()...)
			}

			for _, p := range append(preemptorPods, tt.initPods...) {
				objs = append(objs, p)
			}

			for _, n := range tt.nodes {
				objs = append(objs, n)
			}
			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(objs...), 0)
			parallelism := parallelize.DefaultParallelism
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fwk, err := tf.NewFramework(
				ctx,
				registeredPlugins, "",
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithParallelism(parallelism),
				frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot(preemptorPods, tt.nodes)),
				frameworkruntime.WithLogger(logger),
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

			for cycle, preemptor := range tt.preemptors {
				state := framework.NewCycleState()
				pe := Evaluator{
					PluginName:                    "FakePostFilter",
					Handler:                       fwk,
					Interface:                     fakePostPlugin,
					EnableWorkloadAwarePreemption: tt.workloadAwarePreemption,
				}
				got, _, _ := pe.DryRunPreemption(ctx, state, preemptor, pe.NewDomains(preemptor, nodeInfos), nil, 0, int32(len(nodeInfos)))
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

func TestSelectCandidate(t *testing.T) {
	tests := []struct {
		name                    string
		nodeNames               []string
		preemptors              []Preemptor
		initPods                []*v1.Pod
		expected                string
		workloadAwarePreemption bool
	}{
		{
			name:      "pod has different number of containers on each node",
			nodeNames: []string{"node1", "node2", "node3"},
			preemptors: []Preemptor{
				NewPodPreemptor(
					st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj()),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Containers([]v1.Container{
					st.MakeContainer().Name("container1").Obj(),
					st.MakeContainer().Name("container2").Obj(),
				}).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Containers([]v1.Container{
					st.MakeContainer().Name("container1").Obj(),
				}).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Containers([]v1.Container{
					st.MakeContainer().Name("container1").Obj(),
					st.MakeContainer().Name("container2").Obj(),
					st.MakeContainer().Name("container3").Obj(),
				}).Obj(),
			},
			expected: "node2",
		},
		{
			name:      "group of pods as preemptor and whole cluster as domain",
			nodeNames: []string{"node1", "node2", "node3"},
			preemptors: []Preemptor{
				newPodGroupPreemptor(
					highPriority,
					[]*v1.Pod{
						st.MakePod().Name("pr1").UID("pr1").WorkloadRef(w1).Priority(highPriority).Req(veryLargeRes).Obj(),
						st.MakePod().Name("pr2").UID("pr2").WorkloadRef(w1).Priority(highPriority).Req(veryLargeRes).Obj()},
					nil,
				),
			},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1.1").UID("p1.1").Node("node1").Priority(midPriority).Containers([]v1.Container{
					st.MakeContainer().Name("container1").Obj(),
					st.MakeContainer().Name("container2").Obj(),
				}).Obj(),
				st.MakePod().Name("p2.1").UID("p2.1").Node("node2").Priority(midPriority).Containers([]v1.Container{
					st.MakeContainer().Name("container1").Obj(),
				}).Obj(),
				st.MakePod().Name("p3.1").UID("p3.1").Node("node3").Priority(midPriority).Containers([]v1.Container{
					st.MakeContainer().Name("container1").Obj(),
					st.MakeContainer().Name("container2").Obj(),
					st.MakeContainer().Name("container3").Obj(),
				}).Obj(),
			},
			workloadAwarePreemption: true,
			expected:                "Cluster-Scope-pg1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			nodes := make([]*v1.Node, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodes[i] = st.MakeNode().Name(nodeName).Capacity(veryLargeRes).Obj()
			}
			registeredPlugins := append([]tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New)},
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			var objs []runtime.Object
			for _, pod := range tt.initPods {
				objs = append(objs, pod)
			}
			var preemptorPods []*v1.Pod
			for _, preemptor := range tt.preemptors {
				preemptorPods = append(preemptorPods, preemptor.Members()...)
			}
			for _, pod := range preemptorPods {
				objs = append(objs, pod)
			}
			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(objs...), 0)
			snapshot := internalcache.NewSnapshot(tt.initPods, nodes)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fwk, err := tf.NewFramework(
				ctx,
				registeredPlugins,
				"",
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithLogger(logger),
			)
			if err != nil {
				t.Fatal(err)
			}

			state := framework.NewCycleState()
			// Some tests rely on PreFilter plugin to compute its CycleState.
			for _, pod := range preemptorPods {
				if _, status, _ := fwk.RunPreFilterPlugins(ctx, state, pod); !status.IsSuccess() {
					t.Errorf("Unexpected PreFilter Status: %v", status)
				}
			}

			nodeInfos, err := snapshot.NodeInfos().List()
			if err != nil {
				t.Fatal(err)
			}

			fakePreemptionScorePostFilterPlugin := &FakePreemptionScorePostFilterPlugin{}

			for _, preemptor := range tt.preemptors {
				state := framework.NewCycleState()
				pe := Evaluator{
					PluginName:                    "FakePreemptionScorePostFilter",
					Handler:                       fwk,
					Interface:                     fakePreemptionScorePostFilterPlugin,
					EnableWorkloadAwarePreemption: tt.workloadAwarePreemption,
				}
				candidates, _, _ := pe.DryRunPreemption(ctx, state, preemptor, pe.NewDomains(preemptor, nodeInfos), nil, 0, int32(len(nodeInfos)))
				s := pe.SelectCandidate(ctx, candidates)
				if s == nil || len(s.Name()) == 0 {
					t.Errorf("expect any node in %v, but no candidate selected", tt.expected)
					return
				}
				if diff := cmp.Diff(tt.expected, s.Name()); diff != "" {
					t.Errorf("expect any node in %v, but got %v", tt.expected, s.Name())
				}
			}
		})
	}
}

type fakeExtender struct {
	ignorable            bool
	errProcessPreemption bool
	supportsPreemption   bool
	returnsNoVictims     bool
}

func newFakeExtender() *fakeExtender {
	return &fakeExtender{}
}

func (f *fakeExtender) WithIgnorable(ignorable bool) *fakeExtender {
	f.ignorable = ignorable
	return f
}

func (f *fakeExtender) WithErrProcessPreemption(errProcessPreemption bool) *fakeExtender {
	f.errProcessPreemption = errProcessPreemption
	return f
}

func (f *fakeExtender) WithSupportsPreemption(supportsPreemption bool) *fakeExtender {
	f.supportsPreemption = supportsPreemption
	return f
}

func (f *fakeExtender) WithReturnNoVictims(returnsNoVictims bool) *fakeExtender {
	f.returnsNoVictims = returnsNoVictims
	return f
}

func (f *fakeExtender) Name() string {
	return "fakeExtender"
}

func (f *fakeExtender) IsIgnorable() bool {
	return f.ignorable
}

func (f *fakeExtender) ProcessPreemption(
	_ *v1.Pod,
	victims map[string]*extenderv1.Victims,
	_ fwk.NodeInfoLister,
) (map[string]*extenderv1.Victims, error) {
	if f.supportsPreemption {
		if f.errProcessPreemption {
			return nil, errors.New("extender preempt error")
		}
		if f.returnsNoVictims {
			return map[string]*extenderv1.Victims{"mock": {}}, nil
		}
		return victims, nil
	}
	return nil, nil
}

func (f *fakeExtender) SupportsPreemption() bool {
	return f.supportsPreemption
}

func (f *fakeExtender) IsInterested(pod *v1.Pod) bool {
	return pod != nil
}

func (f *fakeExtender) Filter(_ *v1.Pod, _ []fwk.NodeInfo) ([]fwk.NodeInfo, extenderv1.FailedNodesMap, extenderv1.FailedNodesMap, error) {
	return nil, nil, nil, nil
}

func (f *fakeExtender) Prioritize(
	_ *v1.Pod,
	_ []fwk.NodeInfo,
) (hostPriorities *extenderv1.HostPriorityList, weight int64, err error) {
	return nil, 0, nil
}

func (f *fakeExtender) Bind(_ *v1.Binding) error {
	return nil
}

func (f *fakeExtender) IsBinder() bool {
	return true
}

func (f *fakeExtender) IsPrioritizer() bool {
	return true
}

func (f *fakeExtender) IsFilter() bool {
	return true
}

func TestCallExtenders(t *testing.T) {
	var (
		node1Name            = "node1"
		defaultSchedulerName = "default-scheduler"
		singlePreemptor      = NewPodPreemptor(st.MakePod().Name("preemptor").UID("preemptor").
					SchedulerName(defaultSchedulerName).Priority(highPriority).
					Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
					Obj())
		victim = st.MakePod().Name("victim").UID("victim").
			Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()
		makeCandidates = func(nodeName string, pods ...*v1.Pod) []Candidate {
			return []Candidate{
				&candidate{
					name: nodeName,
					victims: &extenderv1.Victims{
						Pods: pods,
					},
				},
			}
		}
	)
	tests := []struct {
		name           string
		extenders      []fwk.Extender
		candidates     []Candidate
		wantStatus     *fwk.Status
		wantCandidates []Candidate
	}{
		{
			name:           "no extenders",
			extenders:      []fwk.Extender{},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     nil,
			wantCandidates: makeCandidates(node1Name, victim),
		},
		{
			name: "one extender supports preemption",
			extenders: []fwk.Extender{
				newFakeExtender().WithSupportsPreemption(true),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     nil,
			wantCandidates: makeCandidates(node1Name, victim),
		},
		{
			name: "one extender with return no victims",
			extenders: []fwk.Extender{
				newFakeExtender().WithSupportsPreemption(true).WithReturnNoVictims(true),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     fwk.AsStatus(fmt.Errorf("expected at least one victim pod on node %q", node1Name)),
			wantCandidates: []Candidate{},
		},
		{
			name: "one extender does not support preemption",
			extenders: []fwk.Extender{
				newFakeExtender().WithSupportsPreemption(false),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     nil,
			wantCandidates: makeCandidates(node1Name, victim),
		},
		{
			name: "one extender with no return victims and is ignorable",
			extenders: []fwk.Extender{
				newFakeExtender().WithSupportsPreemption(true).
					WithReturnNoVictims(true).WithIgnorable(true),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     nil,
			wantCandidates: []Candidate{},
		},
		{
			name: "one extender returns error and is ignorable",
			extenders: []fwk.Extender{
				newFakeExtender().WithIgnorable(true).
					WithSupportsPreemption(true).WithErrProcessPreemption(true),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     nil,
			wantCandidates: makeCandidates(node1Name, victim),
		},
		{
			name: "one extender returns error and is not ignorable",
			extenders: []fwk.Extender{
				newFakeExtender().WithErrProcessPreemption(true).
					WithSupportsPreemption(true),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     fwk.AsStatus(fmt.Errorf("extender preempt error")),
			wantCandidates: nil,
		},
		{
			name: "one extender with empty victims input",
			extenders: []fwk.Extender{
				newFakeExtender().WithSupportsPreemption(true),
			},
			candidates:     []Candidate{},
			wantStatus:     nil,
			wantCandidates: []Candidate{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics.Register()
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			nodes := make([]*v1.Node, len([]string{node1Name}))
			for i, nodeName := range []string{node1Name} {
				nodes[i] = st.MakeNode().Name(nodeName).Capacity(veryLargeRes).Obj()
			}
			registeredPlugins := append([]tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New)},
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			var objs []runtime.Object
			singlePreemptorPod := singlePreemptor.GetRepresentativePod()
			objs = append(objs, singlePreemptorPod)
			cs := clientsetfake.NewClientset(objs...)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			apiDispatcher := apidispatcher.New(cs, 16, apicalls.Relevances)
			apiDispatcher.Run(logger)
			defer apiDispatcher.Close()

			fwk, err := tf.NewFramework(
				ctx,
				registeredPlugins, "",
				frameworkruntime.WithClientSet(cs),
				frameworkruntime.WithAPIDispatcher(apiDispatcher),
				frameworkruntime.WithLogger(logger),
				frameworkruntime.WithExtenders(tt.extenders),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot([]*v1.Pod{singlePreemptorPod}, nodes)),
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
			)
			if err != nil {
				t.Fatal(err)
			}
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			cache := internalcache.New(ctx, apiDispatcher)
			fwk.SetAPICacher(apicache.New(nil, cache))

			fakePreemptionScorePostFilterPlugin := &FakePreemptionScorePostFilterPlugin{}
			pe := Evaluator{
				PluginName: "FakePreemptionScorePostFilter",
				Handler:    fwk,
				Interface:  fakePreemptionScorePostFilterPlugin,
			}
			gotCandidates, status := pe.callExtenders(logger, singlePreemptor, tt.candidates)
			if (tt.wantStatus == nil) != (status == nil) || status.Code() != tt.wantStatus.Code() {
				t.Errorf("callExtenders() status mismatch. got: %v, want: %v", status, tt.wantStatus)
			}

			if len(gotCandidates) != len(tt.wantCandidates) {
				t.Errorf("callExtenders() returned unexpected number of results. got: %d, want: %d", len(gotCandidates), len(tt.wantCandidates))
			} else {
				for i, gotCandidate := range gotCandidates {
					wantCandidate := tt.wantCandidates[i]
					if gotCandidate.Name() != wantCandidate.Name() {
						t.Errorf("callExtenders() node name mismatch. got: %s, want: %s", gotCandidate.Name(), wantCandidate.Name())
					}
					if len(gotCandidate.Victims().Pods) != len(wantCandidate.Victims().Pods) {
						t.Errorf("callExtenders() number of victim pods mismatch for node %s. got: %d, want: %d", gotCandidate.Name(), len(gotCandidate.Victims().Pods), len(wantCandidate.Victims().Pods))
					}
				}
			}
		})
	}
}

func TestBuildPodGroupIndex(t *testing.T) {
	tests := []struct {
		name           string
		enable         bool
		nodeNames      []string
		pods           []*v1.Pod
		expectedKeys   []util.PodGroupKey
		expectedCounts map[util.PodGroupKey]int
	}{
		{
			nodeNames: []string{"node1"},
			name:      "Disabled: Returns empty index regardless of pods",
			enable:    false,
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Priority(highPriority).WorkloadRef(w1).Node("node1").Obj(),
				st.MakePod().Name("p2").UID("p2").Priority(highPriority).WorkloadRef(w1).Node("node1").Obj(),
			},
			expectedKeys: []util.PodGroupKey{},
		},
		{
			name:   "Basic: Groups pods by WorkloadRef",
			enable: true,
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace("default").Priority(highPriority).WorkloadRef(w1).Node("node1").Obj(),
				st.MakePod().Name("p2").UID("p2").Namespace("default").Priority(highPriority).WorkloadRef(w1).Node("node1").Obj(),
			},
			expectedKeys: []util.PodGroupKey{
				util.NewPodGroupKey("default", w1),
			},
			expectedCounts: map[util.PodGroupKey]int{
				util.NewPodGroupKey("default", w1): 2,
			},
		},
		{
			nodeNames: []string{"node1"},
			name:      "Namespace Isolation: Same group name in different namespaces are different keys",
			enable:    true,
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace("ns1").Priority(highPriority).WorkloadRef(w1).Node("node1").Obj(),
				st.MakePod().Name("p2").UID("p2").Namespace("ns2").Priority(highPriority).WorkloadRef(w1).Node("node1").Obj(),
			},
			expectedKeys: []util.PodGroupKey{
				util.NewPodGroupKey("ns1", w1),
				util.NewPodGroupKey("ns2", w1),
			},
			expectedCounts: map[util.PodGroupKey]int{
				util.NewPodGroupKey("ns1", w1): 1,
				util.NewPodGroupKey("ns2", w1): 1,
			},
		},
		{
			name:   "Mixed: Ignores pods without WorkloadRef",
			enable: true,
			pods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Namespace("default").Priority(highPriority).WorkloadRef(w1).Node("node1").Obj(),
				st.MakePod().Name("p2").UID("p2").Namespace("default").Priority(highPriority).Node("node1").Obj(),
			},
			expectedKeys: []util.PodGroupKey{
				util.NewPodGroupKey("default", w1),
			},
			expectedCounts: map[util.PodGroupKey]int{
				util.NewPodGroupKey("default", w1): 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			nodes := make([]*v1.Node, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodes[i] = st.MakeNode().Name(nodeName).Capacity(veryLargeRes).Obj()
			}
			registeredPlugins := append([]tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New)},
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			var objs []runtime.Object
			for _, pod := range tt.pods {
				objs = append(objs, pod)
			}
			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(objs...), 0)
			snapshot := internalcache.NewSnapshot(tt.pods, nodes)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fh, err := tf.NewFramework(
				ctx,
				registeredPlugins,
				"",
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithLogger(logger),
			)
			if err != nil {
				t.Fatal(err)
			}

			index, err := buildPodGroupIndex(fh, tt.enable)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if len(index) != len(tt.expectedKeys) {
				t.Errorf("Expected %d groups, got %d", len(tt.expectedKeys), len(index))
			}

			for _, key := range tt.expectedKeys {
				pods, ok := index[key]
				if !ok {
					t.Errorf("Expected key %v not found in index", key)
					continue
				}

				expectedCount := tt.expectedCounts[key]
				if len(pods) != expectedCount {
					t.Errorf("Group %v: Expected %d pods, got %d", key, expectedCount, len(pods))
				}
			}
		})
	}
}
