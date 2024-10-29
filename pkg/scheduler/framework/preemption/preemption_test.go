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
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2/ktesting"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
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

func (pl *FakePostFilterPlugin) PodEligibleToPreemptOthers(_ context.Context, pod *v1.Pod, nominatedNodeStatus *framework.Status) (bool, string) {
	return true, ""
}

func (pl *FakePostFilterPlugin) OrderedScoreFuncs(ctx context.Context, nodesToVictims map[string]*extenderv1.Victims) []func(node string) int64 {
	return nil
}

type FakePreemptionScorePostFilterPlugin struct{}

func (pl *FakePreemptionScorePostFilterPlugin) SelectVictimsOnNode(
	ctx context.Context, state *framework.CycleState, pod *v1.Pod,
	nodeInfo *framework.NodeInfo, pdbs []*policy.PodDisruptionBudget) (victims []*v1.Pod, numViolatingVictim int, status *framework.Status) {
	return append(victims, nodeInfo.Pods[0].Pod), 1, nil
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

func (pl *FakePreemptionScorePostFilterPlugin) PodEligibleToPreemptOthers(_ context.Context, pod *v1.Pod, nominatedNodeStatus *framework.Status) (bool, string) {
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

func TestDryRunPreemption(t *testing.T) {
	metrics.Register()
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
			logger, ctx := ktesting.NewTestContext(t)
			registeredPlugins := append([]tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New)},
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			var objs []runtime.Object
			for _, p := range append(tt.testPods, tt.initPods...) {
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
				frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot(tt.testPods, tt.nodes)),
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

			for cycle, pod := range tt.testPods {
				state := framework.NewCycleState()
				pe := Evaluator{
					PluginName: "FakePostFilter",
					Handler:    fwk,
					Interface:  fakePostPlugin,
					State:      state,
				}
				got, _, _ := pe.DryRunPreemption(ctx, pod, nodeInfos, nil, 0, int32(len(nodeInfos)))
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
		name      string
		nodeNames []string
		pod       *v1.Pod
		testPods  []*v1.Pod
		expected  string
	}{
		{
			name:      "pod has different number of containers on each node",
			nodeNames: []string{"node1", "node2", "node3"},
			pod:       st.MakePod().Name("p").UID("p").Priority(highPriority).Req(veryLargeRes).Obj(),
			testPods: []*v1.Pod{
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
			objs = append(objs, tt.pod)
			for _, pod := range tt.testPods {
				objs = append(objs, pod)
			}
			informerFactory := informers.NewSharedInformerFactory(clientsetfake.NewClientset(objs...), 0)
			snapshot := internalcache.NewSnapshot(tt.testPods, nodes)
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
			if _, status, _ := fwk.RunPreFilterPlugins(ctx, state, tt.pod); !status.IsSuccess() {
				t.Errorf("Unexpected PreFilter Status: %v", status)
			}
			nodeInfos, err := snapshot.NodeInfos().List()
			if err != nil {
				t.Fatal(err)
			}

			fakePreemptionScorePostFilterPlugin := &FakePreemptionScorePostFilterPlugin{}

			for _, pod := range tt.testPods {
				state := framework.NewCycleState()
				pe := Evaluator{
					PluginName: "FakePreemptionScorePostFilter",
					Handler:    fwk,
					Interface:  fakePreemptionScorePostFilterPlugin,
					State:      state,
				}
				candidates, _, _ := pe.DryRunPreemption(ctx, pod, nodeInfos, nil, 0, int32(len(nodeInfos)))
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

type fakeCandidate struct {
	victims *extenderv1.Victims
	name    string
}

// Victims returns s.victims.
func (s *fakeCandidate) Victims() *extenderv1.Victims {
	return s.victims
}

// Name returns s.name.
func (s *fakeCandidate) Name() string {
	return s.name
}

func TestPrepareCandidate(t *testing.T) {
	var (
		node1Name            = "node1"
		defaultSchedulerName = "default-scheduler"
	)
	condition := v1.PodCondition{
		Type:    v1.DisruptionTarget,
		Status:  v1.ConditionTrue,
		Reason:  v1.PodReasonPreemptionByScheduler,
		Message: fmt.Sprintf("%s: preempting to accommodate a higher priority pod", defaultSchedulerName),
	}

	var (
		victim1 = st.MakePod().Name("victim1").UID("victim1").
			Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()

		victim2 = st.MakePod().Name("victim2").UID("victim2").
			Node(node1Name).SchedulerName(defaultSchedulerName).Priority(50000).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()

		victim1WithMatchingCondition = st.MakePod().Name("victim1").UID("victim1").
						Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
						Conditions([]v1.PodCondition{condition}).
						Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
						Obj()

		preemptor = st.MakePod().Name("preemptor").UID("preemptor").
				SchedulerName(defaultSchedulerName).Priority(highPriority).
				Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
				Obj()
	)

	tests := []struct {
		name           string
		nodeNames      []string
		candidate      *fakeCandidate
		preemptor      *v1.Pod
		testPods       []*v1.Pod
		expectedStatus *framework.Status
	}{
		{
			name: "no victims",
			candidate: &fakeCandidate{
				victims: &extenderv1.Victims{},
			},
			preemptor: preemptor,
			testPods: []*v1.Pod{
				victim1,
			},
			nodeNames:      []string{node1Name},
			expectedStatus: nil,
		},
		{
			name: "one victim without condition",
			candidate: &fakeCandidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1,
					},
				},
			},
			preemptor: preemptor,
			testPods: []*v1.Pod{
				victim1,
			},
			nodeNames:      []string{node1Name},
			expectedStatus: nil,
		},
		{
			name: "one victim with same condition",
			candidate: &fakeCandidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1WithMatchingCondition,
					},
				},
			},
			preemptor: preemptor,
			testPods: []*v1.Pod{
				victim1WithMatchingCondition,
			},
			nodeNames:      []string{node1Name},
			expectedStatus: nil,
		},
		{
			name: "one victim, but patch pod failed (not found victim1 pod)",
			candidate: &fakeCandidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1WithMatchingCondition,
					},
				},
			},
			preemptor:      preemptor,
			testPods:       []*v1.Pod{},
			nodeNames:      []string{node1Name},
			expectedStatus: framework.AsStatus(errors.New("patch pod status failed")),
		},
		{
			name: "one victim, but delete pod failed (not found victim1 pod)",
			candidate: &fakeCandidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1,
					},
				},
			},
			preemptor:      preemptor,
			testPods:       []*v1.Pod{},
			nodeNames:      []string{node1Name},
			expectedStatus: framework.AsStatus(errors.New("delete pod failed")),
		},
		{
			name: "two victims without condition, one passes successfully and the second fails (not found victim2 pod)",
			candidate: &fakeCandidate{
				name: node1Name,
				victims: &extenderv1.Victims{
					Pods: []*v1.Pod{
						victim1,
						victim2,
					},
				},
			},
			preemptor: preemptor,
			testPods: []*v1.Pod{
				victim1,
			},
			nodeNames:      []string{node1Name},
			expectedStatus: framework.AsStatus(errors.New("patch pod status failed")),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics.Register()
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			nodes := make([]*v1.Node, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodes[i] = st.MakeNode().Name(nodeName).Capacity(veryLargeRes).Obj()
			}
			registeredPlugins := append([]tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New)},
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			var objs []runtime.Object
			for _, pod := range tt.testPods {
				objs = append(objs, pod)
			}
			cs := clientsetfake.NewClientset(objs...)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: cs.EventsV1()})
			fwk, err := tf.NewFramework(
				ctx,
				registeredPlugins, "",
				frameworkruntime.WithClientSet(cs),
				frameworkruntime.WithLogger(logger),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithWaitingPods(frameworkruntime.NewWaitingPodsMap()),
				frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot(tt.testPods, nodes)),
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
				frameworkruntime.WithEventRecorder(eventBroadcaster.NewRecorder(scheme.Scheme, "test-scheduler")),
			)
			if err != nil {
				t.Fatal(err)
			}
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			fakePreemptionScorePostFilterPlugin := &FakePreemptionScorePostFilterPlugin{}
			pe := Evaluator{
				PluginName: "FakePreemptionScorePostFilter",
				Handler:    fwk,
				Interface:  fakePreemptionScorePostFilterPlugin,
				State:      framework.NewCycleState(),
			}

			status := pe.prepareCandidate(ctx, tt.candidate, tt.preemptor, "test-plugin")
			if tt.expectedStatus == nil {
				if status != nil {
					t.Errorf("expect nil status, but got %v", status)
				}
			} else {
				if status == nil {
					t.Errorf("expect status %v, but got nil", tt.expectedStatus)
				} else if status.Code() != tt.expectedStatus.Code() {
					t.Errorf("expect status code %v, but got %v", tt.expectedStatus.Code(), status.Code())
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
	_ framework.NodeInfoLister,
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

func (f *fakeExtender) Filter(_ *v1.Pod, _ []*framework.NodeInfo) ([]*framework.NodeInfo, extenderv1.FailedNodesMap, extenderv1.FailedNodesMap, error) {
	return nil, nil, nil, nil
}

func (f *fakeExtender) Prioritize(
	_ *v1.Pod,
	_ []*framework.NodeInfo,
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
		preemptor            = st.MakePod().Name("preemptor").UID("preemptor").
					SchedulerName(defaultSchedulerName).Priority(highPriority).
					Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
					Obj()
		victim = st.MakePod().Name("victim").UID("victim").
			Node(node1Name).SchedulerName(defaultSchedulerName).Priority(midPriority).
			Containers([]v1.Container{st.MakeContainer().Name("container1").Obj()}).
			Obj()
		makeCandidates = func(nodeName string, pods ...*v1.Pod) []Candidate {
			return []Candidate{
				&fakeCandidate{
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
		extenders      []framework.Extender
		candidates     []Candidate
		wantStatus     *framework.Status
		wantCandidates []Candidate
	}{
		{
			name:           "no extenders",
			extenders:      []framework.Extender{},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     nil,
			wantCandidates: makeCandidates(node1Name, victim),
		},
		{
			name: "one extender supports preemption",
			extenders: []framework.Extender{
				newFakeExtender().WithSupportsPreemption(true),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     nil,
			wantCandidates: makeCandidates(node1Name, victim),
		},
		{
			name: "one extender with return no victims",
			extenders: []framework.Extender{
				newFakeExtender().WithSupportsPreemption(true).WithReturnNoVictims(true),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     framework.AsStatus(fmt.Errorf("expected at least one victim pod on node %q", node1Name)),
			wantCandidates: []Candidate{},
		},
		{
			name: "one extender does not support preemption",
			extenders: []framework.Extender{
				newFakeExtender().WithSupportsPreemption(false),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     nil,
			wantCandidates: makeCandidates(node1Name, victim),
		},
		{
			name: "one extender with no return victims and is ignorable",
			extenders: []framework.Extender{
				newFakeExtender().WithSupportsPreemption(true).
					WithReturnNoVictims(true).WithIgnorable(true),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     nil,
			wantCandidates: []Candidate{},
		},
		{
			name: "one extender returns error and is ignorable",
			extenders: []framework.Extender{
				newFakeExtender().WithIgnorable(true).
					WithSupportsPreemption(true).WithErrProcessPreemption(true),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     nil,
			wantCandidates: makeCandidates(node1Name, victim),
		},
		{
			name: "one extender returns error and is not ignorable",
			extenders: []framework.Extender{
				newFakeExtender().WithErrProcessPreemption(true).
					WithSupportsPreemption(true),
			},
			candidates:     makeCandidates(node1Name, victim),
			wantStatus:     framework.AsStatus(fmt.Errorf("extender preempt error")),
			wantCandidates: nil,
		},
		{
			name: "one extender with empty victims input",
			extenders: []framework.Extender{
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
			objs = append(objs, preemptor)
			cs := clientsetfake.NewClientset(objs...)
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			fwk, err := tf.NewFramework(
				ctx,
				registeredPlugins, "",
				frameworkruntime.WithClientSet(cs),
				frameworkruntime.WithLogger(logger),
				frameworkruntime.WithExtenders(tt.extenders),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(internalcache.NewSnapshot([]*v1.Pod{preemptor}, nodes)),
				frameworkruntime.WithPodNominator(internalqueue.NewSchedulingQueue(nil, informerFactory)),
			)
			if err != nil {
				t.Fatal(err)
			}
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			fakePreemptionScorePostFilterPlugin := &FakePreemptionScorePostFilterPlugin{}
			pe := Evaluator{
				PluginName: "FakePreemptionScorePostFilter",
				Handler:    fwk,
				Interface:  fakePreemptionScorePostFilterPlugin,
				State:      framework.NewCycleState(),
			}
			gotCandidates, status := pe.callExtenders(logger, preemptor, tt.candidates)
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
