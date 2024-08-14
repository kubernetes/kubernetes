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
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
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

func (pl *FakePostFilterPlugin) PodEligibleToPreemptOthers(pod *v1.Pod, nominatedNodeStatus *framework.Status) (bool, string) {
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

func (pl *FakePreemptionScorePostFilterPlugin) PodEligibleToPreemptOthers(pod *v1.Pod, nominatedNodeStatus *framework.Status) (bool, string) {
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
			logger, _ := ktesting.NewTestContext(t)
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
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fwk, err := tf.NewFramework(
				ctx,
				registeredPlugins, "",
				frameworkruntime.WithPodNominator(internalqueue.NewTestPodNominator(informerFactory.Core().V1().Pods().Lister())),
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
			logger, _ := ktesting.NewTestContext(t)
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
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fwk, err := tf.NewFramework(
				ctx,
				registeredPlugins,
				"",
				frameworkruntime.WithPodNominator(internalqueue.NewTestPodNominator(informerFactory.Core().V1().Pods().Lister())),
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
				candidates, _, _ := pe.DryRunPreemption(context.Background(), pod, nodeInfos, nil, 0, int32(len(nodeInfos)))
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
