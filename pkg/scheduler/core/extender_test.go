/*
Copyright 2015 The Kubernetes Authors.

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

package core

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	extenderv1 "k8s.io/kubernetes/pkg/scheduler/apis/extender/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/listers"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

type fitPredicate func(pod *v1.Pod, node *v1.Node) (bool, error)
type priorityFunc func(pod *v1.Pod, nodes []*v1.Node) (*framework.NodeScoreList, error)

type priorityConfig struct {
	function priorityFunc
	weight   int64
}

func errorPredicateExtender(pod *v1.Pod, node *v1.Node) (bool, error) {
	return false, fmt.Errorf("Some error")
}

func falsePredicateExtender(pod *v1.Pod, node *v1.Node) (bool, error) {
	return false, nil
}

func truePredicateExtender(pod *v1.Pod, node *v1.Node) (bool, error) {
	return true, nil
}

func machine1PredicateExtender(pod *v1.Pod, node *v1.Node) (bool, error) {
	if node.Name == "machine1" {
		return true, nil
	}
	return false, nil
}

func machine2PredicateExtender(pod *v1.Pod, node *v1.Node) (bool, error) {
	if node.Name == "machine2" {
		return true, nil
	}
	return false, nil
}

func errorPrioritizerExtender(pod *v1.Pod, nodes []*v1.Node) (*framework.NodeScoreList, error) {
	return &framework.NodeScoreList{}, fmt.Errorf("Some error")
}

func machine1PrioritizerExtender(pod *v1.Pod, nodes []*v1.Node) (*framework.NodeScoreList, error) {
	result := framework.NodeScoreList{}
	for _, node := range nodes {
		score := 1
		if node.Name == "machine1" {
			score = 10
		}
		result = append(result, framework.NodeScore{Name: node.Name, Score: int64(score)})
	}
	return &result, nil
}

func machine2PrioritizerExtender(pod *v1.Pod, nodes []*v1.Node) (*framework.NodeScoreList, error) {
	result := framework.NodeScoreList{}
	for _, node := range nodes {
		score := 1
		if node.Name == "machine2" {
			score = 10
		}
		result = append(result, framework.NodeScore{Name: node.Name, Score: int64(score)})
	}
	return &result, nil
}

type machine2PrioritizerPlugin struct{}

func newMachine2PrioritizerPlugin() framework.PluginFactory {
	return func(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
		return &machine2PrioritizerPlugin{}, nil
	}
}

func (pl *machine2PrioritizerPlugin) Name() string {
	return "Machine2Prioritizer"
}

func (pl *machine2PrioritizerPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeName string) (int64, *framework.Status) {
	score := 10
	if nodeName == "machine2" {
		score = 100
	}
	return int64(score), nil
}

func (pl *machine2PrioritizerPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

type FakeExtender struct {
	predicates       []fitPredicate
	prioritizers     []priorityConfig
	weight           int64
	nodeCacheCapable bool
	filteredNodes    []*v1.Node
	unInterested     bool
	ignorable        bool

	// Cached node information for fake extender
	cachedNodeNameToInfo map[string]*schedulernodeinfo.NodeInfo
}

func (f *FakeExtender) Name() string {
	return "FakeExtender"
}

func (f *FakeExtender) IsIgnorable() bool {
	return f.ignorable
}

func (f *FakeExtender) SupportsPreemption() bool {
	// Assume preempt verb is always defined.
	return true
}

func (f *FakeExtender) ProcessPreemption(
	pod *v1.Pod,
	nodeToVictims map[*v1.Node]*extenderv1.Victims,
	nodeInfos listers.NodeInfoLister,
) (map[*v1.Node]*extenderv1.Victims, error) {
	nodeToVictimsCopy := map[*v1.Node]*extenderv1.Victims{}
	// We don't want to change the original nodeToVictims
	for k, v := range nodeToVictims {
		// In real world implementation, extender's user should have their own way to get node object
		// by name if needed (e.g. query kube-apiserver etc).
		//
		// For test purpose, we just use node from parameters directly.
		nodeToVictimsCopy[k] = v
	}

	for node, victims := range nodeToVictimsCopy {
		// Try to do preemption on extender side.
		extenderVictimPods, extendernPDBViolations, fits, err := f.selectVictimsOnNodeByExtender(pod, node)
		if err != nil {
			return nil, err
		}
		// If it's unfit after extender's preemption, this node is unresolvable by preemption overall,
		// let's remove it from potential preemption nodes.
		if !fits {
			delete(nodeToVictimsCopy, node)
		} else {
			// Append new victims to original victims
			nodeToVictimsCopy[node].Pods = append(victims.Pods, extenderVictimPods...)
			nodeToVictimsCopy[node].NumPDBViolations = victims.NumPDBViolations + int64(extendernPDBViolations)
		}
	}
	return nodeToVictimsCopy, nil
}

// selectVictimsOnNodeByExtender checks the given nodes->pods map with predicates on extender's side.
// Returns:
// 1. More victim pods (if any) amended by preemption phase of extender.
// 2. Number of violating victim (used to calculate PDB).
// 3. Fits or not after preemption phase on extender's side.
func (f *FakeExtender) selectVictimsOnNodeByExtender(pod *v1.Pod, node *v1.Node) ([]*v1.Pod, int, bool, error) {
	// If a extender support preemption but have no cached node info, let's run filter to make sure
	// default scheduler's decision still stand with given pod and node.
	if !f.nodeCacheCapable {
		fits, err := f.runPredicate(pod, node)
		if err != nil {
			return nil, 0, false, err
		}
		if !fits {
			return nil, 0, false, nil
		}
		return []*v1.Pod{}, 0, true, nil
	}

	// Otherwise, as a extender support preemption and have cached node info, we will assume cachedNodeNameToInfo is available
	// and get cached node info by given node name.
	nodeInfoCopy := f.cachedNodeNameToInfo[node.GetName()].Clone()

	var potentialVictims []*v1.Pod

	removePod := func(rp *v1.Pod) {
		nodeInfoCopy.RemovePod(rp)
	}
	addPod := func(ap *v1.Pod) {
		nodeInfoCopy.AddPod(ap)
	}
	// As the first step, remove all the lower priority pods from the node and
	// check if the given pod can be scheduled.
	podPriority := podutil.GetPodPriority(pod)
	for _, p := range nodeInfoCopy.Pods() {
		if podutil.GetPodPriority(p) < podPriority {
			potentialVictims = append(potentialVictims, p)
			removePod(p)
		}
	}
	sort.Slice(potentialVictims, func(i, j int) bool { return util.MoreImportantPod(potentialVictims[i], potentialVictims[j]) })

	// If the new pod does not fit after removing all the lower priority pods,
	// we are almost done and this node is not suitable for preemption.
	fits, err := f.runPredicate(pod, nodeInfoCopy.Node())
	if err != nil {
		return nil, 0, false, err
	}
	if !fits {
		return nil, 0, false, nil
	}

	var victims []*v1.Pod

	// TODO(harry): handle PDBs in the future.
	numViolatingVictim := 0

	reprievePod := func(p *v1.Pod) bool {
		addPod(p)
		fits, _ := f.runPredicate(pod, nodeInfoCopy.Node())
		if !fits {
			removePod(p)
			victims = append(victims, p)
		}
		return fits
	}

	// For now, assume all potential victims to be non-violating.
	// Now we try to reprieve non-violating victims.
	for _, p := range potentialVictims {
		reprievePod(p)
	}

	return victims, numViolatingVictim, true, nil
}

// runPredicate run predicates of extender one by one for given pod and node.
// Returns: fits or not.
func (f *FakeExtender) runPredicate(pod *v1.Pod, node *v1.Node) (bool, error) {
	fits := true
	var err error
	for _, predicate := range f.predicates {
		fits, err = predicate(pod, node)
		if err != nil {
			return false, err
		}
		if !fits {
			break
		}
	}
	return fits, nil
}

func (f *FakeExtender) Filter(pod *v1.Pod, nodes []*v1.Node) ([]*v1.Node, extenderv1.FailedNodesMap, error) {
	filtered := []*v1.Node{}
	failedNodesMap := extenderv1.FailedNodesMap{}
	for _, node := range nodes {
		fits, err := f.runPredicate(pod, node)
		if err != nil {
			return []*v1.Node{}, extenderv1.FailedNodesMap{}, err
		}
		if fits {
			filtered = append(filtered, node)
		} else {
			failedNodesMap[node.Name] = "FakeExtender failed"
		}
	}

	f.filteredNodes = filtered
	if f.nodeCacheCapable {
		return filtered, failedNodesMap, nil
	}
	return filtered, failedNodesMap, nil
}

func (f *FakeExtender) Prioritize(pod *v1.Pod, nodes []*v1.Node) (*extenderv1.HostPriorityList, int64, error) {
	result := extenderv1.HostPriorityList{}
	combinedScores := map[string]int64{}
	for _, prioritizer := range f.prioritizers {
		weight := prioritizer.weight
		if weight == 0 {
			continue
		}
		priorityFunc := prioritizer.function
		prioritizedList, err := priorityFunc(pod, nodes)
		if err != nil {
			return &extenderv1.HostPriorityList{}, 0, err
		}
		for _, hostEntry := range *prioritizedList {
			combinedScores[hostEntry.Name] += hostEntry.Score * weight
		}
	}
	for host, score := range combinedScores {
		result = append(result, extenderv1.HostPriority{Host: host, Score: score})
	}
	return &result, f.weight, nil
}

func (f *FakeExtender) Bind(binding *v1.Binding) error {
	if len(f.filteredNodes) != 0 {
		for _, node := range f.filteredNodes {
			if node.Name == binding.Target.Name {
				f.filteredNodes = nil
				return nil
			}
		}
		err := fmt.Errorf("Node %v not in filtered nodes %v", binding.Target.Name, f.filteredNodes)
		f.filteredNodes = nil
		return err
	}
	return nil
}

func (f *FakeExtender) IsBinder() bool {
	return true
}

func (f *FakeExtender) IsInterested(pod *v1.Pod) bool {
	return !f.unInterested
}

var _ SchedulerExtender = &FakeExtender{}

func TestGenericSchedulerWithExtenders(t *testing.T) {
	tests := []struct {
		name            string
		registerPlugins []st.RegisterPluginFunc
		extenders       []FakeExtender
		nodes           []string
		expectedResult  ScheduleResult
		expectsErr      bool
	}{
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []FakeExtender{
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
				{
					predicates: []fitPredicate{errorPredicateExtender},
				},
			},
			nodes:      []string{"machine1", "machine2"},
			expectsErr: true,
			name:       "test 1",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []FakeExtender{
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
				{
					predicates: []fitPredicate{falsePredicateExtender},
				},
			},
			nodes:      []string{"machine1", "machine2"},
			expectsErr: true,
			name:       "test 2",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []FakeExtender{
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
				{
					predicates: []fitPredicate{machine1PredicateExtender},
				},
			},
			nodes: []string{"machine1", "machine2"},
			expectedResult: ScheduleResult{
				SuggestedHost:  "machine1",
				EvaluatedNodes: 2,
				FeasibleNodes:  1,
			},
			name: "test 3",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []FakeExtender{
				{
					predicates: []fitPredicate{machine2PredicateExtender},
				},
				{
					predicates: []fitPredicate{machine1PredicateExtender},
				},
			},
			nodes:      []string{"machine1", "machine2"},
			expectsErr: true,
			name:       "test 4",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []FakeExtender{
				{
					predicates:   []fitPredicate{truePredicateExtender},
					prioritizers: []priorityConfig{{errorPrioritizerExtender, 10}},
					weight:       1,
				},
			},
			nodes: []string{"machine1"},
			expectedResult: ScheduleResult{
				SuggestedHost:  "machine1",
				EvaluatedNodes: 1,
				FeasibleNodes:  1,
			},
			name: "test 5",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []FakeExtender{
				{
					predicates:   []fitPredicate{truePredicateExtender},
					prioritizers: []priorityConfig{{machine1PrioritizerExtender, 10}},
					weight:       1,
				},
				{
					predicates:   []fitPredicate{truePredicateExtender},
					prioritizers: []priorityConfig{{machine2PrioritizerExtender, 10}},
					weight:       5,
				},
			},
			nodes: []string{"machine1", "machine2"},
			expectedResult: ScheduleResult{
				SuggestedHost:  "machine2",
				EvaluatedNodes: 2,
				FeasibleNodes:  2,
			},
			name: "test 6",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterScorePlugin("Machine2Prioritizer", newMachine2PrioritizerPlugin(), 20),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []FakeExtender{
				{
					predicates:   []fitPredicate{truePredicateExtender},
					prioritizers: []priorityConfig{{machine1PrioritizerExtender, 10}},
					weight:       1,
				},
			},
			nodes: []string{"machine1", "machine2"},
			expectedResult: ScheduleResult{
				SuggestedHost:  "machine2",
				EvaluatedNodes: 2,
				FeasibleNodes:  2,
			}, // machine2 has higher score
			name: "test 7",
		},
		{
			// Scheduler is expected to not send pod to extender in
			// Filter/Prioritize phases if the extender is not interested in
			// the pod.
			//
			// If scheduler sends the pod by mistake, the test would fail
			// because of the errors from errorPredicateExtender and/or
			// errorPrioritizerExtender.
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterScorePlugin("Machine2Prioritizer", newMachine2PrioritizerPlugin(), 1),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []FakeExtender{
				{
					predicates:   []fitPredicate{errorPredicateExtender},
					prioritizers: []priorityConfig{{errorPrioritizerExtender, 10}},
					unInterested: true,
				},
			},
			nodes:      []string{"machine1", "machine2"},
			expectsErr: false,
			expectedResult: ScheduleResult{
				SuggestedHost:  "machine2",
				EvaluatedNodes: 2,
				FeasibleNodes:  2,
			}, // machine2 has higher score
			name: "test 8",
		},
		{
			// Scheduling is expected to not fail in
			// Filter/Prioritize phases if the extender is not available and ignorable.
			//
			// If scheduler did not ignore the extender, the test would fail
			// because of the errors from errorPredicateExtender.
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []FakeExtender{
				{
					predicates: []fitPredicate{errorPredicateExtender},
					ignorable:  true,
				},
				{
					predicates: []fitPredicate{machine1PredicateExtender},
				},
			},
			nodes:      []string{"machine1", "machine2"},
			expectsErr: false,
			expectedResult: ScheduleResult{
				SuggestedHost:  "machine1",
				EvaluatedNodes: 2,
				FeasibleNodes:  1,
			},
			name: "test 9",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			extenders := []SchedulerExtender{}
			for ii := range test.extenders {
				extenders = append(extenders, &test.extenders[ii])
			}
			cache := internalcache.New(time.Duration(0), wait.NeverStop)
			for _, name := range test.nodes {
				cache.AddNode(createNode(name))
			}
			queue := internalqueue.NewSchedulingQueue(nil)

			fwk, err := st.NewFramework(test.registerPlugins, framework.WithClientSet(client))
			if err != nil {
				t.Fatal(err)
			}

			scheduler := NewGenericScheduler(
				cache,
				queue,
				emptySnapshot,
				fwk,
				extenders,
				nil,
				informerFactory.Core().V1().PersistentVolumeClaims().Lister(),
				informerFactory.Policy().V1beta1().PodDisruptionBudgets().Lister(),
				false,
				schedulerapi.DefaultPercentageOfNodesToScore,
				false)
			podIgnored := &v1.Pod{}
			result, err := scheduler.Schedule(context.Background(), framework.NewCycleState(), podIgnored)
			if test.expectsErr {
				if err == nil {
					t.Errorf("Unexpected non-error, result %+v", result)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
					return
				}

				if !reflect.DeepEqual(result, test.expectedResult) {
					t.Errorf("Expected: %+v, Saw: %+v", test.expectedResult, result)
				}
			}
		})
	}
}

func createNode(name string) *v1.Node {
	return &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}}
}

func TestIsInterested(t *testing.T) {
	mem := &HTTPExtender{
		managedResources: sets.NewString(),
	}
	mem.managedResources.Insert("memory")

	for _, tc := range []struct {
		label    string
		extender *HTTPExtender
		pod      *v1.Pod
		want     bool
	}{
		{
			label: "Empty managed resources",
			extender: &HTTPExtender{
				managedResources: sets.NewString(),
			},
			pod:  &v1.Pod{},
			want: true,
		},
		{
			label:    "Managed memory, empty resources",
			extender: mem,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "app",
						},
					},
				},
			},
			want: false,
		},
		{
			label:    "Managed memory, container memory",
			extender: mem,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "app",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{"memory": resource.Quantity{}},
								Limits:   v1.ResourceList{"memory": resource.Quantity{}},
							},
						},
					},
				},
			},
			want: true,
		},
		{
			label:    "Managed memory, init container memory",
			extender: mem,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "app",
						},
					},
					InitContainers: []v1.Container{
						{
							Name: "init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{"memory": resource.Quantity{}},
								Limits:   v1.ResourceList{"memory": resource.Quantity{}},
							},
						},
					},
				},
			},
			want: true,
		},
	} {
		t.Run(tc.label, func(t *testing.T) {
			if got := tc.extender.IsInterested(tc.pod); got != tc.want {
				t.Fatalf("IsInterested(%v) = %v, wanted %v", tc.pod, got, tc.want)
			}
		})
	}
}
