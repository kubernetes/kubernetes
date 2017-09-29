/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1beta1"
	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	algorithmpredicates "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	algorithmpriorities "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
	schedulertesting "k8s.io/kubernetes/plugin/pkg/scheduler/testing"
)

func falsePredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	return false, []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
}

func truePredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	return true, nil, nil
}

func matchesPredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}
	if pod.Name == node.Name {
		return true, nil, nil
	}
	return false, []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
}

func hasNoPodsPredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	if len(nodeInfo.Pods()) == 0 {
		return true, nil, nil
	}
	return false, []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate}, nil
}

func numericPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	result := []schedulerapi.HostPriority{}
	for _, node := range nodes {
		score, err := strconv.Atoi(node.Name)
		if err != nil {
			return nil, err
		}
		result = append(result, schedulerapi.HostPriority{
			Host:  node.Name,
			Score: score,
		})
	}
	return result, nil
}

func reverseNumericPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	var maxScore float64
	minScore := math.MaxFloat64
	reverseResult := []schedulerapi.HostPriority{}
	result, err := numericPriority(pod, nodeNameToInfo, nodes)
	if err != nil {
		return nil, err
	}

	for _, hostPriority := range result {
		maxScore = math.Max(maxScore, float64(hostPriority.Score))
		minScore = math.Min(minScore, float64(hostPriority.Score))
	}
	for _, hostPriority := range result {
		reverseResult = append(reverseResult, schedulerapi.HostPriority{
			Host:  hostPriority.Host,
			Score: int(maxScore + minScore - float64(hostPriority.Score)),
		})
	}

	return reverseResult, nil
}

func makeNodeList(nodeNames []string) []*v1.Node {
	result := make([]*v1.Node, 0, len(nodeNames))
	for _, nodeName := range nodeNames {
		result = append(result, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
	}
	return result
}

func TestSelectHost(t *testing.T) {
	scheduler := genericScheduler{}
	tests := []struct {
		list          schedulerapi.HostPriorityList
		possibleHosts sets.String
		expectsErr    bool
	}{
		{
			list: []schedulerapi.HostPriority{
				{Host: "machine1.1", Score: 1},
				{Host: "machine2.1", Score: 2},
			},
			possibleHosts: sets.NewString("machine2.1"),
			expectsErr:    false,
		},
		// equal scores
		{
			list: []schedulerapi.HostPriority{
				{Host: "machine1.1", Score: 1},
				{Host: "machine1.2", Score: 2},
				{Host: "machine1.3", Score: 2},
				{Host: "machine2.1", Score: 2},
			},
			possibleHosts: sets.NewString("machine1.2", "machine1.3", "machine2.1"),
			expectsErr:    false,
		},
		// out of order scores
		{
			list: []schedulerapi.HostPriority{
				{Host: "machine1.1", Score: 3},
				{Host: "machine1.2", Score: 3},
				{Host: "machine2.1", Score: 2},
				{Host: "machine3.1", Score: 1},
				{Host: "machine1.3", Score: 3},
			},
			possibleHosts: sets.NewString("machine1.1", "machine1.2", "machine1.3"),
			expectsErr:    false,
		},
		// empty priorityList
		{
			list:          []schedulerapi.HostPriority{},
			possibleHosts: sets.NewString(),
			expectsErr:    true,
		},
	}

	for _, test := range tests {
		// increase the randomness
		for i := 0; i < 10; i++ {
			got, err := scheduler.selectHost(test.list)
			if test.expectsErr {
				if err == nil {
					t.Error("Unexpected non-error")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if !test.possibleHosts.Has(got) {
					t.Errorf("got %s is not in the possible map %v", got, test.possibleHosts)
				}
			}
		}
	}
}

func TestGenericScheduler(t *testing.T) {
	tests := []struct {
		name          string
		predicates    map[string]algorithm.FitPredicate
		prioritizers  []algorithm.PriorityConfig
		nodes         []string
		pod           *v1.Pod
		pods          []*v1.Pod
		expectedHosts sets.String
		expectsErr    bool
		wErr          error
	}{
		{
			predicates:   map[string]algorithm.FitPredicate{"false": falsePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			expectsErr:   true,
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			name:         "test 1",
			wErr: &FitError{
				Pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
				FailedPredicates: FailedPredicateMap{
					"machine1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"machine2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
				}},
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:         []string{"machine1", "machine2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "ignore"}},
			expectedHosts: sets.NewString("machine1", "machine2"),
			name:          "test 2",
			wErr:          nil,
		},
		{
			// Fits on a machine where the pod ID matches the machine name
			predicates:    map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			prioritizers:  []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			nodes:         []string{"machine1", "machine2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine2"}},
			expectedHosts: sets.NewString("machine2"),
			name:          "test 3",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "ignore"}},
			expectedHosts: sets.NewString("3"),
			name:          "test 4",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			expectedHosts: sets.NewString("2"),
			name:          "test 5",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}, {Function: reverseNumericPriority, Weight: 2}},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			expectedHosts: sets.NewString("1"),
			name:          "test 6",
			wErr:          nil,
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate, "false": falsePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"3", "2", "1"},
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			expectsErr:   true,
			name:         "test 7",
			wErr: &FitError{
				Pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
				FailedPredicates: FailedPredicateMap{
					"3": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
				},
			},
		},
		{
			predicates: map[string]algorithm.FitPredicate{
				"nopods":  hasNoPodsPredicate,
				"matches": matchesPredicate,
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "2"},
					Spec: v1.PodSpec{
						NodeName: "2",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
			},
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"1", "2"},
			expectsErr:   true,
			name:         "test 8",
			wErr: &FitError{
				Pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
				FailedPredicates: FailedPredicateMap{
					"1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
					"2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrFakePredicate},
				},
			},
		},
	}
	for _, test := range tests {
		cache := schedulercache.New(time.Duration(0), wait.NeverStop)
		for _, pod := range test.pods {
			cache.AddPod(pod)
		}
		for _, name := range test.nodes {
			cache.AddNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}})
		}

		scheduler := NewGenericScheduler(
			cache, nil, test.predicates, algorithm.EmptyPredicateMetadataProducer, test.prioritizers, algorithm.EmptyMetadataProducer, []algorithm.SchedulerExtender{})
		machine, err := scheduler.Schedule(test.pod, schedulertesting.FakeNodeLister(makeNodeList(test.nodes)))

		if !reflect.DeepEqual(err, test.wErr) {
			t.Errorf("Failed : %s, Unexpected error: %v, expected: %v", test.name, err, test.wErr)
		}
		if test.expectedHosts != nil && !test.expectedHosts.Has(machine) {
			t.Errorf("Failed : %s, Expected: %s, got: %s", test.name, test.expectedHosts, machine)
		}
	}
}

func TestFindFitAllError(t *testing.T) {
	nodes := []string{"3", "2", "1"}
	predicates := map[string]algorithm.FitPredicate{"true": truePredicate, "false": falsePredicate}
	nodeNameToInfo := map[string]*schedulercache.NodeInfo{
		"3": schedulercache.NewNodeInfo(),
		"2": schedulercache.NewNodeInfo(),
		"1": schedulercache.NewNodeInfo(),
	}
	_, predicateMap, err := findNodesThatFit(&v1.Pod{}, nodeNameToInfo, makeNodeList(nodes), predicates, nil, algorithm.EmptyPredicateMetadataProducer, nil)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(predicateMap) != len(nodes) {
		t.Errorf("unexpected failed predicate map: %v", predicateMap)
	}

	for _, node := range nodes {
		failures, found := predicateMap[node]
		if !found {
			t.Errorf("failed to find node: %s in %v", node, predicateMap)
		}
		if len(failures) != 1 || failures[0] != algorithmpredicates.ErrFakePredicate {
			t.Errorf("unexpected failures: %v", failures)
		}
	}
}

func TestFindFitSomeError(t *testing.T) {
	nodes := []string{"3", "2", "1"}
	predicates := map[string]algorithm.FitPredicate{"true": truePredicate, "match": matchesPredicate}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "1"}}
	nodeNameToInfo := map[string]*schedulercache.NodeInfo{
		"3": schedulercache.NewNodeInfo(),
		"2": schedulercache.NewNodeInfo(),
		"1": schedulercache.NewNodeInfo(pod),
	}
	for name := range nodeNameToInfo {
		nodeNameToInfo[name].SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}})
	}

	_, predicateMap, err := findNodesThatFit(pod, nodeNameToInfo, makeNodeList(nodes), predicates, nil, algorithm.EmptyPredicateMetadataProducer, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(predicateMap) != (len(nodes) - 1) {
		t.Errorf("unexpected failed predicate map: %v", predicateMap)
	}

	for _, node := range nodes {
		if node == pod.Name {
			continue
		}
		failures, found := predicateMap[node]
		if !found {
			t.Errorf("failed to find node: %s in %v", node, predicateMap)
		}
		if len(failures) != 1 || failures[0] != algorithmpredicates.ErrFakePredicate {
			t.Errorf("unexpected failures: %v", failures)
		}
	}
}

func makeNode(node string, milliCPU, memory int64) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: node},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
				"pods":            *resource.NewQuantity(100, resource.DecimalSI),
			},
			Allocatable: v1.ResourceList{

				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
				"pods":            *resource.NewQuantity(100, resource.DecimalSI),
			},
		},
	}
}

func TestHumanReadableFitError(t *testing.T) {
	err := &FitError{
		Pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
		FailedPredicates: FailedPredicateMap{
			"1": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnderMemoryPressure},
			"2": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnderDiskPressure},
			"3": []algorithm.PredicateFailureReason{algorithmpredicates.ErrNodeUnderDiskPressure},
		},
	}
	if strings.Contains(err.Error(), NoNodeAvailableMsg) {
		if strings.Contains(err.Error(), "NodeUnderDiskPressure (2)") && strings.Contains(err.Error(), "NodeUnderMemoryPressure (1)") {
			return
		}
	}
	t.Errorf("Error message doesn't have all the information content: [" + err.Error() + "]")
}

// The point of this test is to show that you:
// - get the same priority for a zero-request pod as for a pod with the defaults requests,
//   both when the zero-request pod is already on the machine and when the zero-request pod
//   is the one being scheduled.
// - don't get the same score no matter what we schedule.
func TestZeroRequest(t *testing.T) {
	// A pod with no resources. We expect spreading to count it as having the default resources.
	noResources := v1.PodSpec{
		Containers: []v1.Container{
			{},
		},
	}
	noResources1 := noResources
	noResources1.NodeName = "machine1"
	// A pod with the same resources as a 0-request pod gets by default as its resources (for spreading).
	small := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMilliCpuRequest, 10) + "m"),
						v1.ResourceMemory: resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMemoryRequest, 10)),
					},
				},
			},
		},
	}
	small2 := small
	small2.NodeName = "machine2"
	// A larger pod.
	large := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMilliCpuRequest*3, 10) + "m"),
						v1.ResourceMemory: resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMemoryRequest*3, 10)),
					},
				},
			},
		},
	}
	large1 := large
	large1.NodeName = "machine1"
	large2 := large
	large2.NodeName = "machine2"
	tests := []struct {
		pod   *v1.Pod
		pods  []*v1.Pod
		nodes []*v1.Node
		test  string
	}{
		// The point of these next two tests is to show you get the same priority for a zero-request pod
		// as for a pod with the defaults requests, both when the zero-request pod is already on the machine
		// and when the zero-request pod is the one being scheduled.
		{
			pod:   &v1.Pod{Spec: noResources},
			nodes: []*v1.Node{makeNode("machine1", 1000, priorityutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, priorityutil.DefaultMemoryRequest*10)},
			test:  "test priority of zero-request pod with machine with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
		},
		{
			pod:   &v1.Pod{Spec: small},
			nodes: []*v1.Node{makeNode("machine1", 1000, priorityutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, priorityutil.DefaultMemoryRequest*10)},
			test:  "test priority of nonzero-request pod with machine with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
		},
		// The point of this test is to verify that we're not just getting the same score no matter what we schedule.
		{
			pod:   &v1.Pod{Spec: large},
			nodes: []*v1.Node{makeNode("machine1", 1000, priorityutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, priorityutil.DefaultMemoryRequest*10)},
			test:  "test priority of larger pod with machine with zero-request pod",
			pods: []*v1.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
		},
	}

	const expectedPriority int = 25
	for _, test := range tests {
		// This should match the configuration in defaultPriorities() in
		// plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go if you want
		// to test what's actually in production.
		priorityConfigs := []algorithm.PriorityConfig{
			{Map: algorithmpriorities.LeastRequestedPriorityMap, Weight: 1},
			{Map: algorithmpriorities.BalancedResourceAllocationMap, Weight: 1},
			{
				Function: algorithmpriorities.NewSelectorSpreadPriority(
					schedulertesting.FakeServiceLister([]*v1.Service{}),
					schedulertesting.FakeControllerLister([]*v1.ReplicationController{}),
					schedulertesting.FakeReplicaSetLister([]*extensions.ReplicaSet{}),
					schedulertesting.FakeStatefulSetLister([]*apps.StatefulSet{})),
				Weight: 1,
			},
		}
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)
		list, err := PrioritizeNodes(
			test.pod, nodeNameToInfo, algorithm.EmptyMetadataProducer, priorityConfigs,
			schedulertesting.FakeNodeLister(test.nodes), []algorithm.SchedulerExtender{})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		for _, hp := range list {
			if test.test == "test priority of larger pod with machine with zero-request pod" {
				if hp.Score == expectedPriority {
					t.Errorf("%s: expected non-%d for all priorities, got list %#v", test.test, expectedPriority, list)
				}
			} else {
				if hp.Score != expectedPriority {
					t.Errorf("%s: expected %d for all priorities, got list %#v", test.test, expectedPriority, list)
				}
			}
		}
	}
}

func printNodeToPods(nodeToPods map[*v1.Node][]*v1.Pod) string {
	var output string
	for node, pods := range nodeToPods {
		output += node.Name + ": ["
		for _, pod := range pods {
			output += pod.Name + ", "
		}
		output += "]"
	}
	return output
}

func checkPreemptionVictims(testName string, expected map[string]map[string]bool, nodeToPods map[*v1.Node][]*v1.Pod) error {
	if len(expected) == len(nodeToPods) {
		for k, pods := range nodeToPods {
			if expPods, ok := expected[k.Name]; ok {
				if len(pods) != len(expPods) {
					return fmt.Errorf("test [%v]: unexpected number of pods. expected: %v, got: %v", testName, expected, printNodeToPods(nodeToPods))
				}
				prevPriority := int32(math.MaxInt32)
				for _, p := range pods {
					// Check that pods are sorted by their priority.
					if *p.Spec.Priority > prevPriority {
						return fmt.Errorf("test [%v]: pod %v of node %v was not sorted by priority", testName, p.Name, k)
					}
					prevPriority = *p.Spec.Priority
					if _, ok := expPods[p.Name]; !ok {
						return fmt.Errorf("test [%v]: pod %v was not expected. Expected: %v", testName, p.Name, expPods)
					}
				}
			} else {
				return fmt.Errorf("test [%v]: unexpected machines. expected: %v, got: %v", testName, expected, printNodeToPods(nodeToPods))
			}
		}
	} else {
		return fmt.Errorf("test [%v]: unexpected number of machines. expected: %v, got: %v", testName, expected, printNodeToPods(nodeToPods))
	}
	return nil
}

type FakeNodeInfo v1.Node

func (n FakeNodeInfo) GetNodeInfo(nodeName string) (*v1.Node, error) {
	node := v1.Node(n)
	return &node, nil
}

func PredicateMetadata(p *v1.Pod, nodeInfo map[string]*schedulercache.NodeInfo) algorithm.PredicateMetadata {
	return algorithmpredicates.NewPredicateMetadataFactory(schedulertesting.FakePodLister{p})(p, nodeInfo)
}

var smallContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMilliCpuRequest, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMemoryRequest, 10)),
			},
		},
	},
}
var mediumContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMilliCpuRequest*2, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMemoryRequest*2, 10)),
			},
		},
	},
}
var largeContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMilliCpuRequest*3, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMemoryRequest*3, 10)),
			},
		},
	},
}
var veryLargeContainers = []v1.Container{
	{
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				"cpu": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMilliCpuRequest*5, 10) + "m"),
				"memory": resource.MustParse(
					strconv.FormatInt(priorityutil.DefaultMemoryRequest*5, 10)),
			},
		},
	},
}
var negPriority, lowPriority, midPriority, highPriority, veryHighPriority = int32(-100), int32(0), int32(100), int32(1000), int32(10000)

// TestSelectNodesForPreemption tests selectNodesForPreemption. This test assumes
// that podsFitsOnNode works correctly and is tested separately.
func TestSelectNodesForPreemption(t *testing.T) {
	tests := []struct {
		name                 string
		predicates           map[string]algorithm.FitPredicate
		nodes                []string
		pod                  *v1.Pod
		pods                 []*v1.Pod
		expected             map[string]map[string]bool // Map from node name to a list of pods names which should be preempted.
		addAffinityPredicate bool
	}{
		{
			name:       "a pod that does not fit on any machine",
			predicates: map[string]algorithm.FitPredicate{"matches": falsePredicate},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new"}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a"}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b"}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{},
		},
		{
			name:       "a pod that fits with no preemption",
			predicates: map[string]algorithm.FitPredicate{"matches": truePredicate},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "new"}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a"}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b"}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {}, "machine2": {}},
		},
		{
			name:       "a pod that fits on one machine with no preemption",
			predicates: map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a"}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b"}, Spec: v1.PodSpec{Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {}},
		},
		{
			name:       "a pod that fits on both machines when lower priority pods are preempted",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {"a": true}, "machine2": {"b": true}},
		},
		{
			name:       "a pod that would fit on the machines, but other pods running are higher priority",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &lowPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{},
		},
		{
			name:       "medium priority pod is preempted, but lower priority one stays as it is small",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "c"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {"b": true}, "machine2": {"c": true}},
		},
		{
			name:       "mixed priority pods are preempted",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "c"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "d"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &highPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "e"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}}},
			expected: map[string]map[string]bool{"machine1": {"b": true, "c": true}},
		},
		{
			name:       "pod with anti-affinity is preempted",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{
				Name:   "machine1",
				Labels: map[string]string{"pod": "preemptor"}}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "a", Labels: map[string]string{"service": "securityscan"}}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1", Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "pod",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"preemptor", "value2"},
										},
									},
								},
								TopologyKey: "hostname",
							},
						},
					}}}},
				{ObjectMeta: metav1.ObjectMeta{Name: "b"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "d"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &highPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "e"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}}},
			expected:             map[string]map[string]bool{"machine1": {"a": true}, "machine2": {}},
			addAffinityPredicate: true,
		},
	}
	for _, test := range tests {
		nodes := []*v1.Node{}
		for _, n := range test.nodes {
			node := makeNode(n, priorityutil.DefaultMilliCpuRequest*5, priorityutil.DefaultMemoryRequest*5)
			node.ObjectMeta.Labels = map[string]string{"hostname": node.Name}
			nodes = append(nodes, node)
		}
		if test.addAffinityPredicate {
			test.predicates[predicates.MatchInterPodAffinity] = algorithmpredicates.NewPodAffinityPredicate(FakeNodeInfo(*nodes[0]), schedulertesting.FakePodLister(test.pods))
		}
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, nodes)
		nodeToPods, err := selectNodesForPreemption(test.pod, nodeNameToInfo, nodes, test.predicates, PredicateMetadata)
		if err != nil {
			t.Error(err)
		}
		if err := checkPreemptionVictims(test.name, test.expected, nodeToPods); err != nil {
			t.Error(err)
		}
	}
}

// TestPickOneNodeForPreemption tests pickOneNodeForPreemption.
func TestPickOneNodeForPreemption(t *testing.T) {
	tests := []struct {
		name       string
		predicates map[string]algorithm.FitPredicate
		nodes      []string
		pod        *v1.Pod
		pods       []*v1.Pod
		expected   []string // any of the items is valid
	}{
		{
			name:       "No node needs preemption",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}}},
			expected: []string{"machine1"},
		},
		{
			name:       "a pod that fits on both machines when lower priority pods are preempted",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: []string{"machine1", "machine2"},
		},
		{
			name:       "a pod that fits on a machine with no preemption",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}}},
			expected: []string{"machine3"},
		},
		{
			name:       "machine with min highest priority pod is picked",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine2"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine3"}},
			},
			expected: []string{"machine3"},
		},
		{
			name:       "when highest priorities are the same, minimum sum of priorities is picked",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine2"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}},
			},
			expected: []string{"machine2"},
		},
		{
			name:       "when highest priority and sum are the same, minimum number of pods is picked",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.3"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.4"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &negPriority, NodeName: "machine2"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.3"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}},
			},
			expected: []string{"machine2"},
		},
		{
			// pickOneNodeForPreemption adjusts pod priorities when finding the sum of the victims. This
			// test ensures that the logic works correctly.
			name:       "sum of adjusted priorities is considered",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &highPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.3"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m2.2"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &negPriority, NodeName: "machine2"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.3"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}},
			},
			expected: []string{"machine2"},
		},
		{
			name:       "non-overlapping lowest high priority, sum priorities, and number of pods",
			predicates: map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources},
			nodes:      []string{"machine1", "machine2", "machine3", "machine4"},
			pod:        &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}, Spec: v1.PodSpec{Containers: veryLargeContainers, Priority: &veryHighPriority}},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.3"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.3"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m3.4"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &lowPriority, NodeName: "machine3"}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m4.1"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine4"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m4.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine4"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m4.3"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine4"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m4.4"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &negPriority, NodeName: "machine4"}},
			},
			expected: []string{"machine1"},
		},
	}
	for _, test := range tests {
		nodes := []*v1.Node{}
		for _, n := range test.nodes {
			nodes = append(nodes, makeNode(n, priorityutil.DefaultMilliCpuRequest*5, priorityutil.DefaultMemoryRequest*5))
		}
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, nodes)
		candidateNodes, _ := selectNodesForPreemption(test.pod, nodeNameToInfo, nodes, test.predicates, PredicateMetadata)
		node := pickOneNodeForPreemption(candidateNodes)
		found := false
		for _, nodeName := range test.expected {
			if node.Name == nodeName {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("test [%v]: unexpected node: %v", test.name, node)
		}
	}
}

func TestNodesWherePreemptionMightHelp(t *testing.T) {
	// Prepare 4 node names.
	nodeNames := []string{}
	for i := 1; i < 5; i++ {
		nodeNames = append(nodeNames, fmt.Sprintf("machine%d", i))
	}

	tests := []struct {
		name          string
		failedPredMap FailedPredicateMap
		pod           *v1.Pod
		expected      map[string]bool // set of expected node names. Value is ignored.
	}{
		{
			name: "No node should be attempted",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{predicates.ErrNodeSelectorNotMatch},
				"machine2": []algorithm.PredicateFailureReason{predicates.ErrPodNotMatchHostName},
				"machine3": []algorithm.PredicateFailureReason{predicates.ErrTaintsTolerationsNotMatch},
				"machine4": []algorithm.PredicateFailureReason{predicates.ErrNodeLabelPresenceViolated},
			},
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
			expected: map[string]bool{},
		},
		{
			name: "pod affinity should be tried",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{predicates.ErrPodAffinityNotMatch},
				"machine2": []algorithm.PredicateFailureReason{predicates.ErrPodNotMatchHostName},
				"machine3": []algorithm.PredicateFailureReason{predicates.ErrNodeUnschedulable},
			},
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}, Spec: v1.PodSpec{Affinity: &v1.Affinity{
				PodAffinity: &v1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "service",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"securityscan", "value2"},
									},
								},
							},
							TopologyKey: "hostname",
						},
					},
				}}}},
			expected: map[string]bool{"machine1": true, "machine4": true},
		},
		{
			name: "pod with both pod affinity and anti-affinity should be tried",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{predicates.ErrPodAffinityNotMatch},
				"machine2": []algorithm.PredicateFailureReason{predicates.ErrPodNotMatchHostName},
			},
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}, Spec: v1.PodSpec{Affinity: &v1.Affinity{
				PodAffinity: &v1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "service",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"securityscan", "value2"},
									},
								},
							},
							TopologyKey: "hostname",
						},
					},
				},
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "service",
										Operator: metav1.LabelSelectorOpNotIn,
										Values:   []string{"blah", "foo"},
									},
								},
							},
							TopologyKey: "region",
						},
					},
				},
			}}},
			expected: map[string]bool{"machine1": true, "machine3": true, "machine4": true},
		},
		{
			name: "Mix of failed predicates works fine",
			failedPredMap: FailedPredicateMap{
				"machine1": []algorithm.PredicateFailureReason{predicates.ErrNodeSelectorNotMatch, predicates.ErrNodeOutOfDisk, predicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 500, 300)},
				"machine2": []algorithm.PredicateFailureReason{predicates.ErrPodNotMatchHostName, predicates.ErrDiskConflict},
				"machine3": []algorithm.PredicateFailureReason{predicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 600, 400)},
				"machine4": []algorithm.PredicateFailureReason{},
			},
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
			expected: map[string]bool{"machine3": true, "machine4": true},
		},
	}

	for _, test := range tests {
		nodes := nodesWherePreemptionMightHelp(test.pod, makeNodeList(nodeNames), test.failedPredMap)
		if len(test.expected) != len(nodes) {
			t.Errorf("test [%v]:number of nodes is not the same as expected. exptectd: %d, got: %d. Nodes: %v", test.name, len(test.expected), len(nodes), nodes)
		}
		for _, node := range nodes {
			if _, found := test.expected[node.Name]; !found {
				t.Errorf("test [%v]: node %v is not expected.", test.name, node.Name)
			}
		}
	}
}

func TestPreempt(t *testing.T) {
	failedPredMap := FailedPredicateMap{
		"machine1": []algorithm.PredicateFailureReason{predicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 500, 300)},
		"machine2": []algorithm.PredicateFailureReason{predicates.ErrDiskConflict},
		"machine3": []algorithm.PredicateFailureReason{predicates.NewInsufficientResourceError(v1.ResourceMemory, 1000, 600, 400)},
	}
	// Prepare 3 node names.
	nodeNames := []string{}
	for i := 1; i < 4; i++ {
		nodeNames = append(nodeNames, fmt.Sprintf("machine%d", i))
	}
	tests := []struct {
		name         string
		pod          *v1.Pod
		pods         []*v1.Pod
		extenders    []*FakeExtender
		expectedNode string
		expectedPods []string // list of preempted pods
	}{
		{
			name: "basic preemption logic",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}, Spec: v1.PodSpec{
				Containers: veryLargeContainers,
				Priority:   &highPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m3.1"}, Spec: v1.PodSpec{Containers: mediumContainers, Priority: &midPriority, NodeName: "machine3"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			expectedNode: "machine1",
			expectedPods: []string{"m1.1", "m1.2"},
		},
		{
			name: "One node doesn't need any preemption",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}, Spec: v1.PodSpec{
				Containers: veryLargeContainers,
				Priority:   &highPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &highPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			expectedNode: "machine3",
			expectedPods: []string{},
		},
		{
			name: "Scheduler extenders allow only machine1, otherwise machine3 would have been chosen",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}, Spec: v1.PodSpec{
				Containers: veryLargeContainers,
				Priority:   &highPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			extenders: []*FakeExtender{
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
				{
					predicates: []fitPredicate{machine1PredicateExtender},
				},
			},
			expectedNode: "machine1",
			expectedPods: []string{"m1.1", "m1.2"},
		},
		{
			name: "Scheduler extenders do not allow any preemption",
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}, Spec: v1.PodSpec{
				Containers: veryLargeContainers,
				Priority:   &highPriority},
			},
			pods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.1"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &midPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
				{ObjectMeta: metav1.ObjectMeta{Name: "m1.2"}, Spec: v1.PodSpec{Containers: smallContainers, Priority: &lowPriority, NodeName: "machine1"}, Status: v1.PodStatus{Phase: v1.PodRunning}},

				{ObjectMeta: metav1.ObjectMeta{Name: "m2.1"}, Spec: v1.PodSpec{Containers: largeContainers, Priority: &midPriority, NodeName: "machine2"}, Status: v1.PodStatus{Phase: v1.PodRunning}},
			},
			extenders: []*FakeExtender{
				{
					predicates: []fitPredicate{falsePredicateExtender},
				},
			},
			expectedNode: "",
			expectedPods: []string{},
		},
	}

	for _, test := range tests {
		stop := make(chan struct{})
		cache := schedulercache.New(time.Duration(0), stop)
		for _, pod := range test.pods {
			cache.AddPod(pod)
		}
		for _, name := range nodeNames {
			cache.AddNode(makeNode(name, priorityutil.DefaultMilliCpuRequest*5, priorityutil.DefaultMemoryRequest*5))
		}
		extenders := []algorithm.SchedulerExtender{}
		for _, extender := range test.extenders {
			extenders = append(extenders, extender)
		}
		scheduler := NewGenericScheduler(
			cache, nil, map[string]algorithm.FitPredicate{"matches": algorithmpredicates.PodFitsResources}, algorithm.EmptyPredicateMetadataProducer, []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}}, algorithm.EmptyMetadataProducer, extenders)
		// Call Preempt and check the expected results.
		node, victims, err := scheduler.Preempt(test.pod, schedulertesting.FakeNodeLister(makeNodeList(nodeNames)), error(&FitError{test.pod, failedPredMap}))
		if err != nil {
			t.Errorf("test [%v]: unexpected error in preemption: %v", test.name, err)
		}
		if (node != nil && node.Name != test.expectedNode) || (node == nil && len(test.expectedNode) != 0) {
			t.Errorf("test [%v]: expected node: %v, got: %v", test.name, test.expectedNode, node)
		}
		if len(victims) != len(test.expectedPods) {
			t.Errorf("test [%v]: expected %v pods, got %v.", test.name, len(test.expectedPods), len(victims))
		}
		for _, victim := range victims {
			found := false
			for _, expPod := range test.expectedPods {
				if expPod == victim.Name {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("test [%v]: pod %v is not expected to be a victim.", test.name, victim.Name)
			}
			// Mark the victims for deletion and record the preemptor's nominated node name.
			now := metav1.Now()
			victim.DeletionTimestamp = &now
			test.pod.Annotations = make(map[string]string)
			test.pod.Annotations[NominatedNodeAnnotationKey] = node.Name
		}
		// Call preempt again and make sure it doesn't preempt any more pods.
		node, victims, err = scheduler.Preempt(test.pod, schedulertesting.FakeNodeLister(makeNodeList(nodeNames)), error(&FitError{test.pod, failedPredMap}))
		if err != nil {
			t.Errorf("test [%v]: unexpected error in preemption: %v", test.name, err)
		}
		if node != nil && len(victims) > 0 {
			t.Errorf("test [%v]: didn't expect any more preemption. Node %v is selected for preemption.", test.name, node)
		}
		close(stop)
	}
}
