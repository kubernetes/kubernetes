/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package scheduler

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strconv"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	algorithmpredicates "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func falsePredicate(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	return false, algorithmpredicates.ErrFakePredicate
}

func truePredicate(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	return true, nil
}

func matchesPredicate(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, fmt.Errorf("node not found")
	}
	if pod.Name == node.Name {
		return true, nil
	}
	return false, algorithmpredicates.ErrFakePredicate
}

func hasNoPodsPredicate(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	if len(nodeInfo.Pods()) == 0 {
		return true, nil
	}
	return false, algorithmpredicates.ErrFakePredicate
}

func numericPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	nodes, err := nodeLister.List()
	result := []schedulerapi.HostPriority{}

	if err != nil {
		return nil, fmt.Errorf("failed to list nodes: %v", err)
	}
	for _, node := range nodes.Items {
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

func reverseNumericPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	var maxScore float64
	minScore := math.MaxFloat64
	reverseResult := []schedulerapi.HostPriority{}
	result, err := numericPriority(pod, nodeNameToInfo, nodeLister)
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

func makeNodeList(nodeNames []string) api.NodeList {
	result := api.NodeList{
		Items: make([]api.Node, len(nodeNames)),
	}
	for ix := range nodeNames {
		result.Items[ix].Name = nodeNames[ix]
	}
	return result
}

func TestSelectHost(t *testing.T) {
	scheduler := genericScheduler{random: rand.New(rand.NewSource(0))}
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
		pod           *api.Pod
		pods          []*api.Pod
		expectedHosts sets.String
		expectsErr    bool
		wErr          error
	}{
		{
			predicates:   map[string]algorithm.FitPredicate{"false": falsePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: EqualPriority, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			expectsErr:   true,
			name:         "test 1",
			wErr:         algorithmpredicates.ErrFakePredicate,
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: EqualPriority, Weight: 1}},
			nodes:         []string{"machine1", "machine2"},
			expectedHosts: sets.NewString("machine1", "machine2"),
			name:          "test 2",
			wErr:          nil,
		},
		{
			// Fits on a machine where the pod ID matches the machine name
			predicates:    map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: EqualPriority, Weight: 1}},
			nodes:         []string{"machine1", "machine2"},
			pod:           &api.Pod{ObjectMeta: api.ObjectMeta{Name: "machine2"}},
			expectedHosts: sets.NewString("machine2"),
			name:          "test 3",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:         []string{"3", "2", "1"},
			expectedHosts: sets.NewString("3"),
			name:          "test 4",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:         []string{"3", "2", "1"},
			pod:           &api.Pod{ObjectMeta: api.ObjectMeta{Name: "2"}},
			expectedHosts: sets.NewString("2"),
			name:          "test 5",
			wErr:          nil,
		},
		{
			predicates:    map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers:  []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}, {Function: reverseNumericPriority, Weight: 2}},
			nodes:         []string{"3", "2", "1"},
			pod:           &api.Pod{ObjectMeta: api.ObjectMeta{Name: "2"}},
			expectedHosts: sets.NewString("1"),
			name:          "test 6",
			wErr:          nil,
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate, "false": falsePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"3", "2", "1"},
			expectsErr:   true,
			name:         "test 7",
			wErr:         nil,
		},
		{
			predicates: map[string]algorithm.FitPredicate{
				"nopods":  hasNoPodsPredicate,
				"matches": matchesPredicate,
			},
			pods: []*api.Pod{
				{
					ObjectMeta: api.ObjectMeta{Name: "2"},
					Spec: api.PodSpec{
						NodeName: "2",
					},
					Status: api.PodStatus{
						Phase: api.PodRunning,
					},
				},
			},
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Name: "2"}},

			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"1", "2"},
			expectsErr:   true,
			name:         "test 8",
			wErr:         nil,
		},
	}
	for _, test := range tests {
		random := rand.New(rand.NewSource(0))
		cache := schedulercache.New(time.Duration(0), wait.NeverStop)
		for _, pod := range test.pods {
			cache.AddPod(pod)
		}
		for _, name := range test.nodes {
			cache.AddNode(&api.Node{ObjectMeta: api.ObjectMeta{Name: name}})
		}
		scheduler := NewGenericScheduler(cache, test.predicates, test.prioritizers, []algorithm.SchedulerExtender{}, random)
		machine, err := scheduler.Schedule(test.pod, algorithm.FakeNodeLister(makeNodeList(test.nodes)))
		if test.expectsErr {
			if err == nil {
				t.Errorf("Unexpected non-error at %s", test.name)
			}
		} else {
			if !reflect.DeepEqual(err, test.wErr) {
				t.Errorf("Failed : %s, Unexpected error: %v, expected: %v", test.name, err, test.wErr)
			}
			if !test.expectedHosts.Has(machine) {
				t.Errorf("Failed : %s, Expected: %s, Saw: %s", test.name, test.expectedHosts, machine)
			}
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
	_, predicateMap, err := findNodesThatFit(&api.Pod{}, nodeNameToInfo, predicates, makeNodeList(nodes), nil)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(predicateMap) != len(nodes) {
		t.Errorf("unexpected failed predicate map: %v", predicateMap)
	}

	for _, node := range nodes {
		failure, found := predicateMap[node]
		if !found {
			t.Errorf("failed to find node: %s in %v", node, predicateMap)
		}
		if failure != "FakePredicateError" {
			t.Errorf("unexpected failures: %v", failure)
		}
	}
}

func TestFindFitSomeError(t *testing.T) {
	nodes := []string{"3", "2", "1"}
	predicates := map[string]algorithm.FitPredicate{"true": truePredicate, "match": matchesPredicate}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}}
	nodeNameToInfo := map[string]*schedulercache.NodeInfo{
		"3": schedulercache.NewNodeInfo(),
		"2": schedulercache.NewNodeInfo(),
		"1": schedulercache.NewNodeInfo(pod),
	}
	for name := range nodeNameToInfo {
		nodeNameToInfo[name].SetNode(&api.Node{ObjectMeta: api.ObjectMeta{Name: name}})
	}

	_, predicateMap, err := findNodesThatFit(pod, nodeNameToInfo, predicates, makeNodeList(nodes), nil)
	if err != nil && !reflect.DeepEqual(err, algorithmpredicates.ErrFakePredicate) {
		t.Errorf("unexpected error: %v", err)
	}

	if len(predicateMap) != (len(nodes) - 1) {
		t.Errorf("unexpected failed predicate map: %v", predicateMap)
	}

	for _, node := range nodes {
		if node == pod.Name {
			continue
		}
		failure, found := predicateMap[node]
		if !found {
			t.Errorf("failed to find node: %s in %v", node, predicateMap)
		}
		if failure != "FakePredicateError" {
			t.Errorf("unexpected failures: %v", failure)
		}
	}
}
