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
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
)

func falsePredicate(pod *api.Pod, existingPods []*api.Pod, node string) (bool, error) {
	return false, nil
}

func truePredicate(pod *api.Pod, existingPods []*api.Pod, node string) (bool, error) {
	return true, nil
}

func matchesPredicate(pod *api.Pod, existingPods []*api.Pod, node string) (bool, error) {
	return pod.Name == node, nil
}

func hasNoPodsPredicate(pod *api.Pod, existingPods []*api.Pod, node string) (bool, error) {
	return len(existingPods) == 0, nil
}

func numericPriority(pod *api.Pod, podLister algorithm.PodLister, minionLister algorithm.MinionLister) (algorithm.HostPriorityList, error) {
	nodes, err := minionLister.List()
	result := []algorithm.HostPriority{}

	if err != nil {
		return nil, fmt.Errorf("failed to list nodes: %v", err)
	}
	for _, minion := range nodes.Items {
		score, err := strconv.Atoi(minion.Name)
		if err != nil {
			return nil, err
		}
		result = append(result, algorithm.HostPriority{
			Host:  minion.Name,
			Score: score,
		})
	}
	return result, nil
}

func reverseNumericPriority(pod *api.Pod, podLister algorithm.PodLister, minionLister algorithm.MinionLister) (algorithm.HostPriorityList, error) {
	var maxScore float64
	minScore := math.MaxFloat64
	reverseResult := []algorithm.HostPriority{}
	result, err := numericPriority(pod, podLister, minionLister)
	if err != nil {
		return nil, err
	}

	for _, hostPriority := range result {
		maxScore = math.Max(maxScore, float64(hostPriority.Score))
		minScore = math.Min(minScore, float64(hostPriority.Score))
	}
	for _, hostPriority := range result {
		reverseResult = append(reverseResult, algorithm.HostPriority{
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
		list          algorithm.HostPriorityList
		possibleHosts util.StringSet
		expectsErr    bool
	}{
		{
			list: []algorithm.HostPriority{
				{Host: "machine1.1", Score: 1},
				{Host: "machine2.1", Score: 2},
			},
			possibleHosts: util.NewStringSet("machine2.1"),
			expectsErr:    false,
		},
		// equal scores
		{
			list: []algorithm.HostPriority{
				{Host: "machine1.1", Score: 1},
				{Host: "machine1.2", Score: 2},
				{Host: "machine1.3", Score: 2},
				{Host: "machine2.1", Score: 2},
			},
			possibleHosts: util.NewStringSet("machine1.2", "machine1.3", "machine2.1"),
			expectsErr:    false,
		},
		// out of order scores
		{
			list: []algorithm.HostPriority{
				{Host: "machine1.1", Score: 3},
				{Host: "machine1.2", Score: 3},
				{Host: "machine2.1", Score: 2},
				{Host: "machine3.1", Score: 1},
				{Host: "machine1.3", Score: 3},
			},
			possibleHosts: util.NewStringSet("machine1.1", "machine1.2", "machine1.3"),
			expectsErr:    false,
		},
		// empty priorityList
		{
			list:          []algorithm.HostPriority{},
			possibleHosts: util.NewStringSet(),
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
		name         string
		predicates   map[string]algorithm.FitPredicate
		prioritizers []algorithm.PriorityConfig
		nodes        []string
		pod          *api.Pod
		pods         []*api.Pod
		expectedHost string
		expectsErr   bool
	}{
		{
			predicates:   map[string]algorithm.FitPredicate{"false": falsePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: EqualPriority, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			expectsErr:   true,
			name:         "test 1",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: EqualPriority, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			// Random choice between both, the rand seeded above with zero, chooses "machine1"
			expectedHost: "machine1",
			name:         "test 2",
		},
		{
			// Fits on a machine where the pod ID matches the machine name
			predicates:   map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: EqualPriority, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Name: "machine2"}},
			expectedHost: "machine2",
			name:         "test 3",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"3", "2", "1"},
			expectedHost: "3",
			name:         "test 4",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"matches": matchesPredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"3", "2", "1"},
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Name: "2"}},
			expectedHost: "2",
			name:         "test 5",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}, {Function: reverseNumericPriority, Weight: 2}},
			nodes:        []string{"3", "2", "1"},
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Name: "2"}},
			expectedHost: "1",
			name:         "test 6",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate, "false": falsePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"3", "2", "1"},
			expectsErr:   true,
			name:         "test 7",
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
						Phase: api.PodFailed,
					},
				},
				{
					ObjectMeta: api.ObjectMeta{Name: "3"},
					Spec: api.PodSpec{
						NodeName: "2",
					},
					Status: api.PodStatus{
						Phase: api.PodSucceeded,
					},
				},
			},
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Name: "2"}},

			prioritizers: []algorithm.PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"1", "2"},
			expectedHost: "2",
			name:         "test 9",
		},
	}

	for _, test := range tests {
		random := rand.New(rand.NewSource(0))
		scheduler := NewGenericScheduler(test.predicates, test.prioritizers, algorithm.FakePodLister(test.pods), random)
		machine, err := scheduler.Schedule(test.pod, algorithm.FakeMinionLister(makeNodeList(test.nodes)))
		if test.expectsErr {
			if err == nil {
				t.Error("Unexpected non-error")
			}
		} else {
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if test.expectedHost != machine {
				t.Errorf("Failed : %s, Expected: %s, Saw: %s", test.name, test.expectedHost, machine)
			}
		}
	}
}

func TestFindFitAllError(t *testing.T) {
	nodes := []string{"3", "2", "1"}
	predicates := map[string]algorithm.FitPredicate{"true": truePredicate, "false": falsePredicate}
	_, predicateMap, err := findNodesThatFit(&api.Pod{}, algorithm.FakePodLister([]*api.Pod{}), predicates, makeNodeList(nodes))

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
		if len(failures) != 1 || !failures.Has("false") {
			t.Errorf("unexpected failures: %v", failures)
		}
	}
}

func TestFindFitSomeError(t *testing.T) {
	nodes := []string{"3", "2", "1"}
	predicates := map[string]algorithm.FitPredicate{"true": truePredicate, "match": matchesPredicate}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}}
	_, predicateMap, err := findNodesThatFit(pod, algorithm.FakePodLister([]*api.Pod{}), predicates, makeNodeList(nodes))

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
		if len(failures) != 1 || !failures.Has("match") {
			t.Errorf("unexpected failures: %v", failures)
		}
	}
}
