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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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

func numericPriority(pod *api.Pod, podLister PodLister, minionLister MinionLister) (HostPriorityList, error) {
	nodes, err := minionLister.List()
	result := []HostPriority{}

	if err != nil {
		fmt.Errorf("failed to list nodes: %v", err)
		return nil, err
	}
	for _, minion := range nodes.Items {
		score, err := strconv.Atoi(minion.Name)
		if err != nil {
			return nil, err
		}
		result = append(result, HostPriority{
			host:  minion.Name,
			score: score,
		})
	}
	return result, nil
}

func reverseNumericPriority(pod *api.Pod, podLister PodLister, minionLister MinionLister) (HostPriorityList, error) {
	var maxScore float64
	minScore := math.MaxFloat64
	reverseResult := []HostPriority{}
	result, err := numericPriority(pod, podLister, minionLister)
	if err != nil {
		return nil, err
	}

	for _, hostPriority := range result {
		maxScore = math.Max(maxScore, float64(hostPriority.score))
		minScore = math.Min(minScore, float64(hostPriority.score))
	}
	for _, hostPriority := range result {
		reverseResult = append(reverseResult, HostPriority{
			host:  hostPriority.host,
			score: int(maxScore + minScore - float64(hostPriority.score)),
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
		list          HostPriorityList
		possibleHosts util.StringSet
		expectsErr    bool
	}{
		{
			list: []HostPriority{
				{host: "machine1.1", score: 1},
				{host: "machine2.1", score: 2},
			},
			possibleHosts: util.NewStringSet("machine2.1"),
			expectsErr:    false,
		},
		// equal scores
		{
			list: []HostPriority{
				{host: "machine1.1", score: 1},
				{host: "machine1.2", score: 2},
				{host: "machine1.3", score: 2},
				{host: "machine2.1", score: 2},
			},
			possibleHosts: util.NewStringSet("machine1.2", "machine1.3", "machine2.1"),
			expectsErr:    false,
		},
		// out of order scores
		{
			list: []HostPriority{
				{host: "machine1.1", score: 3},
				{host: "machine1.2", score: 3},
				{host: "machine2.1", score: 2},
				{host: "machine3.1", score: 1},
				{host: "machine1.3", score: 3},
			},
			possibleHosts: util.NewStringSet("machine1.1", "machine1.2", "machine1.3"),
			expectsErr:    false,
		},
		// empty priorityList
		{
			list:          []HostPriority{},
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
		predicates   map[string]FitPredicate
		prioritizers []PriorityConfig
		nodes        []string
		pod          *api.Pod
		expectedHost string
		expectsErr   bool
	}{
		{
			predicates:   map[string]FitPredicate{"false": falsePredicate},
			prioritizers: []PriorityConfig{{Function: EqualPriority, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			expectsErr:   true,
			name:         "test 1",
		},
		{
			predicates:   map[string]FitPredicate{"true": truePredicate},
			prioritizers: []PriorityConfig{{Function: EqualPriority, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			// Random choice between both, the rand seeded above with zero, chooses "machine1"
			expectedHost: "machine1",
			name:         "test 2",
		},
		{
			// Fits on a machine where the pod ID matches the machine name
			predicates:   map[string]FitPredicate{"matches": matchesPredicate},
			prioritizers: []PriorityConfig{{Function: EqualPriority, Weight: 1}},
			nodes:        []string{"machine1", "machine2"},
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Name: "machine2"}},
			expectedHost: "machine2",
			name:         "test 3",
		},
		{
			predicates:   map[string]FitPredicate{"true": truePredicate},
			prioritizers: []PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"3", "2", "1"},
			expectedHost: "3",
			name:         "test 4",
		},
		{
			predicates:   map[string]FitPredicate{"matches": matchesPredicate},
			prioritizers: []PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"3", "2", "1"},
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Name: "2"}},
			expectedHost: "2",
			name:         "test 5",
		},
		{
			predicates:   map[string]FitPredicate{"true": truePredicate},
			prioritizers: []PriorityConfig{{Function: numericPriority, Weight: 1}, {Function: reverseNumericPriority, Weight: 2}},
			nodes:        []string{"3", "2", "1"},
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Name: "2"}},
			expectedHost: "1",
			name:         "test 6",
		},
		{
			predicates:   map[string]FitPredicate{"true": truePredicate, "false": falsePredicate},
			prioritizers: []PriorityConfig{{Function: numericPriority, Weight: 1}},
			nodes:        []string{"3", "2", "1"},
			expectsErr:   true,
			name:         "test 7",
		},
	}

	for _, test := range tests {
		random := rand.New(rand.NewSource(0))
		scheduler := NewGenericScheduler(test.predicates, test.prioritizers, FakePodLister([]*api.Pod{}), random)
		machine, err := scheduler.Schedule(test.pod, FakeMinionLister(makeNodeList(test.nodes)))
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
	predicates := map[string]FitPredicate{"true": truePredicate, "false": falsePredicate}
	_, predicateMap, err := findNodesThatFit(&api.Pod{}, FakePodLister([]*api.Pod{}), predicates, makeNodeList(nodes))

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
	predicates := map[string]FitPredicate{"true": truePredicate, "match": matchesPredicate}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}}
	_, predicateMap, err := findNodesThatFit(pod, FakePodLister([]*api.Pod{}), predicates, makeNodeList(nodes))

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
