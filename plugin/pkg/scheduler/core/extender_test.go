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
	"fmt"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
	schedulertesting "k8s.io/kubernetes/plugin/pkg/scheduler/testing"
)

type fitPredicate func(pod *v1.Pod, node *v1.Node) (bool, error)
type priorityFunc func(pod *v1.Pod, nodes []*v1.Node) (*schedulerapi.HostPriorityList, error)

type priorityConfig struct {
	function priorityFunc
	weight   int
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

func errorPrioritizerExtender(pod *v1.Pod, nodes []*v1.Node) (*schedulerapi.HostPriorityList, error) {
	return &schedulerapi.HostPriorityList{}, fmt.Errorf("Some error")
}

func machine1PrioritizerExtender(pod *v1.Pod, nodes []*v1.Node) (*schedulerapi.HostPriorityList, error) {
	result := schedulerapi.HostPriorityList{}
	for _, node := range nodes {
		score := 1
		if node.Name == "machine1" {
			score = 10
		}
		result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: score})
	}
	return &result, nil
}

func machine2PrioritizerExtender(pod *v1.Pod, nodes []*v1.Node) (*schedulerapi.HostPriorityList, error) {
	result := schedulerapi.HostPriorityList{}
	for _, node := range nodes {
		score := 1
		if node.Name == "machine2" {
			score = 10
		}
		result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: score})
	}
	return &result, nil
}

func machine2Prioritizer(_ *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	result := []schedulerapi.HostPriority{}
	for _, node := range nodes {
		score := 1
		if node.Name == "machine2" {
			score = 10
		}
		result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: score})
	}
	return result, nil
}

type FakeExtender struct {
	predicates       []fitPredicate
	prioritizers     []priorityConfig
	weight           int
	nodeCacheCapable bool
	filteredNodes    []*v1.Node
}

func (f *FakeExtender) Filter(pod *v1.Pod, nodes []*v1.Node, nodeNameToInfo map[string]*schedulercache.NodeInfo) ([]*v1.Node, schedulerapi.FailedNodesMap, error) {
	filtered := []*v1.Node{}
	failedNodesMap := schedulerapi.FailedNodesMap{}
	for _, node := range nodes {
		fits := true
		for _, predicate := range f.predicates {
			fit, err := predicate(pod, node)
			if err != nil {
				return []*v1.Node{}, schedulerapi.FailedNodesMap{}, err
			}
			if !fit {
				fits = false
				break
			}
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

func (f *FakeExtender) Prioritize(pod *v1.Pod, nodes []*v1.Node) (*schedulerapi.HostPriorityList, int, error) {
	result := schedulerapi.HostPriorityList{}
	combinedScores := map[string]int{}
	for _, prioritizer := range f.prioritizers {
		weight := prioritizer.weight
		if weight == 0 {
			continue
		}
		priorityFunc := prioritizer.function
		prioritizedList, err := priorityFunc(pod, nodes)
		if err != nil {
			return &schedulerapi.HostPriorityList{}, 0, err
		}
		for _, hostEntry := range *prioritizedList {
			combinedScores[hostEntry.Host] += hostEntry.Score * weight
		}
	}
	for host, score := range combinedScores {
		result = append(result, schedulerapi.HostPriority{Host: host, Score: score})
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

var _ algorithm.SchedulerExtender = &FakeExtender{}

func TestGenericSchedulerWithExtenders(t *testing.T) {
	tests := []struct {
		name                 string
		predicates           map[string]algorithm.FitPredicate
		prioritizers         []algorithm.PriorityConfig
		extenders            []FakeExtender
		extenderPredicates   []fitPredicate
		extenderPrioritizers []priorityConfig
		nodes                []string
		expectedHost         string
		expectsErr           bool
	}{
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
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
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
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
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			extenders: []FakeExtender{
				{
					predicates: []fitPredicate{truePredicateExtender},
				},
				{
					predicates: []fitPredicate{machine1PredicateExtender},
				},
			},
			nodes:        []string{"machine1", "machine2"},
			expectedHost: "machine1",
			name:         "test 3",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
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
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
			extenders: []FakeExtender{
				{
					predicates:   []fitPredicate{truePredicateExtender},
					prioritizers: []priorityConfig{{errorPrioritizerExtender, 10}},
					weight:       1,
				},
			},
			nodes:        []string{"machine1"},
			expectedHost: "machine1",
			name:         "test 5",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}},
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
			nodes:        []string{"machine1", "machine2"},
			expectedHost: "machine2",
			name:         "test 6",
		},
		{
			predicates:   map[string]algorithm.FitPredicate{"true": truePredicate},
			prioritizers: []algorithm.PriorityConfig{{Function: machine2Prioritizer, Weight: 20}},
			extenders: []FakeExtender{
				{
					predicates:   []fitPredicate{truePredicateExtender},
					prioritizers: []priorityConfig{{machine1PrioritizerExtender, 10}},
					weight:       1,
				},
			},
			nodes:        []string{"machine1", "machine2"},
			expectedHost: "machine2", // machine2 has higher score
			name:         "test 7",
		},
	}

	for _, test := range tests {
		extenders := []algorithm.SchedulerExtender{}
		for ii := range test.extenders {
			extenders = append(extenders, &test.extenders[ii])
		}
		cache := schedulercache.New(time.Duration(0), wait.NeverStop)
		for _, name := range test.nodes {
			cache.AddNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}})
		}
		scheduler := NewGenericScheduler(
			cache, nil, test.predicates, algorithm.EmptyPredicateMetadataProducer, test.prioritizers, algorithm.EmptyMetadataProducer, extenders)
		podIgnored := &v1.Pod{}
		machine, err := scheduler.Schedule(podIgnored, schedulertesting.FakeNodeLister(makeNodeList(test.nodes)))
		if test.expectsErr {
			if err == nil {
				t.Errorf("Unexpected non-error for %s, machine %s", test.name, machine)
			}
		} else {
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				continue
			}
			if test.expectedHost != machine {
				t.Errorf("Failed : %s, Expected: %s, Saw: %s", test.name, test.expectedHost, machine)
			}
		}
	}
}
